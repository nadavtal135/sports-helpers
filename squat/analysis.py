"""Measure standing-to-standing squat attempts from one side of a COCO-17 pose."""

from dataclasses import dataclass, field
import math

import numpy as np

from running.src.side_view import JOINTS, choose_side, observations
from running.src.signals import fill_short_gaps, knee_flexion, segments, smooth_runs
from .settings import Settings


@dataclass
class Attempt:
    first: int                         # visible review interval, in trimmed frames
    last: int
    start: int | None                  # observed landmarks only
    end: int | None
    status: str
    reason: str = ""
    bottom: int | None = None
    peak_flexion: float | None = None
    peak_frame: int | None = None
    torso_lean: float | None = None
    depth: str = "unknown"
    hip_knee_height: float | None = None    # signed image y difference, in leg lengths
    bottom_span: tuple[int, int] | None = None   # frames [a, b) of the deepest bottom plateau


@dataclass
class Analysis:
    points: np.ndarray                # observed pixel coordinates, also used for rendering
    phases: np.ndarray
    side: str | None
    coverage: dict[str, float]
    attempts: list[Attempt] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    capture_note: str = ""
    capture_frame: int | None = None


def _windows(standing, flexion, hold):
    """Departures and returns to sustained standing delimit entire excursions.

    Inside a window, repeated visits to the bottom are still one attempt. A
    gap keeps the window open for review but disqualifies it from the count.
    """
    previous = None
    for a, b, is_top in segments(standing):
        if not is_top or b - a < hold:
            continue
        if a == 0 or b == len(standing):
            # A clip can cut through a moving, nearly straight leg. Look for a
            # brief settled pose at the boundary before accepting it as standing.
            samples = np.lib.stride_tricks.sliding_window_view(flexion[a:b], hold)
            if np.min(np.ptp(samples, axis=1)) > 2.5:
                continue
        if a > 0:
            yield previous, a
        previous = b - 1
    if previous != len(standing) - 1:
        yield previous, None


def _bottom(drop, flexion):
    """Find a supported reversal and its bottom phase within a gap-free window."""
    peak = int(np.argmax(drop))
    # Both directions must be visible. The last frame of a truncated descent
    # (or the first frame of an ascent) is only the deepest *visible* point.
    above_bottom = (drop < drop[peak] - 0.015) & (flexion < flexion[peak] - 5)
    if not above_bottom[:peak].any() or not above_bottom[peak + 1:].any():
        return None
    near = (drop >= drop[peak] - 0.012) & (flexion >= flexion[peak] - 6)
    # Keep a small rebound between two bottom observations inside the hold.
    basin = (drop >= drop[peak] - 0.075) & (flexion >= flexion[peak] - 12)
    lo, hi = next((a, b) for a, b, on in segments(basin) if on and a <= peak < b)
    near_frames = lo + np.flatnonzero(near[lo:hi])
    a, b = int(near_frames[0]), int(near_frames[-1]) + 1
    # On a hold, use the middle of the deepest plateau. Stay within one
    # contiguous plateau so a bounce cannot place the measurement on its crest.
    flat = drop[a:b] >= drop[peak] - 0.001
    left, right = next((a + i, a + j) for i, j, on in segments(flat) if on and a + i <= peak < a + j)
    return a, (left + right - 1) // 2, b, (left, right)


def _measure_window(start, end, drop, flexion, bent, phases, hold):
    lo = 0 if start is None else start
    hi = len(drop) - 1 if end is None else end
    window = slice(lo, hi + 1)
    if not any(on and b - a >= hold for a, b, on in segments(bent[window])):
        return None
    lost = ~np.isfinite(drop[window])
    seen = np.flatnonzero(~lost)
    first, last = lo + int(seen[0]), lo + int(seen[-1])
    if start is None and end is None and not lost.any():
        if np.ptp(drop[window]) < 0.015 or np.ptp(flexion[window]) < 5:
            return None   # a stationary bent pose alone is insufficient evidence of an attempt
    if lost.any():
        gaps = np.flatnonzero(lost)
        # A standing pose on the other side of a gap cannot time a transition.
        if not bent[lo:lo + gaps[0]].any():
            start = None
        if not bent[lo + gaps[-1] + 1:hi + 1].any():
            end = None
        phases[window] = "uncertain"
        return Attempt(first, last, start, end, "uncertain",
                       "A tracking gap obscures part of this interval.")

    if start is None or end is None:
        if start is None and end is None:
            reason = "Standing was cut off at both clip boundaries."
        elif start is None:
            reason = "The clip starts before a standing position can be confirmed."
        else:
            reason = "The clip ends before a return to standing can be confirmed."
        attempt = Attempt(first, last, start, end, "incomplete", reason)
    else:
        attempt = Attempt(first, last, start, end, "completed")

    bottom = _bottom(drop[window], flexion[window])
    if bottom is None:
        if attempt.status == "completed":
            attempt.status, attempt.reason = "uncertain", "The bottom reversal could not be confirmed."
        elif start is not None:
            phases[window] = "descent"
        elif end is not None:
            phases[window] = "ascent"
        return attempt

    a, middle, b, (left, right) = bottom
    phases[lo:lo + a] = "descent"
    phases[lo + a:lo + b] = "bottom"
    phases[lo + b:hi + 1] = "ascent"
    attempt.bottom = lo + middle
    # A real pause at the bottom (near-bottom for at least 0.3 s) is measured over the
    # whole pause; otherwise only the deepest plateau counts (see _at_bottom). The
    # flat plateau alone shrinks to a frame or two under keypoint jitter.
    paused = b - a >= 3 * hold
    attempt.bottom_span = (lo + a, lo + b) if paused else (lo + left, lo + right)
    return attempt


def _lean(hip, shoulder):
    return math.degrees(math.atan2(abs(shoulder[0] - hip[0]), hip[1] - shoulder[1]))


def _at_bottom(value, ok, k, span, reach):
    """A bottom measurement that is not at the mercy of one frame's keypoint jitter.

    During a held bottom (at least 5 frames) the lifter is still, so the median
    over the hold. Otherwise a quadratic through the observed frames within
    *reach* of the bottom, read at the bottom frame: the motion is smooth there,
    so the fit averages the jitter without the bias a median of the shallower
    neighbouring frames would add. Falls back to the bottom frame alone.
    """
    a, b = span
    hold = [i for i in range(a, b) if ok(i)]
    if b - a >= 5 and len(hold) >= 3:
        return float(np.median([value(i) for i in hold]))
    near = [i for i in range(k - reach, k + reach + 1) if ok(i)]
    if len(near) >= 5 and near[0] < k < near[-1]:
        t = np.array(near, float) - k
        return float(np.polyval(np.polyfit(t, [value(i) for i in near], 2), 0.0))
    return float(value(k))


def _metrics(attempt, leg, visible, flexion, length, settings, fps=30.0):
    """Depth and torso lean at the bottom.

    The cited bottom frame must show the needed keypoints, or the measurement is
    withheld (as before). When it does, the value comes from ``_at_bottom``: on
    test clips 2 px of keypoint jitter moved a level rep into "below" when it
    was read off one frame.
    """
    k = attempt.bottom
    if k is None:
        return
    span = attempt.bottom_span or (k, k + 1)
    reach = max(2, round(0.1 * fps))
    n = len(leg)

    def seen(i, joints):
        return 0 <= i < n and visible[i, joints].all() and np.isfinite(leg[i, joints]).all()

    if visible[k, :2].all():
        dy = _at_bottom(lambda i: (leg[i, 0, 1] - leg[i, 1, 1]) / length,   # image y grows downwards
                        lambda i: seen(i, [0, 1]), k, span, reach)
        attempt.hip_knee_height = round(float(dy), 4)
        if abs(dy) <= settings.level_tolerance:
            attempt.depth = "roughly level"
        else:
            attempt.depth = "below" if dy > 0 else "above"
    if visible[k, 0] and visible[k, 3] and np.linalg.norm(leg[k, 3] - leg[k, 0]) > 1:
        lean = _at_bottom(lambda i: _lean(leg[i, 0], leg[i, 3]),
                          lambda i: seen(i, [0, 3]) and np.linalg.norm(leg[i, 3] - leg[i, 0]) > 1,
                          k, span, reach)
        attempt.torso_lean = round(lean, 1)
    if attempt.status == "completed":
        i = attempt.start + int(np.argmax(flexion[attempt.start:attempt.end + 1]))
        attempt.peak_frame = i    # frame at which visibility of the peak was checked
        if visible[i, :3].all():
            attempt.peak_flexion = round(float(flexion[i]), 1)


def analyze(frames, fps: float, n: int, *, width: int, height: int, settings: Settings) -> Analysis:
    """Use aspect-correct coordinates and keep missing observations separate from smoothing."""
    s = settings
    points, scores = observations(frames, n, width, height, s.min_joint_score)
    side, coverage = choose_side(points, scores)
    phases = np.full(n, "unconfirmed", dtype=object)
    result = Analysis(points, phases, side, coverage)
    if side is None:
        phases[:] = "tracking gap"
        result.capture_note = ("Depth, angles and tempo are unavailable because no leg could be tracked. "
                               "Film from the side with the near hip, knee and ankle in frame.")
        result.capture_frame = 0
        result.warnings.append(result.capture_note)
        return result

    leg = points[:, JOINTS[side]]
    hip, knee, ankle, _ = leg.transpose(1, 0, 2)
    lengths = np.linalg.norm(np.diff(leg[:, :3], axis=1), axis=2)
    observed = (lengths > 1).all(axis=1)
    length = np.median(lengths[observed], axis=0).sum()
    gap_limit = int(fps * s.max_interp_gap_s + 1e-6)
    smooth = max(3, round(fps * s.smoothing_s))

    def signal(values):
        values = np.where(observed, values, np.nan)
        return smooth_runs(fill_short_gaps(values, gap_limit), smooth, smooth)

    flexion = signal(knee_flexion(hip.T, knee.T, ankle.T))
    height_above_ankle = signal((ankle[:, 1] - hip[:, 1]) / length)
    # Calibrate on near-extended poses only. A clip entirely at the bottom
    # cannot establish a standing reference by treating its highest hip as up.
    extended = observed & (flexion <= s.standing_flexion_limit_deg) & (height_above_ankle > 0.7)
    top, straight = 1.0, 0.0
    if extended.any():
        top = np.percentile(height_above_ankle[extended], 80)
        upright = extended & (height_above_ankle >= top - s.standing_height_band)
        straight = np.percentile(flexion[upright], 20)
    drop = top - height_above_ankle
    standing = extended & (drop <= s.standing_height_band) & (flexion <= straight + s.standing_flexion_band_deg)
    bent = (drop >= s.min_hip_drop) & (flexion >= straight + s.min_flexion_change_deg)
    hold = max(1, math.ceil(fps * s.standing_hold_s))
    metric_visible = scores[:, JOINTS[side]] >= s.metric_joint_score
    for start, end in _windows(standing, flexion, hold):
        attempt = _measure_window(start, end, drop, flexion, bent, phases, hold)
        if attempt is not None:
            _metrics(attempt, leg, metric_visible, flexion, length, s, fps)
            result.attempts.append(attempt)

    phases[standing & (phases == "unconfirmed")] = "standing"
    for a in result.attempts:
        if a.start is not None:
            phases[a.start] = "standing"
        if a.end is not None:
            phases[a.end] = "standing"
    lost = ~np.isfinite(drop)
    phases[lost] = "tracking gap"
    if lost.any():
        result.warnings.append(f"The {side} leg lost tracking for {lost.sum() / fps:.2f} s. "
                               "Reps spanning these gaps cannot be counted.")
    if any(a.status == "incomplete" for a in result.attempts):
        result.warnings.append("A clip boundary cuts an attempt short. Include standing pauses when trimming.")
    if any(a.bottom is not None and (a.depth == "unknown" or a.torso_lean is None)
           for a in result.attempts):
        result.warnings.append("Some bottom measurements lack a clear hip, knee or shoulder keypoint.")
    if not any(a.status == "completed" for a in result.attempts):
        result.warnings.append("No full standing-to-standing rep was confirmed.")
    ankle_motion = np.ptp(np.percentile(ankle[observed], [5, 95], axis=0), axis=0)
    if np.linalg.norm(ankle_motion) > 0.18 * length:
        shift = np.linalg.norm(ankle - np.median(ankle[observed], axis=0), axis=1)
        result.capture_frame = int(np.nanargmax(np.where(observed, shift, np.nan)))
        result.capture_note = ("The ankle shifts substantially in the image, which makes depth and angle "
                               "comparisons uncertain. Keep the camera still and level beside the lifter.")
    shoulders = points[:, [JOINTS["left"][3], JOINTS["right"][3]]]
    hips = points[:, [JOINTS["left"][0], JOINTS["right"][0]]]
    wide = ((np.abs(shoulders[:, 0, 0] - shoulders[:, 1, 0]) > 0.40 * length)
            & (np.abs(hips[:, 0, 0] - hips[:, 1, 0]) > 0.20 * length))
    if not result.capture_note and np.mean(wide & observed) > 0.5:
        frames_wide = np.flatnonzero(wide & observed)
        result.capture_frame = int(frames_wide[len(frames_wide) // 2])
        result.capture_note = ("Both hips and shoulders appear widely separated in the image. The view makes "
                               "depth and angle comparisons uncertain. Film directly from beside the lifter.")
    if result.capture_note:
        result.warnings.append(result.capture_note)
    return result

"""Measure standing-to-standing squat attempts from one side of a COCO-17 pose."""

from dataclasses import dataclass, field
import math

import numpy as np

from running.src import pose
from running.src.signals import fill_short_gaps, knee_flexion, segments, smooth_runs
from running.src.skeleton import KPT_NAMES
from .settings import Settings

JOINTS = {side: [KPT_NAMES.index(f"{side}_{j}") for j in ("hip", "knee", "ankle", "shoulder")]
          for side in ("left", "right")}


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
    torso_lean: float | None = None
    depth: str = "unknown"


@dataclass
class Analysis:
    points: np.ndarray                # observed pixel coordinates, also used for rendering
    phases: np.ndarray
    side: str | None
    coverage: dict[str, float]
    attempts: list[Attempt] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _observations(frames, n, width, height, min_score):
    points = pose.keypoints(frames, n, width, height)
    # Older replies can omit joint scores. In that case use coordinate visibility.
    scores = np.ones((n, len(KPT_NAMES)))
    for frame in frames:
        i = frame["index"]
        if 0 <= i < n and frame["persons"]:
            values = frame["persons"][0].get("kpts_score")
            if values is not None:
                scores[i] = values if np.shape(values) == (len(KPT_NAMES),) else 0.0
    valid = ((points >= 0) & (points < [width, height])).all(axis=2)
    valid &= np.isfinite(scores) & (scores >= min_score)
    points[~valid] = np.nan
    scores[~valid] = 0.0
    return points, scores


def _choose_side(points, scores):
    quality, coverage = {}, {}
    for side, (hip, knee, ankle, shoulder) in JOINTS.items():
        leg = points[:, [hip, knee, ankle]]
        lengths = np.linalg.norm(np.diff(leg, axis=1), axis=2)
        seen = (lengths > 1).all(axis=1)
        coverage[side] = float(seen.mean())
        # A usable leg is essential. Shoulder visibility breaks close calls.
        support = scores[:, [hip, knee, ankle]].min(axis=1) * (0.9 + 0.1 * scores[:, shoulder])
        quality[side] = float(np.mean(seen * support))
    return max(quality, key=quality.get) if any(quality.values()) else None, coverage


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
    return a, (left + right - 1) // 2, b


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

    a, middle, b = bottom
    phases[lo:lo + a] = "descent"
    phases[lo + a:lo + b] = "bottom"
    phases[lo + b:hi + 1] = "ascent"
    attempt.bottom = lo + middle
    return attempt


def _metrics(attempt, leg, visible, flexion, length, settings):
    k = attempt.bottom
    if k is None:
        return
    hip, knee, _, shoulder = leg[k]
    if visible[k, :2].all():
        dy = (hip[1] - knee[1]) / length   # image y increases downwards
        if abs(dy) <= settings.level_tolerance:
            attempt.depth = "roughly level"
        else:
            attempt.depth = "below" if dy > 0 else "above"
    if visible[k, 0] and visible[k, 3] and np.linalg.norm(shoulder - hip) > 1:
        attempt.torso_lean = round(math.degrees(math.atan2(abs(shoulder[0] - hip[0]),
                                                          hip[1] - shoulder[1])), 1)
    if attempt.status == "completed":
        i = attempt.start + int(np.argmax(flexion[attempt.start:attempt.end + 1]))
        if visible[i, :3].all():
            attempt.peak_flexion = round(float(flexion[i]), 1)


def analyze(frames, fps: float, n: int, *, width: int, height: int, settings: Settings) -> Analysis:
    """Use aspect-correct coordinates and keep missing observations separate from smoothing."""
    s = settings
    points, scores = _observations(frames, n, width, height, s.min_joint_score)
    side, coverage = _choose_side(points, scores)
    phases = np.full(n, "unconfirmed", dtype=object)
    result = Analysis(points, phases, side, coverage)
    if side is None:
        phases[:] = "tracking gap"
        result.warnings.append("No leg could be tracked. Keep one hip, knee and ankle visible side-on.")
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
            _metrics(attempt, leg, metric_visible, flexion, length, s)
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
        result.warnings.append("The ankle shifts substantially in the image. Check camera movement and framing.")
    return result

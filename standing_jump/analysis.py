"""Find side-view jumps from observed support, airborne movement and recovery.

All landmarks here are local CFR frame indices. Only the exporter adds the trim offset.
"""

from dataclasses import dataclass, field

import numpy as np

from running.src.side_view import JOINTS, choose_side, observations
from running.src.signals import knee_flexion, segments, smooth_runs
from .settings import Settings

LANDMARKS = ("dip_start", "countermovement", "takeoff", "flight", "contact", "landing_bottom",
             "recovery", "step_lift", "step_plant")
METRICS = ("descent_s", "flight_time_estimate_s", "dip_knee_deg", "takeoff_knee_deg",
           "contact_knee_deg", "landing_knee_deg", "landing_bend_deg", "dip_torso_deg", "landing_torso_deg")


@dataclass
class Attempt:
    first: int
    last: int
    frames: dict = field(default_factory=lambda: dict.fromkeys(LANDMARKS))
    metrics: dict = field(default_factory=lambda: dict.fromkeys(METRICS))
    issues: list = field(default_factory=list)
    visible_landing_bend: bool | None = None
    rebounds_into_next: bool = False
    step: str = "unknown"
    methods: dict = field(default_factory=dict)

    def issue(self, phase, frame, detail):
        self.issues.append({"phase": phase, "frame": int(frame), "detail": detail})

    @property
    def confirmed(self):
        return (all(self.frames[k] is not None for k in ("takeoff", "flight", "contact"))
                and not any(issue["phase"] == "flight" for issue in self.issues))


@dataclass
class Analysis:
    points: np.ndarray
    phases: np.ndarray
    side: str | None
    attempts: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def runs(mask, minimum=1):
    return [(a, b) for a, b, on in segments(mask) if on and b - a >= minimum]


def floor_reference(y, scale, fps):
    """Find a sustained, nearly stationary ankle at its lower image level."""
    visible = y[np.isfinite(y)]
    hold = max(2, round(0.13 * fps))
    if len(visible) < hold:
        return np.nan
    lower = np.percentile(visible, 90)
    quiet = np.abs(np.diff(y, prepend=y[0])) < 0.012 * scale
    spans = runs(quiet & (np.abs(y - lower) < 0.04 * scale), hold)
    if not spans:
        return np.nan
    return float(np.median(np.concatenate([y[a:b] for a, b in spans])))


@dataclass
class Signals:
    fps: float
    scale: float
    hip_y: np.ndarray
    knee: np.ndarray
    torso: np.ndarray
    ankles: np.ndarray
    clearance: np.ndarray
    grounded: np.ndarray
    standing_y: float
    standing_knee: float


def make_signals(points, side, width, height, fps, cfg):
    hip, knee, ankle, shoulder = points[:, JOINTS[side]].transpose(1, 0, 2)
    lengths = np.linalg.norm(hip - knee, axis=1) + np.linalg.norm(knee - ankle, axis=1)
    scale = float(np.nanmedian(lengths))
    ankles = points[:, [JOINTS["left"][2], JOINTS["right"][2]]].copy()
    x, y = ankles.transpose(2, 0, 1)
    # An ankle on the crop edge cannot show the shoe meeting the floor.
    cropped = (y > height - 0.085 * scale) | (x < 0.055 * scale) | (x > width - 0.055 * scale)
    ankles[cropped] = np.nan
    levels = np.array([floor_reference(col, scale, fps) for col in y.T])
    clearance = (levels - y) / scale
    grounded = clearance <= cfg.ground_band
    flex = knee_flexion(hip.T, knee.T, ankle.T)
    torso = np.degrees(np.arctan2(np.abs(shoulder[:, 0] - hip[:, 0]), hip[:, 1] - shoulder[:, 1]))
    torso[np.linalg.norm(hip - shoulder, axis=1) < 1] = np.nan
    hy = smooth_runs(hip[:, 1], max(1, round(0.07 * fps)), 1)
    supported = grounded.any(axis=1) & np.isfinite(hy) & np.isfinite(flex)
    extended = supported & (flex < 38)
    if not runs(extended, max(2, round(0.10 * fps))):
        return None
    return Signals(fps, scale, hy, flex, torso, ankles, clearance, grounded,
                   float(np.percentile(hy[extended], 25)), float(np.percentile(flex[extended], 25)))


def oblique_view(points, s):
    # Assess bilateral separation only where both shoulders and hips are tracked.
    # A hidden far limb is common in a true side view and is not evidence of rotation.
    spans = np.abs(points[:, JOINTS["left"], 0] - points[:, JOINTS["right"], 0]) / s.scale
    hips, shoulders = spans[:, 0], spans[:, 3]
    seen = np.isfinite(hips + shoulders) & s.grounded.any(axis=1)
    return (seen.sum() >= max(4, round(0.25 * s.fps))
            and np.mean((hips[seen] > 0.15) & (shoulders[seen] > 0.23)) >= 0.65)


def excursions(s, cfg):
    """Gaps between sustained support. A dip or a one-foot step alone stays grounded."""
    n = len(s.hip_y)
    ground_runs = runs(s.grounded.any(axis=1), max(2, round(0.065 * s.fps)))
    previous = 0
    known = np.isfinite(s.clearance)
    raised = known.any(axis=1) & np.all(~known | (s.clearance >= cfg.flight_clearance), axis=1)
    hip_up = s.hip_y < s.standing_y - 0.04 * s.scale
    hold = max(2, round(0.055 * s.fps))
    for start, end in [*ground_runs, (n, n)]:
        lo, hi = previous, start
        previous = end
        if hi - lo < hold:
            continue
        # Require support evidence from the body as well as the lifting ankles.
        evidence = runs((raised & hip_up)[lo:hi], hold)
        if evidence:
            a, b = max(evidence, key=lambda span: span[1] - span[0])
            section = s.clearance[lo + a:lo + b]
            lowest_foot = np.where(np.isfinite(section), section, np.inf).min(axis=1)
            peak = int(lo + a + np.argmax(lowest_foot))
        elif runs((~known.any(axis=1) & hip_up)[lo:hi], hold):
            peak = None   # body moved above its standing level while both feet were lost
        else:
            continue
        yield lo, hi, peak


def preparation(s, lo, takeoff_bound):
    """Last visible hip dip and its standing departure before the flight window."""
    first = max(lo, takeoff_bound - round(2 * s.fps))
    y = s.hip_y[first:takeoff_bound]
    bend = s.knee[first:takeoff_bound]
    dipping = (y > s.standing_y + 0.035 * s.scale) & (bend > s.standing_knee + 12)
    dips = runs(dipping, max(2, round(0.065 * s.fps)))
    if not dips:
        return None, None
    a, b = dips[-1]
    if len(y) - b > 0.50 * s.fps:
        return None, None
    bottom = int(a + np.argmax(y[a:b]))
    if bottom == 0 or not np.isfinite(y[bottom:]).all() or not np.any(y[bottom:] < y[bottom] - 0.02 * s.scale):
        return None, None
    upright = (y[:bottom] < s.standing_y + 0.018 * s.scale) & (bend[:bottom] < s.standing_knee + 9)
    holds = runs(upright, max(2, round(0.075 * s.fps)))
    start = first + holds[-1][1] - 1 if holds else None
    if start is not None and not np.isfinite(s.hip_y[start:first + bottom + 1]).all():
        start = None
    return start, first + bottom


def recovery_step(s, start, stop, cfg):
    """One ankle lifts, changes horizontal position and replants beside a stationary support."""
    for foot in (0, 1):
        other = 1 - foot
        clear = s.clearance[start:stop, foot]
        for a, b in runs(clear > cfg.ground_band, max(2, round(0.07 * s.fps))):
            if a == 0 or b == len(clear):
                continue
            first, last = start + a - 1, start + b
            span = slice(first, last + 1)
            if not (s.grounded[first, foot] and s.grounded[last, foot] and s.grounded[span, other].all()):
                continue
            x = s.ankles[span, :, 0]
            if (np.max(clear[a:b]) >= 0.075 and abs(x[-1, foot] - x[0, foot]) >= 0.12 * s.scale
                    and np.ptp(x[:, other]) < 0.055 * s.scale):
                return start + a + int(np.argmax(clear[a:b])), last
    return None, None


def settled_stance(s, begin, stop, allow_shift):
    upright = (s.knee < s.standing_knee + 13) & (s.hip_y < s.standing_y + 0.05 * s.scale)
    # Settling requires still feet and a stable hip as well as a near-standing pose.
    ankle_dx = np.abs(np.diff(s.ankles[:, :, 0], axis=0, prepend=s.ankles[:1, :, 0])).max(axis=1)
    hip_dy = np.abs(np.diff(s.hip_y, prepend=s.hip_y[0]))
    quiet = upright & (ankle_dx < 0.015 * s.scale) & (hip_dy < 0.012 * s.scale)
    hold = max(3, round(0.15 * s.fps))
    holds = runs((quiet & s.grounded.all(axis=1))[begin:stop], hold)
    if holds:
        return begin + holds[0][0] + hold - 1
    if allow_shift:
        # A replant at a different spot can change the apparent floor level.
        # Only accept a new stationary stance after an observed contact.
        dy = np.abs(np.diff(s.ankles[:, :, 1], axis=0, prepend=s.ankles[:1, :, 1])).max(axis=1)
        quiet &= (dy < 0.012 * s.scale) & (s.clearance < 0.11).all(axis=1)
        for lo, hi in runs(quiet[begin:stop], hold):
            for end in range(begin + lo + hold, begin + hi + 1):
                soles = s.ankles[end - hold:end]
                if np.ptp(soles, axis=0).max() < 0.035 * s.scale:
                    return end - 1
    return None


def landing(s, a, ground_start, stop, cfg, oblique):
    """Look for observed replanting, settling, and the first supported landing bend."""
    f = a.frames
    stop = min(stop, ground_start + round(cfg.recovery_window_s * s.fps) + 1)
    if stop <= ground_start:
        return
    f["step_lift"], f["step_plant"] = recovery_step(s, ground_start, stop, cfg)
    begin = f["step_plant"] if f["step_plant"] is not None else ground_start
    f["recovery"] = settled_stance(s, begin, stop, f["contact"] is not None)
    a.last = max(ground_start, stop - 1)
    if f["step_plant"] is not None:
        a.step = "observed"
    elif f["contact"] is not None and f["recovery"] is not None:
        feet = s.ankles[ground_start:f["recovery"] + 1]
        if np.isfinite(feet).all() and np.ptp(feet[:, :, 0], axis=0).max() < 0.12 * s.scale:
            a.step = "no visible step"
    if f["contact"] is None:
        return
    # A late step or a new countermovement must not become the landing knee peak.
    end = min(stop, ground_start + round(0.75 * s.fps) + 1)
    if f["step_lift"] is not None:
        end = min(end, f["step_lift"])
    if f["recovery"] is not None:
        end = min(end, f["recovery"] + 1)
    bend = s.knee[ground_start:end]
    if len(bend) < 3 or not np.isfinite(bend).all():
        return
    # With appreciable view rotation, use the lowest observed hip as the
    # landing landmark; projected knee angle comparisons would be misleading.
    motion = s.hip_y[ground_start:end] if oblique else bend
    if not np.isfinite(motion).all():
        return
    smoothed = smooth_runs(motion, max(1, round(0.06 * s.fps)), 1)
    peak = int(np.argmax(smoothed))
    reversal = 0.025 * s.scale if oblique else 4
    if np.any(smoothed[peak + 1:] < smoothed[peak] - reversal) or f["recovery"] is not None:
        f["landing_bottom"] = ground_start + peak


def measure(s, a, oblique):
    f, m = a.frames, a.metrics
    for phase, metric in (("countermovement", "dip"), ("takeoff", "takeoff"),
                          ("contact", "contact"), ("landing_bottom", "landing")):
        i = f[phase]
        if i is None or oblique:
            continue
        if np.isfinite(s.knee[i]):
            m[metric + "_knee_deg"] = round(float(s.knee[i]), 1)
        if metric in ("dip", "landing") and np.isfinite(s.torso[i]):
            m[metric + "_torso_deg"] = round(float(s.torso[i]), 1)
    c, b = f["contact"], f["landing_bottom"]
    if c is not None and b is not None and np.isfinite(s.knee[[c, b]] + s.hip_y[[c, b]]).all():
        a.visible_landing_bend = bool(s.knee[b] - s.knee[c] > 25 and s.hip_y[b] - s.hip_y[c] > 0.06 * s.scale)
    if f["dip_start"] is not None and f["countermovement"] is not None:
        m["descent_s"] = round((f["countermovement"] - f["dip_start"]) / s.fps, 4)
    if a.confirmed and np.isfinite(s.clearance[f["takeoff"]:f["contact"] + 1]).all():
        m["flight_time_estimate_s"] = round((f["contact"] - f["takeoff"]) / s.fps, 4)
    if m["contact_knee_deg"] is not None and m["landing_knee_deg"] is not None:
        m["landing_bend_deg"] = round(m["landing_knee_deg"] - m["contact_knee_deg"], 1)


def analyze(frames, fps, n, *, width, height, settings: Settings, verify_contact=None):
    points, scores = observations(frames, n, width, height, settings.joint_score)
    side, _ = choose_side(points, scores) if n else (None, {})
    result = Analysis(points, np.full(n, "grounded", dtype=object), side)
    s = make_signals(points, side, width, height, fps, settings) if side else None
    if s is None:
        result.phases[:] = "unconfirmed"
        result.warnings.append("A standing leg and a stable floor reference could not be established. "
                               "Keep both shoes and the floor in frame and include a short standing pause.")
        return result

    oblique = oblique_view(points, s)
    if oblique:
        result.warnings.append("Both shoulders and hips appear separated across much of this view. "
                               "Knee and torso angle measurements are uncertain. Film directly beside the jumper "
                               "with the camera level for angle comparisons.")
    windows = list(excursions(s, settings))
    dips = [preparation(s, windows[i - 1][1] if i else 0, lo) for i, (lo, _, _) in enumerate(windows)]
    known = np.isfinite(s.clearance).all(axis=1)
    for i, (lo, hi, peak) in enumerate(windows):
        start, dip = dips[i]
        stop = n
        rebound = False
        if i + 1 < len(windows):
            next_start, next_dip = dips[i + 1]
            # In repeated jumps the deepest landing can also be the next dip.
            # Keep the upward reversal visible for the landing-bend measurement.
            rebound = next_start is None and next_dip is not None
            stop = next_start if next_start is not None else windows[i + 1][0]
        a = Attempt(next((v for v in (start, dip, max(0, lo - 1)) if v is not None), 0), min(hi, n - 1))
        f = a.frames
        f.update(dip_start=start, countermovement=dip, flight=peak)
        if start is None or dip is None:
            a.issue("countermovement", a.first, "The full standing-to-dip transition is unclear. Include the standing pause before takeoff.")
        if peak is not None and lo > 0 and known[lo - 1:lo + 1].all():
            f["takeoff"] = lo
        else:
            a.issue("takeoff", lo, "Both feet must be visible as they leave the floor. Keep the shoes in frame from standing.")
        if peak is None:
            a.issue("flight", lo, "The body rose while both ankles were hidden. Airborne movement could not be confirmed.")
        elif not known[peak]:
            a.issue("flight", peak, "Only one ankle is visible above floor level here. Keep both shoes visible to confirm two-foot flight.")
        if peak is not None and hi < n and known[hi - 1:hi + 1].all():
            f["contact"] = hi
        else:
            detail = ("The trim ends before landing is visible. Keep filming through the return to standing."
                      if hi == n else "The feet are obscured around first contact. Leave the shoes and floor visible through landing.")
            gap = np.flatnonzero(~known[max(lo, peak or lo):min(hi + 1, n)])
            issue_frame = max(lo, peak or lo) + int(gap[0]) if gap.size else min(hi, n - 1)
            a.issue("contact", issue_frame, detail)
        for phase in ("takeoff", "contact"):
            frame = f[phase]
            if frame is not None:
                a.methods[phase] = "ankle motion"
                if verify_contact is not None:
                    refined = verify_contact(phase, frame, peak, s)
                    if refined is not None:
                        f[phase] = refined
                        a.methods[phase] = "visible shoe edge"
        landing_start = f["contact"] if f["contact"] is not None else hi
        landing(s, a, landing_start, stop, settings, oblique)
        measure(s, a, oblique)
        if f["contact"] is not None and f["landing_bottom"] is None:
            a.issue("landing_bottom", f["contact"], "The contact-to-deepest knee bend is uncertain. Keep the near hip, knee and ankle visible until you stand again.")
        if f["recovery"] is None:
            bottom = f["landing_bottom"]
            a.rebounds_into_next = bool(rebound and bottom is not None and abs(next_dip - bottom) <= 0.28 * fps
                                        and known[landing_start:max(bottom, next_dip) + 1].all())
            detail = ("The landing flows into the next dip without a separate settled stance. Replay the grounded interval."
                      if a.rebounds_into_next else "A settled landing stance could not be confirmed. Keep recording both feet through recovery.")
            a.issue("recovery", bottom if a.rebounds_into_next else a.last, detail)
        if f["takeoff"] is not None and f["contact"] is not None and a.metrics["flight_time_estimate_s"] is None:
            if not any(issue["phase"] == "flight" for issue in a.issues):
                a.issue("flight", peak, "Ankle tracking dropped during flight. The elapsed flight estimate is uncertain.")
        depart = f["takeoff"] if f["takeoff"] is not None else lo
        touch = f["contact"] if f["contact"] is not None else hi
        if dip is not None:
            result.phases[a.first:dip + 1] = "countermovement"
            result.phases[dip + 1:depart] = "push off"
        result.phases[depart:touch] = "flight" if peak is not None else "unconfirmed"
        result.phases[touch:a.last + 1] = "recovery" if f["contact"] is not None else "landing unclear"
        if f["landing_bottom"] is not None:
            result.phases[touch:f["landing_bottom"] + 1] = "landing"
        if f["takeoff"] is not None:
            result.phases[depart] = "takeoff estimate"
        if f["contact"] is not None:
            result.phases[touch] = "contact estimate"
        result.attempts.append(a)
    result.phases[~np.isfinite(s.clearance).any(axis=1)] = "feet obscured"
    if not result.attempts:
        result.warnings.append("No sustained airborne movement was visible. Keep both feet and the floor in view from standing through landing.")
    return result

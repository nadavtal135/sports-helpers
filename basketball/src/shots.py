# Adapted from jeremyipark/vision-demos basketball/src/shots.py (Apache-2.0),
# commit c06cfb78ccd8e2c63629222695aac710fe4c199c. Modifications: judge() no
# longer turns tracking loss into a make and bounds evidence gaps (see its
# docstring); an explicit "unknown" outcome; find_release reports its method;
# streaks() treats unknown as neither make nor miss; tally() added.
"""Shots: found in the pose, judged by the ball against the rim, plus the elbow and stance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hoop import Hoop
from .skeleton import KPT_NAMES

J = {name: i for i, name in enumerate(KPT_NAMES)}
HEAD = [J[n] for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]


# ── pose, as arrays ──────────────────────────────────────────────────────────

def keypoints(by_index: dict, n_frames: int, width: int, height: int) -> np.ndarray:
    """``(n_frames, 17, 2)`` in pixels; NaN where nobody is posed or a joint is missing."""
    out = np.full((n_frames, len(KPT_NAMES), 2), np.nan)
    for i in range(n_frames):
        persons = by_index.get(i)
        if not persons:
            continue
        kpts = np.asarray(persons[0].get("kpts_xy") or [], float)
        if kpts.shape != (len(KPT_NAMES), 2):
            continue
        missing = (kpts == 0).all(axis=1)   # (0, 0) is the model's "not visible"
        kpts = kpts * [width, height]
        kpts[missing] = np.nan
        out[i] = kpts
    return out


def elbow_angle(kp: np.ndarray, side: str, *, min_projected: float = 0.0) -> np.ndarray:
    """The elbow angle per frame in the image, degrees (180 = straight).

    NaN where the upper arm or forearm shows under *min_projected* of its usual
    length: pointing at the camera, its angle means nothing.
    """
    s, e, w = (kp[:, J[f"{side}_{j}"]] for j in ("shoulder", "elbow", "wrist"))
    a, b = s - e, w - e
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = np.degrees(np.arccos(np.clip((a * b).sum(axis=1) / (la * lb), -1, 1)))
        if min_projected > 0:
            short = (la < min_projected * np.nanpercentile(la, 90)) | (
                lb < min_projected * np.nanpercentile(lb, 90))
            angle[short] = np.nan
    return angle


def shoulder_ratio(kp: np.ndarray) -> np.ndarray:
    """Shoulder width / torso length per frame: ~0.1-0.45 side-on, ~0.6 facing or facing away."""
    shoulders = np.linalg.norm(kp[:, J["left_shoulder"]] - kp[:, J["right_shoulder"]], axis=1)
    mid_s = (kp[:, J["left_shoulder"]] + kp[:, J["right_shoulder"]]) / 2
    mid_h = (kp[:, J["left_hip"]] + kp[:, J["right_hip"]]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return _median(shoulders / np.linalg.norm(mid_s - mid_h, axis=1), 9)


def _median(values: np.ndarray, frames: int) -> np.ndarray:
    """Running median over *frames*, ignoring NaN."""
    half = frames // 2
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = values[max(0, i - half):i + half + 1]
        if np.isfinite(window).any():
            out[i] = np.nanmedian(window)
    return out


# ── finding shots ────────────────────────────────────────────────────────────

def candidates(kp: np.ndarray, fps: float, *, armed_below: str, min_gap_seconds: float
               ) -> list[tuple[int, int]]:
    """``[(first, last frame with both wrists over the head), ...]``, one per shot.

    The wrists must drop below *armed_below* between shots, so a follow-through can't count twice.
    """
    head_top = np.nanmin(kp[:, HEAD, 1], axis=1)
    wrists = np.nanmax(kp[:, [J["left_wrist"], J["right_wrist"]], 1], axis=1)   # the lower wrist
    arm_line = np.nanmax(kp[:, [J[f"left_{armed_below}"], J[f"right_{armed_below}"]], 1], axis=1)
    with np.errstate(invalid="ignore"):
        up = _median((wrists < head_top).astype(float), 5) > 0.5
        down = _median((wrists > arm_line).astype(float), 5) > 0.5

    spans: list[list[int]] = []
    armed = True
    i = 0
    while i < len(up):
        if armed and up[i]:
            j = i
            while j + 1 < len(up) and up[j + 1]:
                j += 1
            if spans and i - spans[-1][1] < min_gap_seconds * fps:
                spans[-1][1] = j
            else:
                spans.append([i, j])
            armed = False
            i = j + 1
            continue
        if down[i]:
            armed = True
        i += 1
    return [(a, b) for a, b in spans]


def shooting_side(kp: np.ndarray, spans: list[tuple[int, int]], fps: float) -> str:
    """The arm that straightens most through the shots; the guide hand barely does."""
    swing = {}
    for side in ("left", "right"):
        angle = elbow_angle(kp, side)
        moves = []
        for a, _ in spans:
            before = angle[max(0, a - int(0.4 * fps)):a + 1]
            after = angle[a:a + int(0.6 * fps)]
            if np.isfinite(before).any() and np.isfinite(after).any():
                moves.append(np.nanmax(after) - np.nanmin(before))
        swing[side] = np.median(moves) if moves else 0.0
    return max(swing, key=swing.get)


# ── one shot ─────────────────────────────────────────────────────────────────

@dataclass
class Shot:
    number: int
    start: int                 # first frame with both hands overhead
    end: int                   # last frame overhead
    release: int               # the ball leaves the hand
    set_frame: int             # the deepest elbow bend before the release
    elbow_set: float           # degrees, projected into the image (NaN when not side-on)
    elbow_release: float
    side_on: bool = True
    release_method: str = "ball"    # "ball" (ball/wrist separation) or "pose" (wrist peak fallback)
    made: bool | None = None   # None = unknown: the evidence does not support a call
    reason: str = ""
    evidence: list[str] = field(default_factory=list)   # the checks behind the verdict, in order
    crossing: float | None = None   # frame the ball crossed the rim line going in
    decided: int | None = None      # frame the verdict shows on the video
    feet_px: tuple[float, float] | None = None
    body_px: float | None = None    # shoulder to ankle, px
    court_xy: tuple[float, float] | None = None   # meters from the hoop (estimate)
    distance_m: float | None = None
    path: list[tuple[int, float, float]] = field(default_factory=list)   # ball (frame, x, y)
    motion: int | None = None
    segment: int | None = None      # camera segment (index into the hoop timeline)

    @property
    def result(self) -> str:
        return "make" if self.made else "miss" if self.made is False else "unknown"

    @property
    def label(self) -> str:
        return self.result.upper()

    def summary(self, fps: float, frame_offset: int = 0) -> dict:
        """JSON-ready. Times are seconds in the *original* clip (trim offset added)."""
        def t(frame):
            return None if frame is None else round((frame + frame_offset) / fps, 3)
        return {
            "shot": self.number, "result": self.result, "reason": self.reason,
            "evidence": list(self.evidence),
            "hands_up_s": t(self.start), "release_s": t(self.release),
            "release_method": self.release_method, "camera_segment": self.segment,
            "rim_crossing_s": t(self.crossing), "decided_s": t(self.decided),
            "source_frame_release": self.release + frame_offset,
            "motion_s": t(self.motion),
            "side_on": self.side_on,
            "elbow_set_projected_deg": _round(self.elbow_set),
            "elbow_release_projected_deg": _round(self.elbow_release),
            "court_xy_m_estimate": [round(v, 2) for v in self.court_xy] if self.court_xy else None,
            "distance_m_estimate": round(self.distance_m, 2) if self.distance_m is not None else None,
            "distance_ft_estimate": round(self.distance_m * 3.2808, 1) if self.distance_m is not None else None,
        }


def _round(value: float) -> float | None:
    return round(float(value), 1) if value is not None and np.isfinite(value) else None


def motion_start(start: int, x: np.ndarray, y: np.ndarray, size: np.ndarray, wrists: np.ndarray,
                 fps: float, *, lookback: float, reach: float, max_gap: int = 3) -> int:
    """Where the shooting motion begins: back from the hands going up, the ball's lowest point
    while it stays in the hands. (Reference, unchanged.)"""
    best, gap = None, 0
    for f in range(start, max(-1, start - int(lookback * fps)), -1):
        if not np.isfinite(x[f]):
            gap += 1
            if gap > max_gap:
                break
            continue
        gap = 0
        near = np.hypot(wrists[f, :, 0] - x[f], wrists[f, :, 1] - y[f])
        if not np.isfinite(near).any() or np.nanmin(near) > reach * size[f]:
            break
        if best is None or y[f] > y[best]:
            best = f
    return start if best is None else best


def find_release(start: int, end: int, ball_y: np.ndarray, wrist_y: np.ndarray,
                 diameter: float, fps: float) -> tuple[int, str]:
    """``(frame, method)``: the first frame the ball is 1.5 diameters over the shooting
    wrist and rising ("ball"); without that, the wrist's peak ("pose")."""
    lo, hi = max(0, start - int(0.3 * fps)), min(len(ball_y) - 3, end + int(0.5 * fps))
    if np.isfinite(diameter):
        for f in range(lo, hi):
            if not (np.isfinite(ball_y[f]) and np.isfinite(wrist_y[f])):
                continue
            nxt = ball_y[f + 1:f + 3]
            rising = np.isfinite(nxt).all() and nxt[0] < ball_y[f] and nxt[1] < nxt[0]
            if ball_y[f] < wrist_y[f] - 1.5 * diameter and rising:
                return f, "ball"
    window = wrist_y[start:min(len(wrist_y), start + int(0.7 * fps))]
    return (start + int(np.nanargmin(window)) if np.isfinite(window).any() else start), "pose"


def _max_gap(frames: list[int]) -> int:
    return max((b - a for a, b in zip(frames, frames[1:])), default=0)


def judge(shot: Shot, x: np.ndarray, y: np.ndarray, size: np.ndarray, hoop: Hoop, fps: float, *,
          until: int, inner_fraction: float, net_min_seconds: float, net_side_tolerance: float,
          rebound_margin: float, max_gap: int, evidence_gap: int, min_flight: int,
          ball_to_rim: float, front_size: float, rim_diameter_m: float = 0.49,
          fall_speed_tolerance: float = 0.25, net_slow_factor: float = 1.3,
          max_ball_speed_mps: float = 15.0) -> None:
    """Make, miss or unknown.

    A make comes down from above the rim, crosses the rim line inside it at the
    rim's distance (not in front of it), is *seen leaving the net* slowed by it,
    and never rises back over the rim. Every downward crossing is tried, so a
    roll-in is a make. (Reference rules.)

    Changed from the reference, which called a ball that crossed inside the rim
    and was then never seen again a make ("into the net"), and which bridged any
    gap between the crossing and the next sighting:

    * Losing the ball after an inside crossing is **unknown**, not a make.
    * Every gap between the crossing and the net exit must be ≤ *evidence_gap*
      frames, and the first sighting below the net must be at the net exit
      (within a rim width of it); otherwise the shot is **unknown**.
    * A crossing hidden by a gap longer than *max_gap* right at the rim is
      **unknown**, not "never came down through the rim".
    * "Never above the rim" is a miss only if the ball was tracked for
      *min_flight* frames after release without a long gap; otherwise unknown.
    * Lost while still above the rim (never seen at or below rim height again)
      is **unknown**.
    Misses still need positive evidence (outside the rim, rimmed out, in front,
    fell past the net, wide of the net, off the rim).

    Net timing is scaled to the clip (changed from a fixed *net_min_seconds* floor,
    which now only still counts as net drag). At 30 fps a fast clean swish
    goes rim to net bottom in ~3 frames, under the old 0.15 s floor, and was
    called "fell past the net". Gravity is converted to pixels per frame from the
    rim's real width and the frame rate, and:

    * a ball moving through the net faster than free fall from its own entry
      speed allows (by more than *fall_speed_tolerance*) cannot be inside the
      net: it is nearer the camera, falling past it. Miss.
    * otherwise, crossing inside the rim and leaving under the net within the
      rim's width, with no rebound, is a make; taking *net_slow_factor* × the
      free-fall time (or the old *net_min_seconds*) or longer is reported as net
      drag, faster as "not clearly slowed at this frame rate" (evidence only).
    * leaving at the net's edge (outside the rim's width, inside the net
      tolerance) needs the net drag too; without it the shot is unknown.

    A "crossing" between two sightings that would need the ball to move faster
    than *max_ball_speed_mps* is not a crossing: the two sightings are of
    different moments (e.g. the first shot's rebound, then the ball back in the
    hands for a second shot) and say nothing about the rim.
    """
    w = hoop.width
    # Gravity in pixels per frame², at the rim's distance (where the rim's real width is known).
    g = 9.81 * (w / rim_diameter_m) / (fps * fps)
    max_step = max_ball_speed_mps * (w / rim_diameter_m) / fps     # px per frame
    frames = [f for f in range(shot.release, min(until, len(x))) if np.isfinite(x[f])]
    shot.path = [(f, float(x[f]), float(y[f])) for f in frames]
    shot.evidence = []
    if not frames:
        shot.made, shot.reason = None, "ball not tracked after release"
        return

    above = [f for f in frames if y[f] < hoop.cy - 0.5 * w]
    if not above:
        tracked = frames[-1] - frames[0]
        if tracked >= min_flight and _max_gap(frames) <= evidence_gap:
            shot.made, shot.reason, shot.decided = False, "never above the rim", frames[-1]
            shot.evidence = [f"ball tracked {tracked / fps:.2f}s after release, never above the rim"]
        else:
            shot.made, shot.decided = None, frames[-1]
            shot.reason = "ball lost before it could reach the rim"
            shot.evidence = [f"only {tracked / fps:.2f}s of ball track after release"]
        return

    reason = "never came down through the rim"
    unknown: tuple[str, int, list[str]] | None = None   # (reason, decided frame, evidence)
    for a, b in zip(frames, frames[1:]):
        if a < above[0] or not (y[a] < hoop.cy <= y[b]):
            continue
        if np.hypot(x[b] - x[a], y[b] - y[a]) / (b - a) > max_step:
            continue   # an impossible jump, not one ball's path
        t = (hoop.cy - y[a]) / (y[b] - y[a])
        cross = a + t * (b - a)
        cross_x = x[a] + t * (x[b] - x[a])
        if b - a > max_gap:
            # The ball was hidden while it came down. Near the rim that is exactly
            # where a make hides, so it cannot be called a miss either.
            if abs(cross_x - hoop.cx) < w and unknown is None:
                unknown = ("ball hidden while crossing the rim", b,
                           [f"{(b - a) / fps:.2f}s gap in the ball track at the rim"])
            continue
        offset = abs(cross_x - hoop.cx) / (w / 2)
        if offset > inner_fraction:
            reason = "outside the rim"
            continue
        checks = ["came down from above the rim",
                  f"crossed the rim line inside it ({offset:.0%} of the half-width from center)"]

        after = [f for f in frames if f > cross]
        out = rebound = None
        lost_gap = None
        for c, d in zip([a] + after, after):
            if d - c > evidence_gap and c >= cross:
                lost_gap = (c, d)
                break
            if y[d] < hoop.cy - rebound_margin * w:
                rebound = d
                break
            if y[d] >= hoop.net_bottom:
                if y[d] > hoop.net_bottom + w or abs(x[d] - hoop.cx) > w:
                    # The next sighting is not the ball leaving the net (e.g. it is
                    # back in someone's hands): the exit itself was not observed.
                    lost_gap = (c, d)
                    break
                s = (hoop.net_bottom - y[c]) / (y[d] - y[c]) if y[d] != y[c] else 1.0
                out = (max(cross, c + s * (d - c)), x[c] + s * (x[d] - x[c]), d)
                break
        if rebound is not None:
            reason = "rimmed out"
            continue
        end = out[2] if out is not None else cross + 0.3 * fps
        sizes = [size[f] for f in frames if cross - 0.2 * fps <= f <= end and np.isfinite(size[f])]
        if sizes and np.median(sizes) / w > front_size * ball_to_rim:
            reason = "in front of the rim"
            continue
        if sizes:
            checks.append("ball size matches the rim's distance (not in front of it)")
        if out is None:
            # The reference called this a make ("into the net"). Tracking loss is
            # not evidence of a make: leave it unknown unless a later crossing decides.
            last = after[-1] if after else int(np.ceil(cross))
            if lost_gap is not None and not (y[lost_gap[1]] >= hoop.net_bottom
                                              and lost_gap[1] - lost_gap[0] <= evidence_gap):
                why = (f"crossed inside the rim, then a {(lost_gap[1] - lost_gap[0]) / fps:.2f}s "
                       "gap in the ball track before any net exit")
            else:
                why = "crossed inside the rim, but the ball was not seen leaving the net"
            if unknown is None or unknown[0].startswith("ball hidden"):
                unknown = (why, last, checks + [why])
            continue
        out_frame, out_x, seen = out
        fall = _net_fall(frames, x, y, a, b, cross, seen, g, hoop)
        if fall["too_fast"] > 1 + fall_speed_tolerance:
            reason = "fell past the net"
            continue
        edge = abs(out_x - hoop.cx) - w / 2          # > 0: left the net outside the rim's width
        if edge > net_side_tolerance * w:
            reason = "wide of the net"
            continue
        passage = out_frame - cross
        # The old fixed floor still counts as net drag, so a make it accepted stays one.
        slowed = passage >= min(net_slow_factor * fall["free_fall_frames"], net_min_seconds * fps)
        timing = (f"{passage / fps:.2f}s rim to net bottom; free fall would take "
                  f"{fall['free_fall_frames'] / fps:.2f}s at this entry speed")
        if edge > 0 and not slowed:
            if unknown is None:
                why = "left the net at its edge, too fast to show the net slowing it"
                unknown = (why, seen, checks + [timing, why])
            continue
        checks += [("left the net under the rim" if edge <= 0 else "left the net at its edge")
                   + f" ({abs(out_x - hoop.cx) / (w / 2):.0%} of the half-width from center)",
                   ("slowed by the net: " if slowed else "not clearly slowed at this frame rate: ")
                   + timing,
                   "never rose back over the rim"]
        shot.made, shot.reason, shot.crossing, shot.decided = True, "through the net", cross, seen
        shot.evidence = checks
        return

    if unknown is not None:
        shot.made, shot.reason, shot.decided = None, unknown[0], unknown[1]
        shot.evidence = unknown[2]
        return

    if reason == "never came down through the rim":
        came_down = [f for f in frames if f > above[-1] and y[f] >= hoop.cy]
        if not came_down:
            shot.made, shot.decided = None, frames[-1]
            shot.reason = "ball lost while still above the rim"
            shot.evidence = ["no sighting at or below rim height after its last frame above it"]
            return

    # A miss shows once the ball leaves the rim for good; "off the rim" if it bounced up there.
    near = [f for f in frames if abs(x[f] - hoop.cx) < w and abs(y[f] - hoop.cy) < w]
    bounced = any(y[f] - min((y[g] for g in frames if g > f), default=y[f]) > 0.4 * w
                  for f in near)
    later = [f for f in frames if near and f > near[-1]]
    shot.made, shot.decided = False, (later[0] if later else frames[-1])
    shot.reason = "off the rim" if bounced else reason
    shot.evidence = [shot.reason]


def not_a_shot(shot: Shot, x: np.ndarray, y: np.ndarray, hoop: Hoop, fps: float, *,
               window_s: float, min_obs: int, recede_rims: float, early_s: float = 0.5,
               gap_s: float = 0.3, lead_s: float = 0.3,
               bounce_lookback_s: float = 1.0) -> tuple[str, list[str]] | None:
    """``(reason, evidence)`` when a candidate is clearly not a shot attempt, else None.

    Only a candidate whose release came from the pose fallback (the ball was not
    seen leaving the hands) is checked. It is dropped only when the ball WAS seen
    for at least *min_obs* frames around the release (at least 3 after it),
    never got above the rim in the window, and, right after the release, moved
    away from the hoop horizontally by at least *recede_rims* rim widths. That is
    a player catching or gathering the ball with both hands up.

    The recession is read from the ball's first moves, not fitted over the whole
    window: a gathered ball often bounces back towards the hoop a second later,
    and a line through those later sightings cancelled the recession on a real
    clip. It is measured over one unbroken run of sightings, from the first one
    up to *lead_s* before the release to the last one *early_s* after it (a gap
    longer than *gap_s* ends the run, so sightings after it count for nothing),
    as the furthest the ball got from the hoop after the release minus its
    distance at the start of the run. The ball must also still be moving away
    at the end of that run (not already coming back towards the hoop).

    A ball lost early, or one heading towards the hoop, is kept (its shot is then
    judged, usually unknown). A floor bounce just before is reported as
    supporting evidence only: real jump shots often follow a dribble.
    """
    if shot.release_method != "pose":
        return None
    w = hoop.width
    lo = max(0, shot.start - int(lead_s * fps))
    hi = min(len(x), shot.release + int(window_s * fps) + 1)
    seen = [f for f in range(lo, hi) if np.isfinite(x[f])]
    after = [f for f in seen if f >= shot.release]
    if len(seen) < min_obs or len(after) < 3:
        return None
    if any(y[f] < hoop.cy - 0.5 * w for f in seen if f >= shot.start):
        return None

    def dist(f):
        return abs(float(x[f]) - hoop.cx)

    # One unbroken run of sightings around the release: [release - lead, release + early].
    early_end = shot.release + early_s * fps
    near = [f for f in seen if shot.release - lead_s * fps <= f <= early_end]
    runs, run = [], []
    for f in near:
        if run and f - run[-1] > gap_s * fps:
            runs.append(run)
            run = []
        run.append(f)
    if run:
        runs.append(run)
    run = next((r for r in runs if r[-1] >= shot.release), None)
    post = [f for f in run or [] if f >= shot.release]
    if not post:
        return None
    base = run[0]
    at_release = min(run, key=lambda f: (abs(f - shot.release), f))
    furthest = max(post, key=dist)
    receded = dist(furthest) - dist(base)
    still_leaving = dist(post[-1]) >= dist(at_release)
    if receded < recede_rims * w or not still_leaving:
        return None

    evidence = [f"ball seen on {len(seen)} frames around the hands-up, never above the rim",
                f"ball moved {receded / w:.1f} rim widths away from the hoop",
                "release time came from the pose alone (the ball was not seen leaving the hands)",
                f"early window: {dist(base):.0f} px from the hoop at {base / fps:.2f}s, "
                f"{dist(furthest):.0f} px at {furthest / fps:.2f}s "
                f"(+{receded:.0f} px, {(furthest - shot.release) / fps:.2f}s after the release)"]
    before = [f for f in range(max(0, shot.start - int(bounce_lookback_s * fps)), shot.start + 1)
              if np.isfinite(y[f])]
    if before:
        low = max(before, key=lambda f: y[f])
        dropped = y[low] - min(y[f] for f in before if f <= low)
        rose = y[low] - min((y[f] for f in seen if f > low), default=y[low])
        if dropped > 2 * w and rose > w:
            evidence.append("the ball bounced on the floor just before")
    return "not a shot: the ball went away from the hoop (a catch or gather)", evidence


def clearly_not_a_shot(shot: Shot, x: np.ndarray, y: np.ndarray, hoop: Hoop, fps: float, *,
                       window_s: float, min_obs: int, recede_rims: float, early_s: float = 0.5,
                       gap_s: float = 0.3, lead_s: float = 0.3) -> bool:
    """The stricter test that may *remove* a candidate (``not_a_shot`` alone only flags it).

    Both: ``not_a_shot`` holds with twice the recession, and no sighting in the whole
    window after the release, including after a tracking gap, comes back closer to the
    hoop than the ball was at the start of its early run. On a real clip the ball of a
    real shot taken right after another was seen moving away, lost, then seen closer
    to the hoop again; that must stay an (unknown) attempt, never vanish.
    """
    if not_a_shot(shot, x, y, hoop, fps, window_s=window_s, min_obs=min_obs,
                  recede_rims=2 * recede_rims, early_s=early_s, gap_s=gap_s, lead_s=lead_s) is None:
        return False
    lo = shot.release - lead_s * fps
    hi = min(len(x), shot.release + int(window_s * fps) + 1)
    seen = [f for f in range(max(0, int(np.ceil(lo))), hi) if np.isfinite(x[f])]
    if not seen:
        return False
    base = abs(float(x[seen[0]]) - hoop.cx)
    return all(abs(float(x[f]) - hoop.cx) >= base for f in seen if f >= shot.release)


def attempt_check(shot: Shot, x: np.ndarray, y: np.ndarray, hoop: Hoop, fps: float, **kw
                  ) -> tuple[str, tuple[str, list[str]] | None]:
    """``("keep", None)``, ``("flag", found)`` or ``("drop", found)`` for one candidate.

    "flag": ``not_a_shot`` holds, so the shot is kept but shown as a possible
    non-shot (``flag_possible_non_shot``). "drop": the stricter
    ``clearly_not_a_shot`` holds too, the only case a candidate is removed.
    *kw* are ``not_a_shot``'s keyword arguments.
    """
    found = not_a_shot(shot, x, y, hoop, fps, **kw)
    if found is None:
        return "keep", None
    if clearly_not_a_shot(shot, x, y, hoop, fps, **kw):
        return "drop", found
    return "flag", found


def flag_possible_non_shot(shot: Shot, found: tuple[str, list[str]]) -> None:
    """Show a judged shot as UNKNOWN "possible non-shot", unless judge() found a make
    (a ball seen going through the net is a shot). The judge's own call is kept in
    the evidence, so nothing is hidden."""
    if shot.made is True:
        shot.evidence = list(shot.evidence) + ["(flagged as a possible non-shot, but the ball "
                                               "went through the net)"]
        return
    judged = f"judge alone: {shot.label.lower()} ({shot.reason})"
    shot.made = None
    shot.reason = "possible non-shot: ball moved away from the hoop"
    shot.evidence = list(found[1]) + [judged]


def _net_fall(frames, x, y, a: int, b: int, cross: float, exit_frame: int, g: float,
              hoop: Hoop) -> dict:
    """How the ball's fall through the net compares with free fall from its entry speed.

    ``too_fast``: the largest ratio, over observed steps between the crossing and
    the exit sighting, of the measured fall speed to the most free fall from the
    entry speed could reach by then (1 means exactly free fall; a net only slows
    a ball, so well above 1 means it is not in the net). ``free_fall_frames``:
    the time free fall would take from the rim line to the net bottom.
    Speeds are averaged over each step, so a frame hidden by the net only
    makes them less precise, never faster.
    """
    seen = [f for f in frames if f <= b]
    entry = (y[b] - y[a]) / (b - a)
    if len(seen) >= 3:      # the step before the crossing pair, if it was also falling
        p, q = seen[-3], seen[-2]
        entry = max(entry, (y[q] - y[p]) / (q - p))
    entry = max(entry, 0.0)
    height = hoop.net_bottom - hoop.cy
    free_fall = ((-entry + np.sqrt(entry * entry + 2 * g * height)) / g if g > 0
                 else height / max(entry, 1e-6))
    steps = [f for f in frames if a <= f <= exit_frame]
    worst = 0.0
    for c, d in zip(steps, steps[1:]):
        if d <= cross:
            continue
        speed = (y[d] - y[c]) / (d - c)
        allowed = entry + g * max(0.0, (c + d) / 2 - cross) + 1.0   # 1 px/frame of keypoint jitter
        worst = max(worst, speed / allowed)
    return {"too_fast": worst, "free_fall_frames": float(free_fall), "entry_px_per_frame": float(entry)}


def ball_unresolved(x: np.ndarray, y: np.ndarray, start: int, end: int, hoop: Hoop) -> bool:
    """Whether the ball was last seen at the rim, so its window should run longer."""
    seen = [f for f in range(start, min(end, len(x))) if np.isfinite(x[f])]
    if not seen:
        return False
    f = seen[-1]
    return y[f] < hoop.net_bottom and abs(x[f] - hoop.cx) < 1.5 * hoop.width


def stance(kp: np.ndarray, start: int, fps: float) -> tuple[tuple[float, float], float] | None:
    """``((feet x, y), shoulder-to-ankle px)`` for a shot.

    The feet: median over the 0.5 s before the hands go up. The size: standing tall
    (90th percentile over 2 s), since bent knees in the gather would read as farther away.
    """
    def legs(lo: int, hi: int):
        seg = kp[max(0, lo):hi]
        ankles = np.nanmean(seg[:, [J["left_ankle"], J["right_ankle"]]], axis=1)
        shoulders = np.nanmean(seg[:, [J["left_shoulder"], J["right_shoulder"]]], axis=1)
        return ankles, np.linalg.norm(ankles - shoulders, axis=1)

    ankles, _ = legs(start - int(0.5 * fps), start + 1)
    _, size = legs(start - int(1.5 * fps), start + int(0.5 * fps))
    if not (np.isfinite(size).any() and np.isfinite(ankles).any()):
        return None
    return (float(np.nanmedian(ankles[:, 0])), float(np.nanmedian(ankles[:, 1]))), float(
        np.nanpercentile(size[np.isfinite(size)], 90))


def streaks(shots: list[Shot], n_frames: int) -> np.ndarray:
    """Confirmed makes in a row as of each frame, from the frame each shot is decided.

    A miss resets the run. An **unknown** neither extends nor resets it: it is
    not evidence of a miss, and not of a make either. (The reference had no
    unknown outcome on decided shots.)
    """
    out = np.zeros(n_frames, int)
    run = 0
    for shot in sorted((s for s in shots if s.decided is not None), key=lambda s: s.decided):
        if shot.made is True:
            run += 1
        elif shot.made is False:
            run = 0
        out[min(shot.decided, n_frames):] = run
    return out


def tally(shots: list[Shot]) -> dict:
    """Makes, misses, unknowns, and FG% over *judged* shots (makes + misses).

    Unknown shots are excluded from the denominator rather than counted as
    misses (the reference divided by every attempt).
    """
    makes = sum(1 for s in shots if s.made is True)
    misses = sum(1 for s in shots if s.made is False)
    unknown = sum(1 for s in shots if s.made is None)
    judged = makes + misses
    return {"attempts": len(shots), "makes": makes, "misses": misses, "unknown": unknown,
            "judged": judged, "fg_pct": round(makes / judged, 4) if judged else None,
            "fg_denominator": "makes + misses (unknown shots excluded)"}


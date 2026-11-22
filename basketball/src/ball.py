# Adapted from jeremyipark/vision-demos basketball/src/ball.py (Apache-2.0),
# commit c06cfb78ccd8e2c63629222695aac710fe4c199c. Modifications: the request
# moved to src/providers; cache keys moved to src/cache.py. Selection unchanged except
# for an opt-in motion-blur rescue pass (select(blur_rescue=True)), not in the reference.
"""The ball: SAM 3.1 tracking it through a window around each shot.

In the hands and the net SAM's box grows past the ball, so boxes off the ball's
size are dropped, and frames with no ball stay empty rather than guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

CONTENT_OBJECT = "vid.segment.masks"
CONTRACT = "sam31-ball-track-v1"

TRACK_MAX_FRAMES = 128   # SAM 3.1's per-call limit


def detections(payload: dict, *, frame_offset: int, width: int, height: int,
               min_score: float) -> dict[int, list[tuple[float, float, float, float, float]]]:
    """``{clip frame: [(cx, cy, w, h, score), ...]}`` in pixels of the inference frame."""
    content = payload.get("content")
    if not isinstance(content, dict) or content.get("object") != CONTENT_OBJECT:
        raise RuntimeError(f"Expected a {CONTENT_OBJECT} payload, got "
                           f"{json.dumps(payload)[:300]}")
    out: dict[int, list] = {}
    for item in content.get("items") or []:
        box = item.get("bbox_xywh")
        if item.get("frame_id") is None or not box or float(item.get("score") or 0) < min_score:
            continue
        x, y, w, h = box
        out.setdefault(int(item["frame_id"]) + frame_offset, []).append(
            ((x + w / 2) * width, (y + h / 2) * height, w * width, h * height,
             float(item.get("score") or 0)))
    return out


@dataclass
class BallTrack:
    """The ball's center and diameter on every frame of the clip; NaN where unseen."""

    x: np.ndarray
    y: np.ndarray
    d: np.ndarray
    covered: np.ndarray                      # bool: SAM looked at this frame
    windows: list[tuple[int, int]] = field(default_factory=list)
    diameter: float = float("nan")           # the ball's typical size, px
    shown: BallTrack | None = None           # a looser pick for drawing only, held in the hands too
    blurred: np.ndarray | None = None        # bool: found by the motion-blur rescue
    static: set = field(default_factory=set)     # (frame, cx, cy) of boxes that never move
    gated: set = field(default_factory=set)      # (frame, cx, cy) rejected as off the ball's path
    recovered: np.ndarray | None = None      # bool: found in the net by the local search

    @property
    def found(self) -> np.ndarray:
        return np.isfinite(self.x)


def typical_diameter(raw: dict[int, list], frames=None) -> float:
    """The ball's size: the median of the round boxes (one grown into a hand isn't round),
    over *frames* if given."""
    sizes = [max(w, h) for f, boxes in raw.items() if frames is None or f in frames
             for _, _, w, h, _ in boxes if abs(w - h) <= 0.2 * max(w, h)]
    return float(np.median(sizes)) if sizes else float("nan")


def key(frame: int, cx: float, cy: float) -> tuple[int, float, float]:
    """How a box is named in ``BallTrack.static`` / ``gated``."""
    return int(frame), round(float(cx), 1), round(float(cy), 1)


def static_boxes(raw: dict[int, list], diameter: float, fps: float, *, min_seconds: float = 0.5,
                 tolerance: float = 0.3, max_gap_seconds: float = 0.2) -> set:
    """Boxes of a ball that never moves (on a chair, a rack, a bag): ``{key(...)}``.

    Boxes are grouped by place (centers within *tolerance* ball diameters of the
    group's first box). A group is static when it is seen for more than *min_seconds*
    with no gap over *max_gap_seconds*, and nothing ball-sized leaves it within three
    diameters right after its last frame: a ball held still and then shot moves off
    from where it was held, a ball on a chair does not.
    """
    if not np.isfinite(diameter) or diameter <= 0:
        return set()
    groups: list[dict] = []
    for f in sorted(raw):
        for cx, cy, w, h, _s in raw[f]:
            g = next((g for g in groups if np.hypot(cx - g["x"], cy - g["y"]) <= tolerance * diameter
                      and f - g["frames"][-1] <= max_gap_seconds * fps), None)
            if g is None:
                groups.append({"x": cx, "y": cy, "frames": [f], "boxes": [key(f, cx, cy)]})
            elif f != g["frames"][-1]:
                g["frames"].append(f)
                g["boxes"].append(key(f, cx, cy))
            else:
                g["boxes"].append(key(f, cx, cy))
    out = set()
    for g in groups:
        first, last = g["frames"][0], g["frames"][-1]
        if last - first <= min_seconds * fps:
            continue
        leaves = any(tolerance * diameter < np.hypot(cx - g["x"], cy - g["y"]) <= 3 * diameter
                     for f in range(last + 1, last + 1 + int(max_gap_seconds * fps) + 1)
                     for cx, cy, *_ in raw.get(f, []))
        if not leaves:
            out.update(g["boxes"])
    return out


def select(raw: dict[int, list], n_frames: int, windows: list[tuple[int, int]], *,
           size_tolerance: float, local_size: bool = False, min_score: float = 0.0,
           blur_rescue: bool = False, max_stretch: float = 2.5, fps: float | None = None,
           static_seconds: float | None = None, max_speed_mps: float | None = None,
           gate_diameters: float = 3.0, gate_reset_seconds: float = 0.3) -> BallTrack:
    """One ball per frame: ball-sized, round, and on a smooth path.

    *local_size* sizes the ball per window, since it looks bigger near the camera.
    Boxes scoring under *min_score* are ignored except by the blur rescue
    (``rescue_blurred``), which *blur_rescue* turns on.

    New, both off unless *fps* is given:

    * *static_seconds*: boxes of a ball that stays put that long (``static_boxes``)
      are never candidates; they are listed in ``BallTrack.static``.
    * *max_speed_mps*: the continuity gate. Within *gate_reset_seconds* of the last
      sighting, a box only extends the track if reaching it needs no more than
      *max_speed_mps* (converted with the ball's diameter, 0.239 m) and it lies within
      *gate_diameters* ball diameters, plus twice the recent speed times the frames
      elapsed, of where the recent motion predicts. Rejected boxes are listed in
      ``BallTrack.gated``; after a longer gap the track may start afresh anywhere.
    """
    all_raw = raw
    raw = {f: [b for b in boxes if b[4] >= min_score] for f, boxes in all_raw.items()}
    static: set = set()
    if fps and static_seconds:
        static = static_boxes(raw, typical_diameter(raw), fps, min_seconds=static_seconds)
        raw = {f: [b for b in boxes if key(f, b[0], b[1]) not in static] for f, boxes in raw.items()}
    gated: set = set()
    diameter = typical_diameter(raw)
    x = np.full(n_frames, np.nan)
    y = np.full(n_frames, np.nan)
    d = np.full(n_frames, np.nan)
    covered = np.zeros(n_frames, bool)
    for start, length in windows:
        covered[start:start + length] = True
    if not np.isfinite(diameter):
        return BallTrack(x, y, d, covered, windows, diameter)

    local = np.full(n_frames, diameter)
    for start, length in windows if local_size else []:
        size = typical_diameter(raw, range(start, start + length))
        if np.isfinite(size):
            local[start:start + length] = size

    previous = None
    accepted: list[int] = []
    for frame in sorted(raw):
        if not 0 <= frame < n_frames:
            continue
        candidates = [b for b in raw[frame]
                      if abs(max(b[2], b[3]) / local[frame] - 1) <= size_tolerance
                      and abs(b[2] - b[3]) <= 0.35 * max(b[2], b[3])]
        if not candidates:
            continue
        if previous is None:
            best = max(candidates, key=lambda b: b[4])
        else:
            best = min(candidates, key=lambda b: np.hypot(b[0] - previous[0], b[1] - previous[1]))
        if fps and max_speed_mps and accepted and frame - accepted[-1] <= gate_reset_seconds * fps:
            ok = [b for b in candidates if _on_path(b, frame, accepted, x, y, local[frame], fps,
                                                    max_speed_mps, gate_diameters)]
            for b in candidates:
                if b not in ok:
                    gated.add(key(frame, b[0], b[1]))
            if not ok:
                continue
            best = min(ok, key=lambda b: np.hypot(b[0] - previous[0], b[1] - previous[1]))
        x[frame], y[frame], d[frame] = best[0], best[1], max(best[2], best[3])
        previous = best
        accepted.append(frame)

    # Drop lone points far off the path their neighbours draw (a head, a shoe).
    # Twice, since dropping one can expose the next.
    for _ in range(2):
        kept = np.nonzero(np.isfinite(x))[0]
        drop = []
        for k in range(1, len(kept) - 1):
            a, i, b = kept[k - 1], kept[k], kept[k + 1]
            if b - a > 8:
                continue
            t = (i - a) / (b - a)
            px, py = x[a] + t * (x[b] - x[a]), y[a] + t * (y[b] - y[a])
            if np.hypot(x[i] - px, y[i] - py) > 2.5 * local[i]:
                drop.append(i)
        x[drop] = y[drop] = d[drop] = np.nan
    blurred = np.zeros(n_frames, bool)
    if blur_rescue:
        if static:
            all_raw = {f: [b for b in boxes if key(f, b[0], b[1]) not in static]
                       for f, boxes in all_raw.items()}
        rescue_blurred(all_raw, x, y, d, covered, local, blurred,
                       size_tolerance=size_tolerance, max_stretch=max_stretch)
    return BallTrack(x, y, d, covered, windows, diameter, blurred=blurred, static=static,
                     gated=gated)


def _on_path(box, frame: int, accepted: list[int], x, y, diam: float, fps: float,
             max_speed_mps: float, gate_diameters: float) -> bool:
    """The continuity gate for one box (see ``select``)."""
    p1 = accepted[-1]
    dt = frame - p1
    step = float(np.hypot(box[0] - x[p1], box[1] - y[p1]))
    max_step = max_speed_mps / 0.239 * diam / fps          # px per frame at this ball size
    if step > max_step * dt + 0.5 * diam:
        return False
    if len(accepted) < 2 or p1 - accepted[-2] > 0.3 * fps:
        return True
    p2 = accepted[-2]
    vx, vy = (x[p1] - x[p2]) / (p1 - p2), (y[p1] - y[p2]) / (p1 - p2)
    speed = float(np.hypot(vx, vy))
    err = float(np.hypot(box[0] - (x[p1] + vx * dt), box[1] - (y[p1] + vy * dt)))
    return err <= gate_diameters * diam + 2 * speed * dt



def rescue_blurred(raw: dict[int, list], x, y, d, covered, local, blurred, *,
                   size_tolerance: float, max_stretch: float, max_step: int = 6,
                   min_speed: float = 0.3) -> int:
    """Fill frames with no accepted ball from boxes stretched by motion blur. In place.

    A ball moving fast at night smears along its motion, so SAM's box grows
    along that direction: it then fails both the size check (its long side) and
    the roundness check, and the ball vanishes exactly in flight. Here, only in
    a frame with no accepted ball and only while the ball is moving fast enough
    to blur (*min_speed* diameters per frame, from the two sightings just
    before, at most *max_step* frames apart), a box is accepted when:

    * its SHORT side is ball-sized (within *size_tolerance*) and its long side at
      most *max_stretch* diameters;
    * it is stretched along the direction of motion (wider than tall when the ball
      moves more sideways than up/down, and vice versa);
    * its center is within one diameter, plus half the distance travelled since
      the last sighting, of where the ball should be by then.

    Boxes of any score that reached ``detections`` are considered. Returns how
    many frames were filled; each is marked in *blurred*.
    """
    filled = 0
    found = [f for f in range(len(x)) if np.isfinite(x[f])]
    for f in sorted(raw):
        if not (0 <= f < len(x)) or not covered[f] or np.isfinite(x[f]):
            continue
        before = [g for g in found if g < f][-2:]
        if len(before) < 2 or f - before[1] > max_step or before[1] - before[0] > max_step:
            continue
        p2, p1 = before
        vx, vy = (x[p1] - x[p2]) / (p1 - p2), (y[p1] - y[p2]) / (p1 - p2)
        speed = float(np.hypot(vx, vy))
        diam = float(local[f])
        if speed < min_speed * diam:
            continue
        px, py = x[p1] + vx * (f - p1), y[p1] + vy * (f - p1)
        reach = diam + 0.5 * speed * (f - p1)
        options = []
        for cx, cy, w, h, _score in raw[f]:
            short, long_ = min(w, h), max(w, h)
            if abs(short / diam - 1) > size_tolerance or long_ / diam > max_stretch:
                continue
            if long_ / short >= 1.15 and (w > h) != (abs(vx) >= abs(vy)):
                continue   # stretched across the motion, not along it
            dist = float(np.hypot(cx - px, cy - py))
            if dist <= reach:
                options.append((dist, cx, cy, short))
        if options:
            _, cx, cy, short = min(options)
            x[f], y[f], d[f] = cx, cy, short
            blurred[f] = True
            found = sorted(found + [f])
            filled += 1
    return filled


def smoothed(track: BallTrack, *, max_gap: int, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The ball's ``(x, y)`` for drawing: short gaps filled along the path, then lightly
    smoothed, so the ring glides instead of blinking. Judging uses the strict track."""
    track = track.shown or track
    x, y = track.x.copy(), track.y.copy()
    seen = np.nonzero(np.isfinite(x))[0]
    for a, b in zip(seen, seen[1:]):
        if 1 < b - a <= max_gap + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            x[a + 1:b] = x[a] + t * (x[b] - x[a])
            y[a + 1:b] = y[a] + t * (y[b] - y[a])
    half = window // 2
    out = []
    for v in (x, y):
        s = v.copy()
        for i in np.nonzero(np.isfinite(v))[0]:
            seg = v[max(0, i - half):i + half + 1]
            s[i] = np.nanmean(seg)
        out.append(s)
    return out[0], out[1]

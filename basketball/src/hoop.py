# Adapted from jeremyipark/vision-demos basketball/src/hoop.py (Apache-2.0),
# commit c06cfb78ccd8e2c63629222695aac710fe4c199c. Modifications: the request
# moved to src/providers; cache keys moved to src/cache.py; rim parsing unchanged.
# Further modifications: every hoop SAM returns is parsed (candidates), the hoop
# is chosen by size / ball approaches / a user override rather than by score
# alone, and a HoopTimeline holds one hoop per camera segment.
"""The hoop: SAM 3.1 on one frame, then the rim read off the mask.

SAM segments the hoop (rim + net). The rim is read from the top of the mask: the
first width maximum going down, since the net tapers below it. (The reference took
the widest rows anywhere in the mask; on a real gym clip SAM's mask also held
pieces of a net-tension strap and a support post below the net, wider than the
rim, and the rim line landed on the net 0.65 rim widths too low.)
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

CONTENT_OBJECT = "img.segment.masks"
CONTRACT = "sam31-hoop-v2"   # v2: hoops located per camera segment


@dataclass(frozen=True)
class Hoop:
    """The rim and net in pixels of the inference frame."""

    cx: float          # rim center
    cy: float          # rim center line: the ball crosses this to go in
    left: float        # rim edges
    right: float
    top: float         # top of the rim ellipse
    net_bottom: float
    score: float
    frame: int         # the frame SAM looked at
    notes: tuple = ()  # why the mask looks contaminated (strap, post, ...); empty when clean

    @property
    def width(self) -> float:
        return self.right - self.left

    def to_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else list(v) if isinstance(v, tuple) else v
                for k, v in asdict(self).items()}


def _label_map(mask) -> np.ndarray | None:
    if not isinstance(mask, dict) or mask.get("format") != "png":
        return None
    raw = base64.b64decode(str(mask.get("data", "")).split(",", 1)[-1])
    labels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    return labels[:, :, 0] if labels.ndim == 3 else labels


def _runs(row: np.ndarray, merge: int = 2) -> list[tuple[int, int]]:
    """``[(first, last)]`` column runs of one mask row; gaps of <= *merge* px are joined."""
    cols = np.flatnonzero(row)
    if not cols.size:
        return []
    breaks = np.flatnonzero(np.diff(cols) > merge + 1)
    starts = np.r_[cols[0], cols[breaks + 1]]
    ends = np.r_[cols[breaks], cols[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def rim_from_mask(blob: np.ndarray, *, shrink: float = 0.9, tall: float = 1.3) -> dict | None:
    """The rim read top-down from a hoop mask (bool array).

    Going down from the mask's top row, the band of rim rows ends where the width
    falls below *shrink* × the widest row so far (the net tapering) or, once that
    maximum is reached, at a row that is both split into separate runs and wider
    than the rim (a strap or post beside the net). The rim rows are the band's rows
    within 3% of its widest. A mask taller than *tall* × that width also ends the
    band at any split row after the maximum (strict mode). Returns ``cy``,
    ``left``, ``right``, ``top``, ``net_bottom`` and ``notes``, or None for an
    empty mask.

    The net bottom is the last row of the mask that still reaches between the rim
    edges, so a post piece wholly beside the net does not lengthen it.
    """
    rows = np.flatnonzero(blob.any(axis=1))
    if not rows.size:
        return None
    runs = {r: _runs(blob[r]) for r in rows}
    span = {r: runs[r][-1][1] - runs[r][0][0] + 1 for r in rows}
    height = rows[-1] - rows[0] + 1
    strict = height > tall * max(span.values())
    band, best, peaked = [], 0, False
    for r in range(rows[0], rows[-1] + 1):
        if r not in span:
            if band:
                break           # a gap in the mask ends the rim band
            continue
        w = span[r]
        split = len(runs[r]) > 1
        if band and w < shrink * best:
            break
        if peaked and split and (w > 1.1 * best or strict):
            break
        band.append(r)
        if w > best:
            best = w
        elif w < best:
            peaked = True
    rim_rows = [r for r in band if span[r] >= 0.97 * best]
    cy = float(np.mean(rim_rows))
    cols = np.flatnonzero(blob[rim_rows].any(axis=0))
    left, right = float(cols.min()), float(cols.max() + 1)
    reach = [r for r in rows if any(a < right and b + 1 > left for a, b in runs[r])]
    net_bottom = float(max(reach) + 1)

    notes = []
    if strict:
        notes.append(f"the hoop mask is {height / (right - left):.1f}x taller than the rim is wide")
    below = [r for r in rows if r > band[-1] and len(runs[r]) > 1 and span[r] > right - left]
    if below:
        notes.append(f"{len(below)} mask row(s) below the rim are split into separate pieces "
                     "wider than the rim (a strap, post or bracket)")
    widest = max(span.values())
    old_rows = [r for r in rows if span[r] >= 0.97 * widest]
    moved = abs(float(np.mean(old_rows)) - cy)
    if moved > 0.3 * (right - left):
        notes.append(f"the widest mask rows are {moved:.0f} px ({moved / (right - left):.2f} rim "
                     "widths) below the rim read from the top; they are not the rim")
    return {"cy": cy, "left": left, "right": right, "top": float(rows[0]),
            "net_bottom": net_bottom, "notes": tuple(notes)}


def _parse(item: dict, labels, *, frame: int, width: int, height: int) -> Hoop:
    """One SAM item as a Hoop: the rim read off its mask, or off its box without one."""
    bx, by, bw, bh = item["bbox_xywh"]
    left, right, top, bottom = bx * width, (bx + bw) * width, by * height, (by + bh) * height
    cy = top + 0.2 * (bottom - top)   # box only: the rim is in its top fifth
    notes = ()
    instance = item.get("instance_id")
    if labels is not None and instance is not None:
        rim = rim_from_mask(labels == int(instance))
        if rim is not None:
            cy, left, right, top, bottom, notes = (rim["cy"], rim["left"], rim["right"],
                                                   rim["top"], rim["net_bottom"], rim["notes"])
    return Hoop(cx=(left + right) / 2, cy=cy, left=left, right=right, top=top,
                net_bottom=bottom, score=float(item.get("score") or 0), frame=frame,
                notes=tuple(notes))


def candidates(payload: dict, *, frame: int, width: int, height: int) -> list[Hoop]:
    """Every hoop in a SAM reply, widest rim first. Empty when SAM found none."""
    content = payload.get("content")
    if not isinstance(content, dict) or content.get("object") != CONTENT_OBJECT:
        raise RuntimeError(f"Expected a {CONTENT_OBJECT} payload, got "
                           f"{json.dumps(payload)[:300]}")
    labels = _label_map(content.get("mask"))
    hoops = [_parse(item, labels, frame=frame, width=width, height=height)
             for item in content.get("items") or [] if item.get("bbox_xywh")]
    return sorted((h for h in hoops if h.width > 0), key=lambda h: -h.width)


CHOICES = ("auto", "largest", "highest_score", "left", "right")


def choose(hoops: list[Hoop], mode: str = "auto") -> tuple[Hoop, str]:
    """``(hoop, why)`` among *hoops* (non-empty) for a choice mode in CHOICES.

    "auto" and "largest" take the widest rim: a camera filming a shooter is near
    the hoop being shot at, so it projects largest, while a far-court hoop on the
    back wall is small. (The reference took the highest score, which picked that
    far hoop on real footage.) "auto" may later be revised by ball_affinity.
    """
    if mode not in CHOICES:
        raise ValueError(f"hoop choice must be one of {CHOICES}")
    if len(hoops) == 1:
        return hoops[0], "only hoop found"
    if mode == "highest_score":
        return max(hoops, key=lambda h: h.score), "highest SAM score (setting)"
    if mode == "left":
        return min(hoops, key=lambda h: h.cx), "leftmost hoop (setting)"
    if mode == "right":
        return max(hoops, key=lambda h: h.cx), "rightmost hoop (setting)"
    return max(hoops, key=lambda h: h.width), (
        "largest rim (setting)" if mode == "largest" else "largest rim")


def ball_affinity(hoop: Hoop, x: np.ndarray, y: np.ndarray, frames=None) -> tuple[int, int]:
    """How much the ball track engages *hoop*: ``(downward rim-line crossings within a
    rim width of its center, observations near it)``. Compared lexicographically."""
    idx = np.flatnonzero(np.isfinite(x))
    if frames is not None:
        idx = idx[np.isin(idx, np.asarray(list(frames)))]
    w = hoop.width
    crossings = near = 0
    for a, b in zip(idx, idx[1:]):
        if b - a <= 6 and y[a] < hoop.cy <= y[b]:
            t = (hoop.cy - y[a]) / (y[b] - y[a])
            if abs(x[a] + t * (x[b] - x[a]) - hoop.cx) < w:
                crossings += 1
    for f in idx:
        if abs(x[f] - hoop.cx) < 1.5 * w and hoop.top - 3 * w < y[f] < hoop.net_bottom + w:
            near += 1
    return crossings, near


def prefer_by_ball(hoops: list[Hoop], current: Hoop, x, y, frames=None) -> tuple[Hoop, bool]:
    """The hoop the ball actually comes down through, if that is not *current*.

    Switches only on rim crossings (the strongest evidence): proximity alone is
    biased towards large hoops. Returns ``(hoop, switched)``.
    """
    scored = [(ball_affinity(h, x, y, frames)[0], h) for h in hoops]
    best_n, best = max(scored, key=lambda s: (s[0], s[1] is current, s[1].width))
    cur_n = next(n for n, h in scored if h is current)
    return (best, True) if best is not current and best_n > cur_n else (current, False)


@dataclass
class HoopSegment:
    start: int                 # frames [start, end) of the analyzed clip
    end: int
    hoop: Hoop | None          # None: no hoop located for this segment
    candidates: list           # every hoop SAM found here (list[Hoop])
    why: str = ""              # how the hoop was chosen

    def to_dict(self) -> dict:
        return {"start_frame": self.start, "end_frame": self.end,
                "hoop": self.hoop.to_dict() if self.hoop else None,
                "hoops_found": len(self.candidates), "chosen_by": self.why}


@dataclass
class HoopTimeline:
    """One hoop per stretch of the clip (camera segment, or part of a long one)."""

    segments: list

    def at(self, frame: int) -> Hoop | None:
        for seg in self.segments:
            if seg.start <= frame < seg.end:
                return seg.hoop
        return None

    def segment_index(self, frame: int) -> int | None:
        return next((i for i, s in enumerate(self.segments) if s.start <= frame < s.end), None)

    def hoops(self) -> list[Hoop]:
        return [s.hoop for s in self.segments if s.hoop is not None]


def locate(payload: dict, *, frame: int, width: int, height: int, mode: str = "auto") -> Hoop:
    """The chosen hoop in one SAM reply (see ``choose``)."""
    hoops = candidates(payload, frame=frame, width=width, height=height)
    if not hoops:
        raise RuntimeError("SAM found no hoop in the frame. Is the hoop in shot?")
    return choose(hoops, mode)[0]

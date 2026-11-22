"""Find the ball in the net when SAM loses it there (new, not in the reference).

A small ball inside the net, behind the net strings and blurred, often gets no
box from SAM at all. When a shot's track ends at the rim, this looks again in a
crop around the hoop, without any model call: the crop's per-pixel median over
the next ~1.5 s is the static scene (rim, net, backboard), and the ball is an
orange-brown blob that differs from it. A blob is accepted only on the ball's
continuation: within reach of the last sighting at the ball's speed limit,
inside the rim's width at the rim line, and then falling under the net. Pure
functions on images; the pipeline reads the frames.
"""

from __future__ import annotations

import cv2
import numpy as np


def crop_box(hoop, width: int, height: int) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)``: 4 rim widths wide, from half a rim width over the rim
    line to 4 rim widths under the net (where the ball falls out of it)."""
    w = hoop.width
    return (max(0, int(hoop.cx - 2 * w)), max(0, int(hoop.cy - 0.5 * w)),
            min(width, int(np.ceil(hoop.cx + 2 * w))), min(height, int(np.ceil(hoop.net_bottom + 4 * w))))


def blobs(crops: np.ndarray, *, min_diff: float = 60.0) -> list[list[tuple[float, float, int]]]:
    """Per crop, the ball-coloured moving blobs ``(cx, cy, area)`` in crop pixels.

    Moving = differs from the median of all *crops* by more than *min_diff* (sum
    over BGR). Ball-coloured = orange to red-brown hue, saturated, not dark."""
    stack = crops.astype(np.int16)
    background = np.median(stack, axis=0)
    out = []
    for c in stack:
        moving = np.abs(c - background).sum(axis=2) > min_diff
        hsv = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hue = hsv[..., 0]
        colour = ((hue <= 22) | (hue >= 170)) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 50)
        n, _, stats, centers = cv2.connectedComponentsWithStats((moving & colour).astype(np.uint8))
        out.append([(float(centers[i][0]), float(centers[i][1]), int(stats[i, 4]))
                    for i in range(1, n)])
    return out


def follow(found: list[list[tuple[float, float, int]]], first_frame: int, origin: tuple[int, float, float],
           hoop, diameter: float, fps: float, *, max_speed_mps: float = 12.0,
           min_area_fraction: float = 0.15, max_gap_seconds: float = 0.3,
           below_seconds: float = 0.6, below_rims: float = 4.0,
           offset: tuple[int, int] = (0, 0)) -> list[tuple[int, float, float]]:
    """The ball's continuation ``[(frame, x, y)]`` in frame pixels, from *origin*
    ``(frame, x, y)``, the last good sighting.

    Each step takes the blob nearest the previous point that is at least
    *min_area_fraction* of the ball's area (the net hides much of it), reachable
    at *max_speed_mps* (half a diameter of slack), and not rising more than half a
    diameter. The first point at or under the rim line must be inside the rim's
    width; above the rim line points must be within one rim width of the rim's
    center. Nothing is returned unless the ball reached the rim line inside the rim.

    Once the ball is at the net's bottom it is followed below the net for up to
    *below_seconds*: only within one rim width of the rim's center, down to
    *below_rims* rim widths under the net, so the ball falling out of the net is
    seen (and a person or ball beside the post is not). A gap longer than
    *max_gap_seconds* ends the search.
    """
    w = hoop.width
    ox, oy = offset
    min_area = min_area_fraction * np.pi * (diameter / 2) ** 2
    max_step = max_speed_mps / 0.239 * diameter / fps
    pf, px, py = origin
    path: list[tuple[int, float, float]] = []
    crossed = False
    below_from = None              # frame the ball reached the net's bottom
    floor = hoop.net_bottom + below_rims * w
    for k, options in enumerate(found):
        f = first_frame + k
        if f <= pf:
            continue
        if f - pf > max_gap_seconds * fps:
            break
        if below_from is not None and f - below_from > below_seconds * fps:
            break
        near = []
        for bx, by, area in options:
            x, y = bx + ox, by + oy
            if area < min_area or np.hypot(x - px, y - py) > max_step * (f - pf) + 0.5 * diameter:
                continue
            if y < py - 0.5 * diameter:
                continue
            if y < hoop.cy and abs(x - hoop.cx) > w:
                continue
            if y > hoop.net_bottom and (abs(x - hoop.cx) > w or y > floor):
                continue
            near.append((np.hypot(x - px, y - py), x, y))
        if not near:
            continue
        _, x, y = min(near)
        if not crossed and y >= hoop.cy:
            if abs(x - hoop.cx) > w / 2:
                return []          # came down outside the rim: not this search's business
            crossed = True
        path.append((f, x, y))
        pf, px, py = f, x, y
        if below_from is None and y >= hoop.net_bottom:
            below_from = f
    return path if crossed else []

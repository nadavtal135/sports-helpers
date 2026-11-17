# Adapted from jeremyipark/vision-demos basketball/src/hoop.py (Apache-2.0),
# commit c06cfb78ccd8e2c63629222695aac710fe4c199c. Modifications: the request
# moved to src/providers; cache keys moved to src/cache.py; rim parsing unchanged.
"""The hoop: SAM 3.1 on one frame, then the rim read off the mask.

SAM segments the hoop (rim + net). The rim is the mask's widest rows, since the
net tapers below it.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

CONTENT_OBJECT = "img.segment.masks"
CONTRACT = "sam31-hoop-v1"


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

    @property
    def width(self) -> float:
        return self.right - self.left

    def to_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def _label_map(mask) -> np.ndarray | None:
    if not isinstance(mask, dict) or mask.get("format") != "png":
        return None
    raw = base64.b64decode(str(mask.get("data", "")).split(",", 1)[-1])
    labels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    return labels[:, :, 0] if labels.ndim == 3 else labels


def locate(payload: dict, *, frame: int, width: int, height: int) -> Hoop:
    """The best-scoring hoop, with the rim read off its mask (or its box, if no mask)."""
    content = payload.get("content")
    if not isinstance(content, dict) or content.get("object") != CONTENT_OBJECT:
        raise RuntimeError(f"Expected a {CONTENT_OBJECT} payload, got "
                           f"{json.dumps(payload)[:300]}")
    items = content.get("items") or []
    if not items:
        raise RuntimeError("SAM found no hoop in the frame. Is the hoop in shot? "
                           "Try another HOOP_FRAME_SECONDS.")
    best = max(items, key=lambda i: float(i.get("score") or 0))
    bx, by, bw, bh = best["bbox_xywh"]
    left, right, top, bottom = bx * width, (bx + bw) * width, by * height, (by + bh) * height
    cy = top + 0.2 * (bottom - top)   # box only: the rim is in its top fifth

    labels = _label_map(content.get("mask"))
    instance = best.get("instance_id")
    if labels is not None and instance is not None:
        blob = labels == int(instance)
        rows = np.nonzero(blob.any(axis=1))[0]
        if rows.size:
            spans = np.array([np.ptp(np.nonzero(blob[r])[0]) + 1 for r in rows])
            # Rows within 3% of the widest are rim.
            rim_rows = rows[spans >= 0.97 * spans.max()]
            cy = float(np.mean(rim_rows))
            cols = np.nonzero(blob[rim_rows].any(axis=0))[0]
            left, right = float(cols.min()), float(cols.max() + 1)
            top, bottom = float(rows.min()), float(rows.max() + 1)
    return Hoop(cx=(left + right) / 2, cy=cy, left=left, right=right, top=top,
                net_bottom=bottom, score=float(best.get("score") or 0), frame=frame)

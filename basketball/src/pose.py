"""ViTPose replies: validation, per-frame grouping, duplicate suppression, arrays.

Adapted from jeremyipark/vision-demos ``basketball/src/pose.py`` and
``running/src/pose.py`` (Apache-2.0, commit c06cfb78ccd8e2c63629222695aac710fe4c199c).
The request itself lives in ``providers/gateway.py``; this module only reads
replies, so it never touches the network. Modification: ``observed`` masks are
returned alongside the keypoints so callers can tell model output from fill.
"""

from __future__ import annotations

import json

import numpy as np

from .skeleton import KPT_NAMES

CONTENT_OBJECT_KPTS = "vid.pose.kpts"


def unwrap(payload: dict) -> list[dict]:
    """``[{"index": frame, "persons": [...]}, ...]`` sorted by frame; empty frames kept."""
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, dict):
        raise RuntimeError(f"Unexpected pose response envelope: {json.dumps(payload)[:300]}")
    if content.get("object") != CONTENT_OBJECT_KPTS:
        raise RuntimeError(f"Expected a video pose payload ({CONTENT_OBJECT_KPTS}), "
                           f"got {content.get('object')!r}.")
    labels = content.get("kpts_labels")
    if labels is not None and list(labels) != KPT_NAMES:
        raise RuntimeError(f"Unexpected joint order {list(labels)}; expected COCO-17.")
    by_index: dict[int, list[dict]] = {int(f["frame_id"]): [] for f in content.get("frames") or []}
    for person in content.get("items") or []:
        if person.get("frame_id") is not None:
            by_index.setdefault(int(person["frame_id"]), []).append(person)
    return [{"index": i, "persons": by_index[i]} for i in sorted(by_index)]


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = (max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
             * max(0.0, min(ay + ah, by + bh) - max(ay, by)))
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def dedupe(frames: list[dict], *, max_persons: int | None, iou: float) -> int:
    """Drop duplicate boxes per frame, keeping the best match to the previous frame."""
    previous = None
    dropped = 0
    for frame in frames:
        persons = frame.get("persons") or []
        if not persons:
            continue
        order = sorted(persons, key=lambda p: (
            -(_iou(p["bbox_xywh"], previous) if previous is not None else 0.0),
            -float(p["bbox_xywh"][2] * p["bbox_xywh"][3])))
        kept: list[dict] = []
        for person in order:
            if all(_iou(person["bbox_xywh"], k["bbox_xywh"]) < iou for k in kept):
                kept.append(person)
        if max_persons:
            kept = kept[:max_persons]
        dropped += len(persons) - len(kept)
        frame["persons"] = kept
        if kept:
            previous = kept[0]["bbox_xywh"]
    return dropped


def keypoints(frames: list[dict], n_frames: int, width: float = 1.0, height: float = 1.0
              ) -> np.ndarray:
    """``(n_frames, 17, 2)`` scaled by ``(width, height)``; NaN where not observed.

    (0, 0) is the model's "not visible" sentinel. Frames the reply does not
    mention stay NaN: nothing is interpolated here.
    """
    out = np.full((n_frames, len(KPT_NAMES), 2), np.nan)
    for frame in frames:
        i = frame["index"]
        if not 0 <= i < n_frames or not frame["persons"]:
            continue
        kpts = np.asarray(frame["persons"][0].get("kpts_xy") or [], float)
        if kpts.shape != (len(KPT_NAMES), 2):
            continue
        missing = (kpts == 0).all(axis=1) | ~np.isfinite(kpts).all(axis=1)
        kpts = kpts * [width, height]
        kpts[missing] = np.nan
        out[i] = kpts
    return out

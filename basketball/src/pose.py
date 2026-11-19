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


def pick_shooter(frames: list[dict], *, min_frames: int = 3) -> int | None:
    """The track id of the person who shoots: the one with both wrists above the
    head on the most frames. None when the reply has no track ids or nobody raised
    both hands on at least *min_frames* frames; the caller then keeps the old rule.

    New (not in the reference): the reference kept the largest person, which on
    real game footage with ten players followed a bystander, so no shot was found.
    """
    counts: dict[int, int] = {}
    for frame in frames:
        for person in frame.get("persons") or []:
            tid = person.get("track_id")
            k = np.asarray(person.get("kpts_xy") or [], float)
            if tid is None or k.shape != (17, 2):
                continue
            head = k[:5][(k[:5] != 0).all(axis=1)]
            wrists = k[[9, 10]]
            if len(head) and (wrists != 0).all() and wrists[:, 1].max() < head[:, 1].min():
                counts[tid] = counts.get(tid, 0) + 1
    if not counts:
        return None
    tid, n = max(counts.items(), key=lambda kv: kv[1])
    return tid if n >= min_frames else None


def keep_track(frames: list[dict], track_id: int) -> None:
    """Keep only *track_id*'s detections; frames without it become empty (not posed)."""
    for frame in frames:
        frame["persons"] = [p for p in frame.get("persons") or [] if p.get("track_id") == track_id]


def _center(box) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _links(box, last, *, iou: float, reach: float) -> bool:
    """Whether *box* plausibly continues *last*: overlapping, or its center within
    *reach* box-sizes of the last one (a short gap, a re-detection)."""
    if _iou(box, last) >= iou:
        return True
    (ax, ay), (bx, by) = _center(box), _center(last)
    size = max(last[2], last[3], 1e-6)
    return float(np.hypot(ax - bx, ay - by)) <= reach * size


def follow(frames: list[dict], seed: int, *, iou: float = 0.3, reach: float = 0.35) -> int:
    """Keep one person per frame: track *seed*, re-linked across tracker id changes.

    Trackers hand out a new id after an occlusion or a re-detection, so keeping
    one id (``keep_track``) can drop most of a clip; on a real workout clip it
    kept 461 of 2639 frames. Walking out from the seed's first frame in both
    directions, each frame keeps, in order of preference: the current id, the
    seed id, or the person whose box continues the last kept box (IoU or a short
    hop); the current id then follows that person. Frames with no match are left
    empty. Returns how many id changes were re-linked.
    """
    first = next((k for k, f in enumerate(frames)
                  if any(p.get("track_id") == seed for p in f.get("persons") or [])), None)
    if first is None:
        return 0
    relinks = 0
    keep: dict[int, dict | None] = {}
    for order in (range(first, len(frames)), range(first - 1, -1, -1)):
        current, last = seed, None
        if order.start < first:     # backward pass starts from the seed's first box
            seed_person = next(p for p in frames[first]["persons"] if p.get("track_id") == seed)
            last = seed_person["bbox_xywh"]
        for k in order:
            persons = frames[k].get("persons") or []
            pick = (next((p for p in persons if p.get("track_id") == current), None)
                    or next((p for p in persons if p.get("track_id") == seed), None))
            if pick is None and last is not None:
                near = [p for p in persons if _links(p["bbox_xywh"], last, iou=iou, reach=reach)]
                if near:
                    pick = max(near, key=lambda p: _iou(p["bbox_xywh"], last))
                    relinks += 1
            if pick is not None:
                current, last = pick.get("track_id"), pick["bbox_xywh"]
            keep[k] = pick
    for k, frame in enumerate(frames):
        frame["persons"] = [keep[k]] if keep.get(k) is not None else []
    return relinks


def select_shooter(frames: list[dict], spans: list[tuple[int, int]], *, min_frames: int = 3
                   ) -> dict:
    """Pick and follow the shooter separately in each camera segment.

    *spans* are ``[(start, end), ...]`` frame-index ranges (camera segments).
    In each one the shooter is the person with both wrists overhead on the most
    frames (``pick_shooter``); failing that, the clip-wide shooter if present
    there. The pick is then followed through id changes (``follow``). A segment
    with one person and no tracker ids is left alone, and a segment with several
    people and no evident shooter is left for ``dedupe`` (largest person), as
    before. Mutates *frames*; returns a summary for the run metadata.
    """
    clip_shooter = pick_shooter(frames, min_frames=min_frames)
    summary = {"segments": [], "relinks": 0, "unresolved_segments": 0}
    for start, end in spans:
        seg = [f for f in frames if start <= f["index"] < end]
        ids = {p.get("track_id") for f in seg for p in f.get("persons") or []}
        crowded = any(len(f.get("persons") or []) > 1 for f in seg)
        if not seg or (None in ids and not crowded):
            summary["segments"].append(None)
            continue
        seed = pick_shooter(seg, min_frames=min_frames)
        if seed is None and clip_shooter in ids:
            seed = clip_shooter
        if seed is None and len(ids - {None}) > 1 and not crowded:
            # One person at a time, but their id changes: follow whoever is there.
            seed = next(p.get("track_id") for f in seg for p in f.get("persons") or [])
        if seed is None:
            summary["segments"].append(None)
            summary["unresolved_segments"] += 1 if crowded else 0
            continue
        summary["relinks"] += follow(seg, seed)
        summary["segments"].append(seed)
    return summary

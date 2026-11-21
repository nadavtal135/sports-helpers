"""Observed side-view pose coordinates and clip-wide side selection."""

import numpy as np

from . import pose
from .skeleton import KPT_NAMES

JOINTS = {side: [KPT_NAMES.index(f"{side}_{j}") for j in ("hip", "knee", "ankle", "shoulder")]
          for side in ("left", "right")}


def observations(frames, n, width, height, min_score):
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


def choose_side(points, scores):
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

"""Draw a synthetic side view and encode the same scene as a ViTPose reply."""

import cv2
import numpy as np

from .skeleton import KPT_NAMES

J = {name: i for i, name in enumerate(KPT_NAMES)}


def draw_figure(points, width, height, floor_y, label):
    img = np.full((height, width, 3), (48, 43, 38), np.uint8)
    cv2.rectangle(img, (0, floor_y), (width, height), (70, 66, 61), -1)
    cv2.line(img, (0, floor_y), (width, floor_y), (139, 125, 112), 2)
    pts = points.round().astype(int)
    for side, color in (("left", (136, 123, 109)), ("right", (157, 200, 228))):
        def point(name):
            return tuple(pts[J[f"{side}_{name}"]])
        for a, b in (("hip", "knee"), ("knee", "ankle"), ("shoulder", "hip"),
                     ("shoulder", "elbow"), ("elbow", "wrist")):
            cv2.line(img, point(a), point(b), color, 20 if a == "shoulder" and b == "hip" else 12, cv2.LINE_AA)
        x, y = point("ankle")
        cv2.line(img, (x - 5, y + 6), (x + 25, y + 6), color, 10, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[J["right_ear"]]), 23, (157, 200, 228), -1, cv2.LINE_AA)
    cv2.putText(img, label, (18, height - 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def fixture_poses(source_offset, n_frames, fps, pose_at, size, hidden) -> dict:
    items = []
    for k in range(n_frames):
        t = (source_offset + k) / fps
        pts = pose_at(t) / size
        lo, hi = pts.min(axis=0) - 0.03, pts.max(axis=0) + 0.03
        scores = [0.70 if name.startswith("left") else 0.94 for name in KPT_NAMES]
        for joint in hidden(t):
            scores[joint] = 0.0
            pts[joint] = 0.0
        items.append({"frame_id": k, "track_id": 0, "score": 0.95, "kpts_xy": pts.round(4).tolist(),
                      "kpts_score": scores, "bbox_xywh": [*lo, *(hi - lo)]})
    return {"content": {"object": "vid.pose.kpts", "kpts_labels": KPT_NAMES,
                        "frames": [{"frame_id": k} for k in range(n_frames)], "items": items}}

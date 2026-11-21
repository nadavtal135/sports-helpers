"""The squat demo and its gateway-shaped poses come from the same side-view scene."""

import math
from pathlib import Path

import cv2
import numpy as np

from running.sample import write_sample
from running.src.skeleton import KPT_NAMES

W, H = 480, 640
FPS, DURATION = 30.0, 12.0
# Descent starts, arrives at bottom, leaves bottom, stands, thigh angle at bottom.
CYCLES = ((0.8, 1.8, 1.8, 2.8, 104), (3.3, 4.3, 5.1, 6.1, 90),
          (6.6, 7.4, 7.4, 8.2, 50), (8.8, 9.9, 9.9, 11.0, 100))
GAP = (9.55, 10.25)   # both knees obscured in the fixture reply
J = {name: i for i, name in enumerate(KPT_NAMES)}


def pose_at(t: float) -> np.ndarray:
    bend, thigh = 0.0, 0.0
    for start, bottom, leave, end, angle in CYCLES:
        if start <= t <= end:
            if t < bottom:
                u = (t - start) / (bottom - start)
            elif t <= leave:
                u = 1.0
            else:
                u = (end - t) / (end - leave)
            bend = 0.5 - 0.5 * math.cos(math.pi * u)
            thigh = angle * bend
            break

    def segment(length, degrees):
        a = math.radians(degrees)
        return length * np.array([math.sin(a), -math.cos(a)])

    pts = np.empty((17, 2))
    for side, dx in (("left", -8), ("right", 0)):
        ankle = np.array([265 + dx, 555])
        knee = ankle + segment(108, 3 + 23 * bend)
        hip = knee + segment(115, -3 - thigh)
        shoulder = hip + segment(135, 6 + 27 * bend)
        elbow = shoulder + segment(69, 155 - 58 * bend)
        wrist = elbow + segment(64, 142 - 64 * bend)
        for joint, point in zip(("ankle", "knee", "hip", "shoulder", "elbow", "wrist"),
                                (ankle, knee, hip, shoulder, elbow, wrist)):
            pts[J[f"{side}_{joint}"]] = point
        ear = shoulder + [6, -35]
        pts[J[f"{side}_ear"]] = ear
        pts[J[f"{side}_eye"]] = ear + [12, -4]
    pts[J["nose"]] = pts[J["right_eye"]] + [7, 6]
    return pts


def draw_frame(t: float):
    img = np.full((H, W, 3), (48, 43, 38), np.uint8)
    cv2.rectangle(img, (0, 568), (W, H), (70, 66, 61), -1)
    cv2.line(img, (0, 568), (W, 568), (139, 125, 112), 2)
    pts = pose_at(t).round().astype(int)
    for side, color in (("left", (136, 123, 109)), ("right", (157, 200, 228))):
        def point(name):
            return tuple(pts[J[f"{side}_{name}"]])
        for a, b in (("hip", "knee"), ("knee", "ankle"), ("shoulder", "hip"),
                     ("shoulder", "elbow"), ("elbow", "wrist")):
            cv2.line(img, point(a), point(b), color, 20 if a == "shoulder" and b == "hip" else 12, cv2.LINE_AA)
        x, y = point("ankle")
        cv2.line(img, (x - 5, y + 6), (x + 25, y + 6), color, 10, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[J["right_ear"]]), 23, (157, 200, 228), -1, cv2.LINE_AA)
    cv2.putText(img, "SYNTHETIC SQUAT SAMPLE", (18, H - 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def make_sample(dst: Path) -> Path:
    return write_sample(dst, size=(W, H), fps=FPS, duration=DURATION, draw_frame=draw_frame)


def pose_payload(source_offset: int, n_frames: int, fps: float) -> dict:
    items = []
    for k in range(n_frames):
        t = (source_offset + k) / fps
        pts = pose_at(t) / (W, H)
        lo, hi = pts.min(axis=0) - 0.03, pts.max(axis=0) + 0.03
        scores = [0.70 if name.startswith("left") else 0.94 for name in KPT_NAMES]
        if GAP[0] <= t < GAP[1]:
            for j in (J["left_knee"], J["right_knee"]):
                scores[j] = 0.0
                pts[j] = 0.0
        items.append({"frame_id": k, "track_id": 0, "score": 0.95, "kpts_xy": pts.round(4).tolist(),
                      "kpts_score": scores, "bbox_xywh": [*lo, *(hi - lo)]})
    return {"content": {"object": "vid.pose.kpts", "kpts_labels": KPT_NAMES,
                        "frames": [{"frame_id": k} for k in range(n_frames)], "items": items}}


class FixtureProvider:
    name = "fixture"
    simulated = True

    def pose(self, video: Path, *, model: str, extra_body: dict, ctx: dict):
        return pose_payload(ctx["source_offset"], ctx["n_frames"], ctx["fps"]), {"cost": 0}

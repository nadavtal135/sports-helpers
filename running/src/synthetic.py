"""A synthetic treadmill runner, side-on: one parametric scene that draws the
fixture-mode sample clip and answers the fixture provider's "model" calls.

Not a model. Ground truth: cadence ``CADENCE`` steps/min exactly, a stance of
``STANCE`` of each stride. The fixture reply also drops the right leg for
``GAP`` (a simulated tracking loss) so the gap handling shows up in the sample.
"""

from __future__ import annotations

import math

import numpy as np

from .skeleton import KPT_NAMES

W, H = 960, 540
FPS = 30.0
DURATION = 10.0
CADENCE = 172.0
STRIDE = 120.0 / CADENCE        # seconds per same-foot stride
STANCE = 0.36                   # share of the stride with the foot down
L = 220.0                       # leg length px (thigh = shank = L / 2)
HIP_X, HIP_Y0 = 470.0, 255.0
GROUND_Y = HIP_Y0 + 195.0
GAP = (5.0, 5.4)                # seconds with no right-leg keypoints in the fixture reply

J = {n: i for i, n in enumerate(KPT_NAMES)}


def _ankle(phase: float) -> tuple[float, float]:
    reach = 0.3 * L
    if phase < STANCE:
        u = phase / STANCE
        return HIP_X + reach * (1 - 2 * u), GROUND_Y
    u = (phase - STANCE) / (1 - STANCE)
    x = HIP_X - reach * math.cos(math.pi * u)
    y = GROUND_Y - 0.5 * L * math.sin(math.pi * u) ** 1.5
    return x, y


def _knee(hip, ankle):
    half = L / 2
    d = np.subtract(ankle, hip)
    dist = min(float(np.hypot(*d)), L - 1e-3)
    d = d / (np.hypot(*d) or 1) * dist
    a = dist / 2
    h = math.sqrt(max(0.0, half ** 2 - a ** 2))
    perp = np.array([-d[1], d[0]]) / dist
    if perp[0] < 0:       # knees bend forward (+x: the runner faces right)
        perp = -perp
    return np.asarray(hip) + d / 2 + h * perp


def pose_at(t: float) -> np.ndarray:
    pts = np.zeros((17, 2))
    hip_y = HIP_Y0 + 6 * math.cos(4 * math.pi * t / STRIDE)
    for foot, shift, dx in (("right", 0.0, 4.0), ("left", 0.5, -4.0)):
        phase = (t / STRIDE + shift) % 1.0
        hip = (HIP_X + dx, hip_y)
        ankle = _ankle(phase)
        pts[J[f"{foot}_hip"]] = hip
        pts[J[f"{foot}_ankle"]] = ankle
        pts[J[f"{foot}_knee"]] = _knee(hip, ankle)
        # Arms swing opposite to the legs.
        sh = (HIP_X + 14 + dx, hip_y - 0.62 * L)
        swing = math.sin(2 * math.pi * (phase + 0.5))
        elbow = (sh[0] + 40 * swing, sh[1] + 70)
        wrist = (elbow[0] + 45, elbow[1] - 25 + 15 * swing)
        pts[J[f"{foot}_shoulder"]], pts[J[f"{foot}_elbow"]], pts[J[f"{foot}_wrist"]] = sh, elbow, wrist
    head = (HIP_X + 30, hip_y - 0.62 * L - 45)
    pts[J["nose"]] = (head[0] + 18, head[1])
    pts[J["left_eye"]] = (head[0] + 12, head[1] - 8)
    pts[J["right_eye"]] = (head[0] + 14, head[1] - 8)
    pts[J["left_ear"]] = (head[0] - 6, head[1] - 4)
    pts[J["right_ear"]] = (head[0] - 2, head[1] - 4)
    return pts


def pose_payload(source_offset: int, n_frames: int, fps: float = FPS, precision: int = 4,
                 noise: float = 0.0015, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed + source_offset)
    items = []
    for k in range(n_frames):
        t = (source_offset + k) / fps
        pts = pose_at(t) / (W, H)
        pts = pts + rng.normal(0, noise, pts.shape)
        kpts = [[round(float(x), precision), round(float(y), precision)] for x, y in pts]
        if GAP[0] <= t < GAP[1]:
            for name in ("right_hip", "right_knee", "right_ankle"):
                kpts[J[name]] = [0, 0]
        lo, hi = pts.min(axis=0) - 0.02, pts.max(axis=0) + 0.02
        items.append({"frame_id": k, "track_id": 0, "score": 0.95, "kpts_xy": kpts,
                      "kpts_score": [0.9] * 17,
                      "bbox_xywh": [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])]})
    return {"content": {"object": "vid.pose.kpts", "kpts_labels": list(KPT_NAMES),
                        "frames": [{"frame_id": k} for k in range(n_frames)], "items": items}}


def draw_frame(t: float):
    import cv2

    img = np.full((H, W, 3), (66, 60, 56), np.uint8)
    cv2.rectangle(img, (120, int(GROUND_Y) + 6), (840, int(GROUND_Y) + 26), (40, 40, 40), -1)
    belt = int((t * 400) % 40)
    for x in range(120 - belt, 840, 40):
        cv2.line(img, (x, int(GROUND_Y) + 8), (x, int(GROUND_Y) + 24), (90, 90, 90), 2)
    p = pose_at(t).astype(int)
    for a, b, col in [(11, 13, (150, 120, 220)), (13, 15, (150, 120, 220)),
                      (12, 14, (120, 170, 240)), (14, 16, (120, 170, 240)),
                      (5, 11, (200, 190, 180)), (5, 7, (200, 190, 180)), (7, 9, (200, 190, 180)),
                      (6, 8, (200, 190, 180)), (8, 10, (200, 190, 180))]:
        cv2.line(img, tuple(p[a]), tuple(p[b]), col, 12, cv2.LINE_AA)
    cv2.circle(img, (int(HIP_X + 30), int(p[0][1])), 26, (200, 190, 180), -1, cv2.LINE_AA)
    cv2.putText(img, "SYNTHETIC SAMPLE - fixture mode", (W - 470, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img

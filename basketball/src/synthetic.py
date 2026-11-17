"""A synthetic shooting session: one parametric scene that both draws the
fixture-mode sample clip and answers the fixture provider's "model" calls.

Nothing here is a real model. It exists so the whole app (upload, conversion,
analysis, rendering, exports) can run without credentials or paid calls, on a
clip whose ground truth is known. Each shot's scripted outcome:

1. make, with the ball briefly hidden in the net
2. rim-out: drops inside the rim line, bounces back over it, falls outside
3. make
4. a simulated tracker dropout: the ball crosses inside the rim and SAM
   "loses" it, so the correct call is UNKNOWN (the reference called it a make)
5. make

Scene units are pixels of a 1280x720 frame; replies are normalized like the
gateway's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .skeleton import KPT_NAMES

W, H = 1280, 720
FPS = 30.0
DURATION = 14.5

RIM_CX, RIM_CY, RIM_W, RIM_H = 900.0, 260.0, 70.0, 12.0
NET_BOTTOM = 332.0
BALL_D = 34.0
FLOOR_Y = 640.0
G = 900.0   # px/s^2

SHOTS = [   # (hands-up time s, shooter x, shoulder-to-ankle px, outcome)
    (1.6, 420.0, 240.0, "make"),
    (4.1, 300.0, 205.0, "rimout"),
    (6.6, 560.0, 260.0, "make"),
    (9.1, 420.0, 225.0, "lost"),
    (12.0, 250.0, 185.0, "make"),
]
EXPECTED = ["make", "miss", "make", "unknown", "make"]

J = {n: i for i, n in enumerate(KPT_NAMES)}


@dataclass
class _Pose:
    pts: np.ndarray        # (17, 2) px
    box: tuple             # x, y, w, h px


def _lerp(a, b, t):
    return a + (b - a) * t


def _smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def _shooter_at(t: float) -> tuple[float, float]:
    """Shooter x and size, gliding between shot spots."""
    for i, (t0, x, b, _) in enumerate(SHOTS):
        if t < t0 + 1.2:
            if i == 0:
                return x, b
            p0, p1 = SHOTS[i - 1], SHOTS[i]
            u = _smooth((t - (p0[0] + 1.2)) / max(0.1, p1[0] - 0.8 - (p0[0] + 1.2)))
            return _lerp(p0[1], x, u), _lerp(p0[2], b, u)
    return SHOTS[-1][1], SHOTS[-1][2]


def _lift(t: float) -> float:
    """0 = hands at the hips, 0.8 = set point, 1 = full overhead extension."""
    for t0, *_ in SHOTS:
        if t0 - 0.6 <= t < t0:
            return 0.8 * _smooth((t - (t0 - 0.6)) / 0.6)
        if t0 <= t < t0 + 0.15:
            return _lerp(0.8, 1.0, (t - t0) / 0.15)
        if t0 + 0.15 <= t < t0 + 0.6:
            return 1.0
        if t0 + 0.6 <= t < t0 + 1.1:
            return 1.0 - _smooth((t - t0 - 0.6) / 0.5)
    return 0.0


def _ik(shoulder, wrist, upper, fore, bend=1.0):
    d = np.subtract(wrist, shoulder)
    dist = min(np.hypot(*d), upper + fore - 1e-3)
    d = d / (np.hypot(*d) or 1) * dist
    a = (upper ** 2 - fore ** 2 + dist ** 2) / (2 * dist)
    h = math.sqrt(max(0.0, upper ** 2 - a ** 2))
    base = np.asarray(shoulder) + d * a / dist
    perp = np.array([-d[1], d[0]]) / dist
    if perp[0] < 0:
        perp = -perp
    return base + bend * h * perp, np.asarray(shoulder) + d


def pose_at(t: float) -> _Pose:
    x, b = _shooter_at(t)
    pts = np.zeros((17, 2))
    sh_y, hip_y, knee_y = FLOOR_Y - b, FLOOR_Y - 0.5 * b, FLOOR_Y - 0.25 * b
    nose_y = sh_y - 0.17 * b
    pts[J["nose"]] = (x + 0.05 * b, nose_y)
    pts[J["left_eye"]] = (x + 0.04 * b, nose_y - 0.03 * b)
    pts[J["right_eye"]] = (x + 0.06 * b, nose_y - 0.03 * b)
    pts[J["left_ear"]] = (x - 0.02 * b, nose_y - 0.02 * b)
    pts[J["right_ear"]] = (x + 0.0 * b, nose_y - 0.02 * b)
    pts[J["left_shoulder"]] = (x - 0.05 * b, sh_y)
    pts[J["right_shoulder"]] = (x + 0.05 * b, sh_y)
    pts[J["left_hip"]] = (x - 0.04 * b, hip_y)
    pts[J["right_hip"]] = (x + 0.04 * b, hip_y)
    pts[J["left_knee"]] = (x - 0.03 * b, knee_y)
    pts[J["right_knee"]] = (x + 0.05 * b, knee_y)
    pts[J["left_ankle"]] = (x - 0.05 * b, FLOOR_Y)
    pts[J["right_ankle"]] = (x + 0.04 * b, FLOOR_Y)
    u = _lift(t)
    arm = 0.28 * b
    for side, dx in (("right", 0.10), ("left", 0.02)):
        rest = (x + 0.15 * b, hip_y - 0.05 * b)
        top = (x + dx * b, sh_y - 0.55 * b)
        wrist = (_lerp(rest[0], top[0], u), _lerp(rest[1], top[1], u))
        elbow, wrist = _ik(pts[J[f"{side}_shoulder"]], wrist, arm, arm)
        pts[J[f"{side}_elbow"]], pts[J[f"{side}_wrist"]] = elbow, wrist
    lo, hi = pts.min(axis=0) - 10, pts.max(axis=0) + 10
    return _Pose(pts, (lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]))


def _hands(t: float) -> np.ndarray:
    p = pose_at(t).pts
    mid = (p[J["left_wrist"]] + p[J["right_wrist"]]) / 2
    return mid - (0, 0.6 * BALL_D)


def ball_at(t: float) -> tuple[float, float, bool] | None:
    """``(x, y, trackable)``: where the ball is drawn, and whether the fake SAM sees it."""
    for t0, _, _, outcome in SHOTS:
        tr = t0 + 0.12
        if not (t0 - 0.8 <= t < t0 + 2.6):
            continue
        if t < tr:
            hx, hy = _hands(t)
            return hx, hy, True
        x0, y0 = _hands(tr)
        tf = 1.0
        target_x = RIM_CX + (0.62 * RIM_W / 2 if outcome == "rimout" else 0.0)
        vx = (target_x - x0) / tf
        vy = (RIM_CY + 4 - y0 - 0.5 * G * tf ** 2) / tf
        s = t - tr
        if s <= tf:
            return x0 + vx * s, y0 + vy * s + 0.5 * G * s * s, True
        s -= tf
        if outcome == "rimout":
            # Bounces up off the rim's back edge and falls outside it.
            bx = target_x + 160 * s
            by = RIM_CY + 4 - 260 * s + 0.5 * G * s * s
            return (bx, min(by, FLOOR_Y - BALL_D / 2), True) if s < 1.4 else None
        # Down through the net (slowed), then free fall to the floor.
        net_t = 0.30
        if s <= net_t:
            y = _lerp(RIM_CY + 4, NET_BOTTOM + 6, s / net_t)
            hidden = 0.10 <= s <= 0.17          # the net hides it for a couple of frames
            visible = not hidden and outcome != "lost"
            return RIM_CX, y, visible
        s -= net_t
        y = NET_BOTTOM + 6 + 120 * s + 0.5 * G * s * s
        if y > FLOOR_Y - BALL_D / 2:
            return None
        return RIM_CX + 30 * s, y, outcome != "lost"
    return None


def hoop_mask(width: int, height: int) -> np.ndarray:
    import cv2

    sx, sy = width / W, height / H
    m = np.zeros((height, width), np.uint8)
    cv2.ellipse(m, (int(RIM_CX * sx), int(RIM_CY * sy)),
                (int(RIM_W / 2 * sx), int(RIM_H / 2 * sy)), 0, 0, 360, 1, -1)
    net = np.array([[RIM_CX - RIM_W * 0.45, RIM_CY], [RIM_CX + RIM_W * 0.45, RIM_CY],
                    [RIM_CX + RIM_W * 0.28, NET_BOTTOM], [RIM_CX - RIM_W * 0.28, NET_BOTTOM]])
    cv2.fillPoly(m, [np.round(net * (sx, sy)).astype(np.int32)], 1)
    return m


# ── drawing the sample clip ──────────────────────────────────────────────────

def draw_frame(t: float):
    import cv2

    img = np.full((H, W, 3), (58, 62, 70), np.uint8)
    img[int(FLOOR_Y) - 20:] = (70, 110, 150)
    cv2.rectangle(img, (int(RIM_CX - 90), int(RIM_CY - 120)), (int(RIM_CX + 60), int(RIM_CY - 10)),
                  (235, 235, 235), 3)
    net = hoop_mask(W, H)
    img[net > 0] = (210, 210, 210)
    cv2.ellipse(img, (int(RIM_CX), int(RIM_CY)), (int(RIM_W / 2), int(RIM_H / 2)), 0, 0, 360,
                (30, 90, 230), 3, cv2.LINE_AA)
    p = pose_at(t).pts.astype(int)
    for a, b in [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
                 (11, 13), (13, 15), (12, 14), (14, 16)]:
        cv2.line(img, tuple(p[a]), tuple(p[b]), (200, 170, 150), 9, cv2.LINE_AA)
    cv2.circle(img, tuple(p[0]), 18, (200, 170, 150), -1, cv2.LINE_AA)
    ball = ball_at(t)
    if ball is None:
        ball = (*_hands(t), True)
    cv2.circle(img, (int(ball[0]), int(ball[1])), int(BALL_D / 2), (20, 110, 230), -1, cv2.LINE_AA)
    cv2.putText(img, "SYNTHETIC SAMPLE - fixture mode", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ── replies in the gateway's shapes ──────────────────────────────────────────

def pose_payload(source_offset: int, n_frames: int, fps: float = FPS, precision: int = 4) -> dict:
    items = []
    for k in range(n_frames):
        pose = pose_at((source_offset + k) / fps)
        kpts = [[round(px / W, precision), round(py / H, precision)] for px, py in pose.pts]
        bx, by, bw, bh = pose.box
        items.append({"frame_id": k, "track_id": 0, "score": 0.95, "kpts_xy": kpts,
                      "kpts_score": [0.9] * 17,
                      "bbox_xywh": [bx / W, by / H, bw / W, bh / H]})
    return {"content": {"object": "vid.pose.kpts", "kpts_labels": list(KPT_NAMES),
                        "frames": [{"frame_id": k} for k in range(n_frames)], "items": items}}


def track_payload(source_offset: int, n_frames: int, fps: float = FPS) -> dict:
    items = []
    for k in range(n_frames):
        ball = ball_at((source_offset + k) / fps)
        if ball is None or not ball[2]:
            continue
        x, y, _ = ball
        items.append({"frame_id": k, "instance_id": 1, "score": 0.9,
                      "bbox_xywh": [(x - BALL_D / 2) / W, (y - BALL_D / 2) / H, BALL_D / W, BALL_D / H]})
        if NET_BOTTOM - 70 < y < NET_BOTTOM and abs(x - RIM_CX) < RIM_W:
            # SAM's box grown into the net: wrong size, must be filtered out.
            g = BALL_D * 1.9
            items.append({"frame_id": k, "instance_id": 2, "score": 0.6,
                          "bbox_xywh": [(x - g / 2) / W, (y - g / 2) / H, g / W, g * 1.3 / H]})
    return {"content": {"object": "vid.segment.masks", "items": items}}


def segment_payload(width: int, height: int) -> dict:
    import base64

    import cv2

    mask = hoop_mask(width, height)
    ok, png = cv2.imencode(".png", mask)
    ys, xs = np.nonzero(mask)
    box = [xs.min() / width, ys.min() / height, (xs.max() - xs.min() + 1) / width,
           (ys.max() - ys.min() + 1) / height]
    return {"content": {"object": "img.segment.masks",
                        "items": [{"instance_id": 1, "score": 0.93, "bbox_xywh": box}],
                        "mask": {"format": "png",
                                 "data": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()}}}

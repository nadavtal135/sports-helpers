"""The squat demo and its gateway-shaped poses come from the same side-view scene."""

import math
from pathlib import Path

import numpy as np

from running.sample import write_sample
from running.src.side_view_sample import J, draw_figure, fixture_poses

W, H = 480, 640
FPS, DURATION = 30.0, 12.0
# Descent starts, arrives at bottom, leaves bottom, stands, thigh angle at bottom.
CYCLES = ((0.8, 1.8, 1.8, 2.8, 104), (3.3, 4.3, 5.1, 6.1, 90),
          (6.6, 7.4, 7.4, 8.2, 50), (8.8, 9.9, 9.9, 11.0, 100))
GAP = (9.55, 10.25)   # both knees obscured in the fixture reply


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
    return draw_figure(pose_at(t), W, H, 568, "SYNTHETIC SQUAT SAMPLE")


def make_sample(dst: Path) -> Path:
    return write_sample(dst, size=(W, H), fps=FPS, duration=DURATION, draw_frame=draw_frame)


def pose_payload(source_offset: int, n_frames: int, fps: float) -> dict:
    def hidden(t):
        return (J["left_knee"], J["right_knee"]) if GAP[0] <= t < GAP[1] else ()
    return fixture_poses(source_offset, n_frames, fps, pose_at, (W, H), hidden)


class FixtureProvider:
    name = "fixture"
    simulated = True

    def pose(self, video: Path, *, model: str, extra_body: dict, ctx: dict):
        return pose_payload(ctx["source_offset"], ctx["n_frames"], ctx["fps"]), {"cost": 0}

"""A deterministic side-view jump scene. The floor, shoes and pose share a frame clock."""

import math
from pathlib import Path

import cv2
import numpy as np

from running.sample import write_sample
from running.src.side_view_sample import J, draw_figure, fixture_poses

W, H, FPS = 480, 680, 30.0
DURATION = 12.5
STARTS = (24, 108, 195, 288)  # frame of dip onset
LEAVE, LAND = 21, 36           # last grounded / first grounded, relative to dip onset
FLOOR = 607
HIDDEN = (STARTS[3] + LAND - 3, STARTS[3] + LAND + 6)


def _ease(u):
    return (1 - math.cos(math.pi * min(1, max(0, u)))) / 2


def pose_at(t):
    frame = round(t * FPS, 6)
    bend, lift, swing = 0.0, 0.0, 0.0
    for i, start in enumerate(STARTS):
        k = frame - start
        if not 0 <= k <= LAND + 24:
            continue
        landing_bend = (0.84, 0.81, 0.36, 0.78)[i]
        if k <= 12:
            bend = _ease(k / 12)
            swing = -0.48 * bend
        elif k <= LEAVE:
            u = _ease((k - 12) / (LEAVE - 12))
            bend, swing = 1 - u, -0.48 + 1.48 * u
        elif k < LAND:
            u = (k - LEAVE) / (LAND - LEAVE)
            arc = 4 * u * (1 - u)
            lift, bend, swing = 89 * arc, 0.15 * arc + 0.16 * u, 1 - 0.55 * u
        elif k <= LAND + 7:
            u = _ease((k - LAND) / 7)
            bend, swing = 0.16 + (landing_bend - 0.16) * u, 0.45 - 0.32 * u
        else:
            u = 1 - _ease((k - LAND - 7) / 17)
            bend, swing = landing_bend * u, 0.13 * u
        break

    # A near-foot lift and replant after jump 2, while the other foot stays put.
    step = min(1, max(0, (frame - STARTS[1] - LAND - 14) / 12))
    pts = np.empty((17, 2))

    def link(length, angle):
        a = math.radians(angle)
        return length * np.array([math.sin(a), -math.cos(a)])

    for side, dx in (("left", -13), ("right", 0)):
        near = side == "right"
        ankle = np.array([234 + dx + 38 * step * near,
                          FLOOR - 11 - lift - 26 * math.sin(math.pi * step) * near])
        knee = ankle + link(101, 4 + 30 * bend)
        hip = knee + link(112, -4 - 58 * bend)
        shoulder = hip + link(126, 6 + 22 * bend)
        elbow = shoulder + link(63, 173 - 98 * swing)
        wrist = elbow + link(60, 156 - 108 * swing)
        for j, p in zip(("ankle", "knee", "hip", "shoulder", "elbow", "wrist"),
                        (ankle, knee, hip, shoulder, elbow, wrist)):
            pts[J[f"{side}_{j}"]] = p
        pts[J[f"{side}_ear"]] = shoulder + [5, -32]
        pts[J[f"{side}_eye"]] = shoulder + [16, -37]
    pts[J["nose"]] = pts[J["right_eye"]] + [7, 7]
    return pts


def hidden(t):
    return (J["left_ankle"], J["right_ankle"]) if HIDDEN[0] <= t * FPS < HIDDEN[1] else ()


def draw_frame(t):
    img = draw_figure(pose_at(t), W, H, FLOOR, "SYNTHETIC STANDING JUMP")
    if hidden(t):
        cv2.rectangle(img, (169, FLOOR - 75), (320, FLOOR + 6), (68, 77, 82), -1)
    return img


def make_sample(dst: Path):
    return write_sample(dst, size=(W, H), fps=FPS, duration=DURATION, draw_frame=draw_frame)


def pose_payload(offset, n_frames, fps):
    return fixture_poses(offset, n_frames, fps, pose_at, (W, H), hidden)


class FixtureProvider:
    name = "fixture"
    simulated = True

    def pose(self, video, *, model, extra_body, ctx):
        return pose_payload(ctx["source_offset"], ctx["n_frames"], ctx["fps"]), {"cost": 0}

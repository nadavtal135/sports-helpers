"""SAM's "basketball" also matches a round (bald) head; the ball must not be tracked there.

From a real gym drill clip (406x720, 59.94 fps): from about 3.0 s a box scoring 0.700,
15-18 px, sat on a coach's head and moved slowly with him, (122, 280) at 3.0 s,
(127, 297) at 3.5 s, (154-161, 307) at 3.85-3.95 s. After SAM lost the real ball in
the net (f225), the track jumped to that head and shot 1, a clean make, came out
wrong. Rim (205, 234), 26 px wide, net bottom 256. No network.
"""

import csv

import cv2
import numpy as np

from basketball import settings
from basketball.src import ball, export, net_search, pose
from basketball.src.hoop import Hoop
from basketball.src.shots import Shot, judge

FPS = 59.94
N = 600
HOOP = Hoop(cx=205.0, cy=234.0, left=192.0, right=218.0, top=230.0, net_bottom=256.0,
            score=0.72, frame=0)


def head_at(f):
    t = f / FPS
    return float(np.interp(t, [3.0, 3.5, 3.9, 4.6], [122, 127, 157.5, 175])), \
        float(np.interp(t, [3.0, 3.5, 3.9, 4.6], [280, 297, 307, 309]))


def coach_heads(frames):
    return {f: [(*head_at(f), 20.0)] for f in frames}


def coach_boxes(frames):
    return {f: [(*head_at(f), 16.5, 16.5, 0.700)] for f in frames}


def select(raw, exclude=None):
    return ball.select(raw, N, [(130, 400)], size_tolerance=0.45, min_score=0.5, fps=FPS,
                       static_seconds=0.5, max_speed_mps=12.0, exclude=exclude)


def arc():
    """The real ball, every third frame, down to (198, 217.5) at f225."""
    pts = {}
    for f in range(165, 226, 3):
        s = (f - 225) / FPS
        pts[f] = [(198.0 + 60 * s, 217.5 + 400 * s + 530 * s * s, 15.0, 15.0, 0.782)]
    return pts


def test_slow_box_on_a_head_for_a_second_is_rejected():
    frames = range(int(3.0 * FPS), int(4.0 * FPS))
    raw = coach_boxes(frames)
    rejected = ball.head_boxes(raw, coach_heads(frames), FPS)
    assert len(rejected) == len(raw)                         # every one of its 60 frames
    t = select(raw, exclude=rejected)
    assert not np.isfinite(t.x).any() and t.on_head == rejected


def test_ball_passing_the_head_on_a_fast_ballistic_path_is_kept():
    heads = {f: [(150.0, 300.0, 20.0)] for f in range(100, 140)}
    raw = {}
    for f in range(100, 140):                                # 8 px/frame, through the head
        raw[f] = [(110.0 + 2 * (f - 100), 340.0 - 8 * (f - 100) + 0.1 * (f - 100) ** 2,
                   15.0, 15.0, 0.8)]
    assert ball.head_boxes(raw, heads, FPS) == set()


def test_ball_held_still_by_the_head_is_kept_when_it_looks_orange():
    heads = {f: [(150.0, 300.0, 20.0)] for f in range(100, 140)}
    raw = {f: [(152.0, 296.0, 17.0, 17.0, 0.8)] for f in range(100, 140)}   # in the pocket
    assert len(ball.head_boxes(raw, heads, FPS)) == 40                    # geometry alone: head
    orange = ball.head_boxes(raw, heads, FPS, colour=lambda f, cx, cy, size: 0.95)
    skin = ball.head_boxes(raw, heads, FPS, colour=lambda f, cx, cy, size: 0.5)
    assert orange == set() and len(skin) == 40


def test_ball_colour_tells_a_ball_from_a_head():
    img = np.full((80, 80, 3), (110, 110, 110), np.uint8)
    cv2.circle(img, (20, 40), 8, (30, 85, 205), -1)          # basketball orange (BGR)
    cv2.circle(img, (60, 40), 8, (95, 115, 150), -1)         # skin: low saturation
    assert ball.ball_colour(img, 20, 40, 16) >= 0.6
    assert ball.ball_colour(img, 60, 40, 16) < 0.6
    assert ball.ball_colour(img, 2, 2, 16) is None           # off the image


def test_box_far_from_the_head_or_not_head_sized_is_not_touched():
    heads = {f: [(150.0, 300.0, 20.0)] for f in range(10)}
    far = {f: [(200.0, 300.0, 16.0, 16.0, 0.7)] for f in range(10)}
    big = {f: [(150.0, 300.0, 60.0, 60.0, 0.7)] for f in range(10)}
    assert ball.head_boxes(far, heads, FPS) == set() and ball.head_boxes(big, heads, FPS) == set()


def test_heads_are_read_for_every_person_before_the_shooter_is_chosen():
    k = np.zeros((17, 2))
    k[0], k[1], k[2], k[3], k[4] = (0.30, 0.40), (0.31, 0.39), (0.29, 0.39), (0.32, 0.40), (0.28, 0.40)
    other = k.copy()
    other[:5] += 0.3
    frames = [{"index": 7, "persons": [
        {"kpts_xy": k.tolist(), "bbox_xywh": [0.25, 0.35, 0.1, 0.4]},
        {"kpts_xy": other.tolist(), "bbox_xywh": [0.55, 0.65, 0.1, 0.3]}]}]
    out = pose.heads(frames, 400, 700)
    assert len(out[7]) == 2
    cx, cy, size = out[7][0]
    assert abs(cx - 120) < 1 and abs(cy - 277) < 2 and size >= 0.13 * 0.4 * 700 - 1e-6


def test_coach_head_after_the_ball_is_lost_is_not_the_ball():
    frames = range(int(3.0 * FPS), int(4.6 * FPS))
    raw = {**coach_boxes(f for f in frames if f % 3 == 0)}
    for f, boxes in arc().items():
        raw.setdefault(f, []).extend(boxes)
    plain = select(raw)
    assert any(np.isfinite(plain.x[f]) and abs(plain.x[f] - head_at(f)[0]) < 3 for f in range(226, 280))
    rejected = ball.head_boxes(raw, coach_heads(frames), FPS)
    t = select(raw, exclude=rejected)
    assert not np.isfinite(t.x[226:]).any()                  # nothing after the ball was lost
    assert all(np.isfinite(t.x[f]) for f in arc())           # the real ball is untouched


# ── the ball followed out of the net, then the judge ──────────────────────────

def frames_with(points, n=90, first=226):
    x0, y0, x1, y1 = net_search.crop_box(HOOP, 406, 720)
    out = []
    for k in range(n):
        f = first + k
        img = np.full((720, 406, 3), (200, 200, 205), np.uint8)
        cv2.ellipse(img, (205, 234), (13, 3), 0, 0, 360, (40, 90, 220), 2)
        for xx in range(194, 218, 3):
            cv2.line(img, (xx, 236), (205 + (xx - 205) // 2, 256), (245, 245, 245), 1)
        if f in points:
            px, py = points[f]
            cv2.circle(img, (int(round(px)), int(round(py))), 7, (30, 85, 205), -1)
        out.append(img[y0:y1, x0:x1])
    return np.array(out), (x0, y0)


def through_and_out():
    """Into the rim, slowed in the net (y 235-256 over 25 frames), then falling below it."""
    pts = {226 + k: (199.0 + 1.5 * k, 220.0 + 4 * k) for k in range(4)}
    pts.update({230 + k: (205.0, 235.0 + 21 * k / 24) for k in range(25)})
    pts.update({255 + k: (204.0 - 0.2 * k, 258.0 + 2 * k + 0.2 * k * k) for k in range(30)})
    return pts


def test_search_keeps_following_the_ball_below_the_net():
    crops, offset = frames_with(through_and_out())
    path = net_search.follow(net_search.blobs(crops), 226, (225, 198.0, 217.5), HOOP, 13.0, FPS,
                             offset=offset)
    ys = [y for _, _, y in path]
    assert max(ys) > HOOP.net_bottom + 3 * HOOP.width          # well under the net
    assert max(ys) <= HOOP.net_bottom + 4 * HOOP.width + 1
    below = [(f, x) for f, x, y in path if y > HOOP.net_bottom]
    assert below and all(abs(x - HOOP.cx) <= HOOP.width for _, x in below)
    assert below[-1][0] - below[0][0] <= 0.6 * FPS + 1


def test_ball_followed_out_of_the_net_is_a_make():
    crops, offset = frames_with(through_and_out())
    path = net_search.follow(net_search.blobs(crops), 226, (225, 198.0, 217.5), HOOP, 13.0, FPS,
                             offset=offset)
    x, y, d = (np.full(N, np.nan) for _ in range(3))
    for f, boxes in arc().items():
        x[f], y[f], d[f] = boxes[0][0], boxes[0][1], 13.0
    for f, px, py in path:
        x[f], y[f] = px, py
    shot = Shot(number=1, start=150, end=160, release=164, set_frame=150, elbow_set=np.nan,
                elbow_release=np.nan)
    judge(shot, x, y, d, HOOP, FPS, until=500, inner_fraction=0.8, net_min_seconds=0.15,
          net_side_tolerance=0.35, rebound_margin=0.15, max_gap=int(0.5 * FPS),
          evidence_gap=int(0.3 * FPS), min_flight=int(0.8 * FPS), ball_to_rim=0.239 / 0.49,
          front_size=1.2)
    assert shot.made is True and shot.reason == "through the net"


def test_person_beside_the_net_is_not_followed_below_it():
    pts = {226 + k: (199.0 + 1.5 * k, 220.0 + 4 * k) for k in range(4)}
    pts.update({230 + k: (205.0, 235.0 + 21 * k / 24) for k in range(25)})
    pts.update({255 + k: (160.0, 300.0) for k in range(30)})   # something orange to the side
    crops, offset = frames_with(pts)
    path = net_search.follow(net_search.blobs(crops), 226, (225, 198.0, 217.5), HOOP, 13.0, FPS,
                             offset=offset)
    assert all(abs(x - HOOP.cx) <= HOOP.width for _, x, y in path if y > HOOP.net_bottom)


# ── export and settings ───────────────────────────────────────────────────────

def test_on_head_band_in_the_detections_export(tmp_path):
    frames = range(int(3.0 * FPS), int(3.5 * FPS))
    raw = coach_boxes(frames)
    t = select(raw, exclude=ball.head_boxes(raw, coach_heads(frames), FPS))
    all_boxes = {f: [(*b, 130) for b in boxes] for f, boxes in raw.items()}
    name = export.write_detections(all_boxes, t, FPS, 0, tmp_path, min_score=0.5, rescue_min=0.3)
    rows = list(csv.DictReader((tmp_path / name).open()))
    assert rows and all(r["score_band"] == "on head" and r["selected"] == "False" for r in rows)


def test_head_filter_settings():
    s = settings.Settings()
    assert s.ball_head_filter and s.ball_head_reach == 0.7 and s.ball_head_min_speed > 0

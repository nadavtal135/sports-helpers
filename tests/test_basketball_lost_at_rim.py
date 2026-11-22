"""A tracking loss at the rim is never a miss; a ball resting in the scene is never the
shot ball; the ball lost in the net is looked for again.

Numbers from a real gym drill clip (406x720 analysis frame, 59.94 fps): rim center
(205, 234), 26 px wide, net bottom 256; the ball (17 px) tracked down to (198, 217.5)
at frame 225, then SAM's only box per frame was a second ball resting on a chair at
about (154-161, 307), which the tracker jumped to, and the judge called the clean
make "miss: never came down through the rim". No network.
"""

import csv

import cv2
import numpy as np

from basketball import settings
from basketball.src import ball, export, net_search
from basketball.src.hoop import Hoop
from basketball.src.shots import Shot, judge

FPS = 59.94
HOOP = Hoop(cx=205.0, cy=234.0, left=192.0, right=218.0, top=230.0, net_bottom=256.0,
            score=0.72, frame=0)
D = 17.0
N = 600
RELEASE = 164           # 2.736 s


def arc():
    """The shot's arc, release to frame 225 at (198, 217.5): peaking about 75 px over the
    rim and coming down at ~400 px/s."""
    pts = {}
    for f in range(RELEASE, 226):
        s = (f - 225) / FPS
        pts[f] = (198.0 + 60 * s, 217.5 + 400 * s + 0.5 * 1060 * s * s)
    return pts


def chair(frames):
    return {f: (157.5 + 3.5 * ((f % 3) - 1) * 0.1, 307.0) for f in frames}


D_RIM = 13.0     # the ball's box near the rim (0.239 / 0.49 of a 26 px rim is 12.7 px)


def raw_boxes(points, score=0.78, size=D_RIM):
    return {f: [(x, y, size, size, score)] for f, (x, y) in points.items()}


def merge(*boxes):
    out = {}
    for b in boxes:
        for f, items in b.items():
            out.setdefault(f, []).extend(items)
    return out


def select(raw, **kw):
    args = dict(size_tolerance=0.45, min_score=0.5, fps=FPS, static_seconds=0.5, max_speed_mps=12.0)
    args.update(kw)
    return ball.select(raw, N, [(RELEASE - 30, 300)], **args)


def run_judge(x, y, d):
    shot = Shot(number=1, start=RELEASE - 10, end=RELEASE, release=RELEASE, set_frame=RELEASE - 10,
                elbow_set=np.nan, elbow_release=np.nan)
    judge(shot, x, y, d, HOOP, FPS, until=RELEASE + 300, inner_fraction=0.8, net_min_seconds=0.15,
          net_side_tolerance=0.35, rebound_margin=0.15, max_gap=int(0.5 * FPS),
          evidence_gap=int(0.3 * FPS), min_flight=int(0.8 * FPS), ball_to_rim=0.239 / 0.49,
          front_size=1.2)
    return shot


def gym_raw():
    """The SAM boxes: the real ball until 225 (score 0.782), then only the chair ball (0.700)."""
    return merge(raw_boxes(arc(), 0.782), raw_boxes(chair(range(226, 286)), 0.700, size=D_RIM))


# ── the bug and its fix ──────────────────────────────────────────────────────

def test_old_selection_jumps_to_the_chair_ball_and_the_judge_called_a_miss():
    t = select(gym_raw(), static_seconds=None, max_speed_mps=None)
    assert np.isfinite(t.x[231]) and abs(t.x[231] - 157.5) < 2       # the chair ball
    # Even so, the judge no longer reads it as the ball coming down: off the ball's path.
    shot = run_judge(t.x, t.y, t.d)
    assert shot.made is None and shot.reason.startswith("ball lost at the rim")
    assert any("off the ball's path" in e for e in shot.evidence)


def test_static_ball_is_excluded_and_the_shot_is_unknown_not_miss():
    t = select(gym_raw())
    assert not np.isfinite(t.x[226:]).any()                          # track ends at 225
    assert ball.key(231, *chair([231])[231]) in t.static
    shot = run_judge(t.x, t.y, t.d)
    assert shot.made is None
    assert shot.reason.startswith("ball lost at the rim")
    assert "a tracking loss is not evidence of a miss" in shot.evidence


def test_continuity_gate_rejects_the_jump_right_after_the_last_sighting():
    """Without the static filter the gate still refuses the first frames of the jump (100 px
    in a few frames, far faster than the ball). After a longer gap a track may start
    again anywhere, which is why the static filter is needed for the chair ball."""
    t = select(gym_raw(), static_seconds=None)
    assert not np.isfinite(t.x[226:231]).any()
    assert ball.key(228, *chair([228])[228]) in t.gated


def test_ball_held_still_then_shot_is_not_static():
    held = {f: (120.0, 400.0) for f in range(100, 170)}                # 1.2 s, perfectly still
    flight = {f: (120.0 + 3 * (f - 169), 400.0 - 6 * (f - 169)) for f in range(170, 200)}
    t = select(raw_boxes({**held, **flight}))
    assert not t.static
    assert np.isfinite(t.x[150]) and np.isfinite(t.x[190])


# ── the ball in the net ───────────────────────────────────────────────────────

def in_net():
    """Through the rim at x ~205, slowed in the net (y 235-262 over 20 frames), then out."""
    pts = {f: (204.0 + 0.1 * (f - 226), 222.0 + 4 * (f - 226)) for f in range(226, 230)}
    pts.update({230 + k: (205.0, 235.0 + 27 * k / 19) for k in range(20)})
    pts.update({250 + k: (204.0 - 0.3 * k, 264.0 + 3 * k + 0.15 * k * k) for k in range(15)})
    return pts


def test_ball_continuing_through_the_net_is_a_make():
    t = select(gym_raw())
    for f, (px, py) in in_net().items():
        t.x[f], t.y[f], t.d[f] = px, py, np.nan        # found in the net: no size
    shot = run_judge(t.x, t.y, t.d)
    assert shot.made is True and shot.reason == "through the net"


def synthetic_frames(points, n=90, first=226):
    """Hoop crops: a static orange rim ring and white net, plus an orange ball at *points*."""
    x0, y0, x1, y1 = net_search.crop_box(HOOP, 406, 720)
    frames = []
    for k in range(n):
        f = first + k
        img = np.full((720, 406, 3), (200, 200, 205), np.uint8)
        cv2.ellipse(img, (205, 234), (13, 3), 0, 0, 360, (40, 90, 220), 2)      # rim (BGR orange)
        for xx in range(194, 218, 3):
            cv2.line(img, (xx, 236), (205 + (xx - 205) // 2, 256), (245, 245, 245), 1)
        cv2.circle(img, (157, 307), 8, (30, 80, 200), -1)                     # the chair ball
        if f in points:
            px, py = points[f]
            cv2.circle(img, (int(round(px)), int(round(py))), int(D / 2), (30, 85, 205), -1)
        frames.append(img[y0:y1, x0:x1])
    return np.array(frames), (x0, y0)


def test_net_search_follows_the_ball_through_the_net_and_ignores_the_chair_ball():
    crops, offset = synthetic_frames(in_net())
    path = net_search.follow(net_search.blobs(crops), 226, (225, 198.0, 217.5), HOOP, D, FPS,
                             offset=offset)
    assert path and path[0][0] <= 230
    assert all(abs(x - 205) < 13 for f, x, y in path if HOOP.cy <= y <= HOOP.net_bottom)
    assert path[-1][2] > HOOP.net_bottom                                  # it left the net
    assert not any(abs(x - 157) < 10 for _, x, _ in path)
    t = select(gym_raw())
    for f, px, py in path:
        t.x[f], t.y[f], t.d[f] = px, py, np.nan
    assert run_judge(t.x, t.y, t.d).made is True


def test_net_search_finds_nothing_when_the_ball_is_not_in_view():
    crops, offset = synthetic_frames({})
    assert net_search.follow(net_search.blobs(crops), 226, (225, 198.0, 217.5), HOOP, D, FPS,
                             offset=offset) == []


def test_net_search_rejects_a_ball_coming_down_outside_the_rim():
    beside = {f: (240.0, 220.0 + 3 * (f - 226)) for f in range(226, 260)}
    crops, offset = synthetic_frames(beside)
    assert net_search.follow(net_search.blobs(crops), 226, (225, 238.0, 217.0), HOOP, D, FPS,
                             offset=offset) == []


# ── a real miss is still a miss ───────────────────────────────────────────────

def test_rebound_above_the_rim_falling_outside_is_still_a_miss():
    pts = dict(arc())
    for k in range(1, 40):                         # off the back of the rim, up and away
        s = k / FPS
        pts[225 + k] = (198.0 + 140 * s, 228.0 - 160 * s + 0.5 * 1060 * s * s)
    t = select(raw_boxes(pts))
    shot = run_judge(t.x, t.y, t.d)
    assert shot.made is False


# ── exports ───────────────────────────────────────────────────────────────────

def test_detections_export_marks_static_gated_and_net_search(tmp_path):
    raw = gym_raw()
    t = select(raw)
    t.recovered = np.zeros(N, bool)
    t.x[240], t.y[240], t.d[240], t.recovered[240] = 205.0, 244.0, np.nan, True
    all_boxes = {f: [(*b, RELEASE - 30) for b in boxes] for f, boxes in raw.items()}
    name = export.write_detections(all_boxes, t, FPS, 0, tmp_path, min_score=0.5, rescue_min=0.3)
    rows = list(csv.DictReader((tmp_path / name).open()))
    bands = {r["score_band"] for r in rows}
    assert {"main", "static", "net_search"} <= bands
    assert all(r["score_band"] == "static" for r in rows
               if int(r["frame"]) >= 231 and r["score_band"] != "net_search")
    net = next(r for r in rows if r["score_band"] == "net_search")
    assert net["frame"] == "240" and net["selected"] == "True" and net["score"] == ""


def test_new_settings_exist_and_are_on():
    s = settings.Settings()
    assert s.ball_static_seconds == 0.5 and s.ball_max_speed_mps > 0 and s.ball_net_search


def test_stride_three_jump_as_on_the_real_clip_is_not_a_miss():
    """SAM looked at every third frame there: the ball at 225, the chair ball at 231, 234, 237."""
    raw = merge(raw_boxes({f: p for f, p in arc().items() if (225 - f) % 3 == 0}, 0.782),
                raw_boxes({231: (154.0, 308.0), 234: (158.0, 307.5), 237: (161.5, 306.5),
                           **{f: (158.0, 307.0) for f in range(240, 300, 3)}}, 0.700))
    for kw in ({}, {"static_seconds": None, "max_speed_mps": None}):
        t = select(raw, **kw)
        shot = run_judge(t.x, t.y, t.d)
        assert shot.made is None and shot.reason.startswith("ball lost at the rim"), kw


# ── the net search is not limited to the frames SAM tracked ──────────────────

def judge_until(x, y, d, until):
    shot = Shot(number=1, start=RELEASE - 10, end=RELEASE, release=RELEASE, set_frame=RELEASE - 10,
                elbow_set=np.nan, elbow_release=np.nan)
    judge(shot, x, y, d, HOOP, FPS, until=until, inner_fraction=0.8, net_min_seconds=0.15,
          net_side_tolerance=0.35, rebound_margin=0.15, max_gap=int(0.5 * FPS),
          evidence_gap=int(0.3 * FPS), min_flight=int(0.8 * FPS), ball_to_rim=0.239 / 0.49,
          front_size=1.2)
    return shot


def test_net_search_may_fill_frames_past_the_tracking_window_and_the_shot_becomes_a_make():
    """Real clip: the shot's ball window is capped at 128 frames (2.1 s at 60 fps) and ended at
    frame 239, just after the ball entered the net (it falls out of it at about frame 254).
    The search reads the video itself, so it must be allowed past the window; before, it was cut
    off there and the shot stayed 'crossed inside the rim, but the ball was not seen leaving'."""
    from basketball import pipeline

    window_end = 232
    crops, _ = synthetic_frames(in_net())

    def read(first, count, box):
        return crops[first - 226:first - 226 + count]

    s = settings.Settings()

    # Before: frames are only added inside the window.
    t_old = select(gym_raw())
    shot_old = judge_until(t_old.x, t_old.y, t_old.d, window_end)
    assert shot_old.made is None and shot_old.reason.startswith("ball lost at the rim")
    old = pipeline._search_net(shot_old, t_old, HOOP, FPS, N, 406, 720, s, read, until=window_end)
    assert old and max(old) < window_end
    assert judge_until(t_old.x, t_old.y, t_old.d, window_end).made is None

    # Now: the search may add frames up to the next shot, and the judge reads them.
    t = select(gym_raw())
    shot = judge_until(t.x, t.y, t.d, window_end)
    found = pipeline._search_net(shot, t, HOOP, FPS, N, 406, 720, s, read, until=window_end,
                                 fill_until=RELEASE + 300)
    assert max(found) > window_end + 15                      # it followed the ball out of the net
    assert np.isfinite(t.x[found[-1]]) and t.y[found[-1]] > HOOP.net_bottom
    again = judge_until(t.x, t.y, t.d, min(RELEASE + 300, found[-1] + 1))
    assert again.made is True and again.reason == "through the net"

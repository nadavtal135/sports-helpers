"""judge(), tally() and streaks() on synthetic ball tracks against a known hoop."""

import numpy as np
import pytest

from basketball.src.hoop import Hoop
from basketball.src.shots import Shot, judge, streaks, tally

FPS = 30.0
HOOP = Hoop(cx=900.0, cy=260.0, left=865.0, right=935.0, top=254.0, net_bottom=332.0,
            score=0.9, frame=0)
N = 200
D = 34.0   # ball diameter at the rim's distance


def track(points):
    """{frame: (x, y)} -> x, y, size arrays (NaN elsewhere)."""
    x, y, d = (np.full(N, np.nan) for _ in range(3))
    for f, (px, py) in points.items():
        x[f], y[f], d[f] = px, py, D
    return x, y, d


def arc(f0, f1, x0, y0, x1, y1, apex=120.0):
    """Frames f0..f1 on a parabola from (x0, y0) to (x1, y1), peaking *apex* px above y0."""
    out = {}
    for f in range(f0, f1 + 1):
        t = (f - f0) / (f1 - f0)
        out[f] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t - 4 * apex * t * (1 - t))
    return out


def through_net(f0, frames=9, x=900.0, y0=264.0, y1=345.0, hide=()):
    """Down through the net from the rim over *frames* frames (~0.3 s: slowed)."""
    return {f0 + k: (x, y0 + (y1 - y0) * k / frames) for k in range(1, frames + 1)
            if f0 + k not in hide}


def run(points, *, until=N, **kw):
    x, y, d = track(points)
    shot = Shot(number=1, start=0, end=5, release=10, set_frame=5, elbow_set=np.nan,
                elbow_release=np.nan)
    args = dict(inner_fraction=0.8, net_min_seconds=0.15, net_side_tolerance=0.35,
                rebound_margin=0.15, max_gap=15, evidence_gap=9, min_flight=24,
                ball_to_rim=0.239 / 0.49, front_size=1.2)
    args.update(kw)
    judge(shot, x, y, d, HOOP, FPS, until=until, **args)
    return shot


def approach(end_x=900.0):
    # From the hands (frame 10) to just under the rim line (frame 40).
    return arc(10, 40, 450.0, 330.0, end_x, 264.0, apex=260.0)


def test_clean_make():
    pts = {**approach(), **through_net(40)}
    shot = run(pts)
    assert shot.made is True and shot.reason == "through the net"
    assert any("slowed" in e for e in shot.evidence)


def test_make_with_short_net_occlusion_is_still_a_make():
    pts = {**approach(), **through_net(40, hide={43, 44, 45})}
    assert run(pts).made is True


def test_lost_after_crossing_is_unknown_not_make():
    """Regression: the reference called this a make ("into the net")."""
    pts = approach()
    shot = run(pts)
    assert shot.made is None
    assert "not seen leaving the net" in shot.reason


def test_long_gap_inside_net_is_unknown():
    # Seen at the rim, then nothing for 0.6 s, then far below: exit not observed.
    pts = {**approach(), 41: (900.0, 270.0), 60: (905.0, 560.0)}
    shot = run(pts)
    assert shot.made is None


def test_next_sighting_back_in_hands_is_not_a_net_exit():
    # Within the evidence gap but nowhere near the net (e.g. the next shot's hands).
    pts = {**approach(), 41: (900.0, 270.0), 48: (420.0, 500.0), 49: (421.0, 501.0)}
    assert run(pts).made is None


def test_hidden_crossing_at_rim_is_unknown():
    # Last seen above the rim, next seen in the net after more than max_gap frames.
    pts = {**arc(10, 34, 450.0, 330.0, 895.0, 215.0, apex=200.0), 52: (900.0, 300.0)}
    shot = run(pts)
    assert shot.made is None and "hidden" in shot.reason


def test_miss_outside_the_rim():
    pts = {**approach(end_x=970.0), **{40 + k: (970.0 + k, 264.0 + 30 * k) for k in range(1, 12)}}
    shot = run(pts)
    assert shot.made is False and shot.reason == "outside the rim"


def test_rim_out_is_a_miss():
    up = {40 + k: (918.0 + 4 * k, 264.0 - 9 * k + 0.6 * k * k) for k in range(1, 30)}
    pts = {**approach(end_x=918.0), **up}
    shot = run(pts)
    assert shot.made is False
    assert shot.reason in ("off the rim", "rimmed out")


def test_fell_past_the_net_too_fast():
    fast = {40 + k: (900.0, 264.0 + 45 * k) for k in range(1, 6)}
    shot = run({**approach(), **fast})
    assert shot.made is False and shot.reason == "fell past the net"


def test_in_front_of_the_rim_by_size():
    x, y, d = track({**approach(), **through_net(40)})
    d[:] = np.where(np.isfinite(d), 1.6 * D, np.nan)   # much bigger: nearer the camera
    shot = Shot(number=1, start=0, end=5, release=10, set_frame=5, elbow_set=np.nan,
                elbow_release=np.nan)
    judge(shot, x, y, d, HOOP, FPS, until=N, inner_fraction=0.8, net_min_seconds=0.15,
          net_side_tolerance=0.35, rebound_margin=0.15, max_gap=15, evidence_gap=9,
          min_flight=24, ball_to_rim=0.239 / 0.49, front_size=1.2)
    assert shot.made is False and shot.reason == "in front of the rim"


def test_no_ball_observations_is_unknown():
    shot = run({})
    assert shot.made is None and shot.reason == "ball not tracked after release"


def test_short_track_below_rim_is_unknown_long_track_is_miss():
    short = {f: (450.0 + 5 * f, 330.0 - 3 * f) for f in range(10, 18)}
    assert run(short).made is None
    airball = {f: (450.0 + 5 * f, 330.0 - 2 * (f - 10) + 0.12 * (f - 10) ** 2) for f in range(10, 50)}
    shot = run(airball)
    assert shot.made is False and shot.reason == "never above the rim"


def test_lost_while_above_rim_is_unknown():
    pts = arc(10, 30, 450.0, 330.0, 880.0, 160.0, apex=200.0)
    shot = run(pts)
    assert shot.made is None and "above the rim" in shot.reason


def test_until_bounds_the_evidence():
    # The net exit happens after `until` (tracking window ended): not a make.
    pts = {**approach(), **through_net(40)}
    assert run(pts, until=42).made is None


def _shot(n, made, decided):
    s = Shot(number=n, start=decided - 20, end=decided - 10, release=decided - 15, set_frame=0,
             elbow_set=np.nan, elbow_release=np.nan)
    s.made, s.decided = made, decided
    return s


def test_tally_excludes_unknown_from_fg():
    shots = [_shot(1, True, 10), _shot(2, False, 20), _shot(3, None, 30), _shot(4, True, 40)]
    t = tally(shots)
    assert (t["attempts"], t["makes"], t["misses"], t["unknown"], t["judged"]) == (4, 2, 1, 1, 3)
    assert t["fg_pct"] == pytest.approx(2 / 3, abs=1e-4)
    assert t["makes"] + t["misses"] + t["unknown"] == t["attempts"]
    assert tally([_shot(1, None, 5)])["fg_pct"] is None
    assert tally([])["fg_pct"] is None


def test_streak_unknown_neither_extends_nor_resets():
    shots = [_shot(1, True, 10), _shot(2, True, 20), _shot(3, None, 30), _shot(4, True, 40),
             _shot(5, False, 50)]
    s = streaks(shots, 60)
    assert s[25] == 2 and s[35] == 2 and s[45] == 3 and s[55] == 0


def test_spread_points_separates_overlaps_and_keeps_far_points():
    from basketball.src.render import spread_points
    out = spread_points([(100, 100), (100, 100), (105, 102), (400, 400)], min_dist=32)
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.hypot(out[i][0] - out[j][0], out[i][1] - out[j][1]) >= 30
    assert out[3] == (400, 400)


def test_pick_shooter_prefers_hands_overhead_over_largest_person():
    from basketball.src import pose

    def person(tid, wrists_up, size):
        k = np.full((17, 2), 0.5)
        k[:5, 1] = 0.30                      # head
        k[[9, 10], 1] = 0.20 if wrists_up else 0.60
        return {"track_id": tid, "kpts_xy": k.tolist(), "bbox_xywh": [0.1, 0.1, size, size]}

    frames = [{"index": i, "persons": [person(2, False, 0.5), person(1, 10 <= i < 16, 0.2)]}
              for i in range(30)]
    assert pose.pick_shooter(frames) == 1
    pose.keep_track(frames, 1)
    assert all(p["track_id"] == 1 for f in frames for p in f["persons"])
    nobody = [{"index": i, "persons": [person(2, False, 0.5)]} for i in range(30)]
    assert pose.pick_shooter(nobody) is None

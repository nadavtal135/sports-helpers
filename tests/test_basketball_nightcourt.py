"""judge() net timing and the shot-attempt check, on tracks copied from a real
night-court clip (768x432, 30 fps, still camera, one player).

Hoop there: rim center (370, 100) px, 40 px wide, net bottom at 135 px.
Shot 4 (30.6 s): a fast clean swish, called "miss: fell past the net" by the old
fixed 0.15 s rim-to-net floor. Shot 3 (20.2 s): hands up while gathering a
bouncing ball that moved away from the hoop, called "unknown" instead of "not a shot".
"""

import numpy as np
import pytest

from basketball import settings
from basketball.src.hoop import Hoop
from basketball.src.shots import Shot, judge, not_a_shot

FPS = 30.0
HOOP = Hoop(cx=370.0, cy=100.0, left=350.0, right=390.0, top=94.0, net_bottom=135.0,
            score=0.9, frame=0)
N = 1500
D = 19.5     # ball px at the rim's distance (0.239 / 0.49 of 40 px)


def F(t):
    return int(round(t * FPS))


def track(points):
    x, y, d = (np.full(N, np.nan) for _ in range(3))
    for f, (px, py) in points.items():
        x[f], y[f], d[f] = px, py, D
    return x, y, d


def run_judge(points, *, start, release, method="ball", **kw):
    x, y, d = track(points)
    shot = Shot(number=1, start=start, end=start + 10, release=release, set_frame=start,
                elbow_set=np.nan, elbow_release=np.nan, release_method=method)
    args = dict(inner_fraction=0.8, net_min_seconds=0.15, net_side_tolerance=0.35,
                rebound_margin=0.15, max_gap=15, evidence_gap=9, min_flight=24,
                ball_to_rim=0.239 / 0.49, front_size=1.2)
    args.update(kw)
    judge(shot, x, y, d, HOOP, FPS, until=N, **args)
    return shot


# The arc up to the rim (assumed; the report gave the points from 31.5 s on).
APPROACH = {F(30.67): (250, 120), F(30.83): (280, 60), F(31.0): (305, 30),
            F(31.17): (325, 22), F(31.33): (342, 40)}
# Reported: 31.5 s (357, 80) -> 31.6 s (371, 113) -> 31.7 s not seen -> 31.8 s (389, 170).
SWISH = {**APPROACH, F(31.5): (357, 80), F(31.6): (371, 113), F(31.8): (389, 170)}


# ── net timing ───────────────────────────────────────────────────────────────

def test_fast_swish_crossing_in_three_frames_is_a_make():
    shot = run_judge(SWISH, start=F(30.5), release=F(30.6))
    assert shot.made is True and shot.reason == "through the net"
    assert any("not clearly slowed" in e or "slowed by the net" in e for e in shot.evidence)
    # The old fixed floor would have rejected it: rim to net bottom took < 0.15 s.
    assert shot.decided - shot.crossing < 0.15 * FPS + 3


def test_fast_swish_is_a_make_at_any_reasonable_tolerance():
    for tol in (0.1, 0.25, 0.5):
        assert run_judge(SWISH, start=F(30.5), release=F(30.6),
                         fall_speed_tolerance=tol).made is True


def test_ball_faster_than_free_fall_through_the_net_is_still_a_miss():
    # Same entry, then 60 px/frame "through the net": impossible inside it, so the
    # ball is nearer the camera, falling past the rim.
    pts = {**APPROACH, F(31.5): (357, 80), F(31.6): (371, 113), F(31.6) + 1: (372, 173)}
    shot = run_judge(pts, start=F(30.5), release=F(30.6))
    assert shot.made is False and shot.reason == "fell past the net"


def test_ball_passing_beside_the_rim_is_still_a_miss():
    beside = {**APPROACH, F(31.5): (380, 80), F(31.6): (392, 113), F(31.8): (398, 170)}
    shot = run_judge(beside, start=F(30.5), release=F(30.6))
    assert shot.made is False and shot.reason in ("outside the rim", "off the rim")


def test_exit_wide_of_the_net_is_still_a_miss():
    # Exits the net bottom ~37 px from center (rim half-width 20, net tolerance 34).
    wide = {**APPROACH, F(31.5): (357, 80), F(31.6): (371, 113), F(31.8): (409, 136)}
    shot = run_judge(wide, start=F(30.5), release=F(30.6))
    assert shot.made is False and shot.reason in ("wide of the net", "off the rim")


def test_fast_exit_at_the_net_edge_is_unknown_not_miss():
    # Exits ~28 px from center (outside the rim's width, inside the net) in ~3.6 frames:
    # no evidence of net drag, so it could be either.
    edge = {**APPROACH, F(31.5): (357, 80), F(31.6): (371, 113), F(31.7): (404, 140)}
    shot = run_judge(edge, start=F(30.5), release=F(30.6))
    assert shot.made is None and "edge" in shot.reason


def test_slow_exit_at_the_net_edge_is_a_make():
    edge = {**APPROACH, F(31.5): (357, 80), F(31.6): (371, 104), F(31.7): (378, 118),
            F(31.8): (384, 128), F(31.9): (389, 140)}
    shot = run_judge(edge, start=F(30.5), release=F(30.6))
    assert shot.made is True and any("slowed by the net" in e for e in shot.evidence)


def test_timing_scales_with_frame_rate():
    """The same physical swish filmed at 60 fps: twice the frames, same verdict."""
    pts60 = {}
    for f, (px, py) in SWISH.items():
        pts60[f * 2] = (px, py)
    pts60[F(31.7) * 2] = (380, 141)        # at 60 fps the net hides fewer frames
    x, y, d = (np.full(2 * N, np.nan) for _ in range(3))
    for f, (px, py) in pts60.items():
        x[f], y[f], d[f] = px, py, D
    shot = Shot(number=1, start=F(30.5) * 2, end=F(30.5) * 2 + 10, release=F(30.6) * 2,
                set_frame=0, elbow_set=np.nan, elbow_release=np.nan)
    judge(shot, x, y, d, HOOP, 60.0, until=2 * N, inner_fraction=0.8, net_min_seconds=0.15,
          net_side_tolerance=0.35, rebound_margin=0.15, max_gap=30, evidence_gap=18,
          min_flight=48, ball_to_rim=0.239 / 0.49, front_size=1.2)
    assert shot.made is True


# ── not a shot ───────────────────────────────────────────────────────────────

# Reported: 19.4-19.8 s the ball falls and bounces (y 132 -> 324), then 20.0-20.4 s it
# rises and recedes from the hoop (x 199 -> 166, y 189 -> 146), lost from 20.5 s.
# Hands up (pose-only release) at 20.23 s.
def _gather():
    pts = {}
    for k, t in enumerate(np.arange(19.4, 19.81, 1 / FPS)):
        pts[F(t)] = (215 - 2 * k, 132 + (324 - 132) * k / 12)
    for k, t in enumerate(np.arange(20.0, 20.41, 1 / FPS)):
        pts[F(t)] = (199 - 33 * k / 12, 189 - 43 * k / 12)
    return pts


def _pose_shot(t):
    return Shot(number=3, start=F(t), end=F(t) + 6, release=F(t), set_frame=F(t),
                elbow_set=np.nan, elbow_release=np.nan, release_method="pose")


def _check(shot, pts, **kw):
    x, y, _ = track(pts)
    s = settings.Settings()
    args = dict(window_s=s.attempt_window_seconds, min_obs=s.attempt_min_observations,
                recede_rims=s.attempt_recede_rims)
    args.update(kw)
    return not_a_shot(shot, x, y, HOOP, FPS, **args)


def test_gathering_a_bouncing_receding_ball_is_not_a_shot():
    result = _check(_pose_shot(20.23), _gather())
    assert result is not None
    reason, evidence = result
    assert reason.startswith("not a shot")
    assert any("away from the hoop" in e for e in evidence)
    assert any("bounced" in e for e in evidence)


def test_real_shot_with_ball_lost_early_stays_unknown():
    # Released towards the hoop, seen for 3 frames, then lost.
    pts = {F(12.0) + k: (200 + 6 * k, 200 - 8 * k) for k in range(3)}
    shot = _pose_shot(12.0)
    assert _check(shot, pts) is None
    judged = run_judge(pts, start=shot.start, release=shot.release, method="pose")
    assert judged.made is None


def test_ball_rising_towards_the_hoop_is_kept_even_if_lost_below_the_rim():
    pts = {F(12.0) + k: (200 + 6 * k, 260 - 6 * k) for k in range(15)}   # long track, approaching
    assert _check(_pose_shot(12.0), pts) is None


def test_ball_release_shots_are_never_dropped():
    shot = _pose_shot(20.23)
    shot.release_method = "ball"
    assert _check(shot, _gather()) is None


def test_ball_that_goes_above_the_rim_is_kept():
    pts = {F(20.0) + k: (199 - 3 * k, 189 - 9 * k) for k in range(14)}   # receding but climbs above rim
    assert _check(_pose_shot(20.23), pts) is None


def test_attempt_check_is_a_named_setting():
    s = settings.Settings()
    assert s.shot_attempt_check is True
    assert s.attempt_recede_rims > 0 and s.attempt_min_observations >= 3
    with pytest.raises(ValueError):
        settings.from_user({"shot_attempt_check": False})   # internal, not a browser setting

"""Back-to-back shots in long continuous clips: the shot-attempt check is opt-in and
never removes a shot silently, a quick second shot keeps its own release window and
verdict, impossible jumps are not rim crossings, and motion-blurred balls in flight
are recovered. All synthetic, no network."""

import dataclasses

import numpy as np
import pytest

from basketball import pipeline, sample, settings
from basketball.src import ball, synthetic
from basketball.src.hoop import Hoop
from basketball.src.shots import (Shot, attempt_check, clearly_not_a_shot,
                                  flag_possible_non_shot, judge)

FPS = 30.0
NIGHT_HOOP = Hoop(cx=370.0, cy=100.0, left=350.0, right=390.0, top=94.0, net_bottom=135.0,
                  score=0.9, frame=0)
# The 20.2 s shot of the night-court clip: a REAL shot taken right after another.
SHOT3 = {19.4: (271, 132), 19.5: (260, 181), 19.6: (249, 249), 19.7: (235, 321), 19.8: (227, 324),
         19.9: (215, 269), 20.1: (199, 189), 20.2: (191, 165), 20.3: (181, 150), 20.4: (166, 146),
         21.0: (229, 248), 21.1: (249, 238), 21.4: (296, 218)}


def _xy(points, n=1500):
    x, y = np.full(n, np.nan), np.full(n, np.nan)
    for t, (px, py) in points.items():
        f = int(round(t * FPS))
        x[f], y[f] = px, py
    return x, y


def _pose_shot(t):
    f = int(round(t * FPS))
    return Shot(number=1, start=f, end=f + 6, release=f, set_frame=f, elbow_set=np.nan,
                elbow_release=np.nan, release_method="pose")


def _attempt_args():
    s = settings.Settings()
    return dict(window_s=s.attempt_window_seconds, min_obs=s.attempt_min_observations,
                recede_rims=s.attempt_recede_rims, early_s=s.attempt_early_seconds,
                gap_s=s.evidence_gap_seconds)


# ── 1. the shot-attempt check: opt-in, and it flags rather than removes ──────

def test_attempt_check_is_off_by_default_and_a_user_setting():
    assert settings.Settings().shot_attempt_check is False
    assert "shot_attempt_check" in settings.USER_SETTINGS
    assert settings.from_user({"shot_attempt_check": "true"}).shot_attempt_check is True


def test_real_back_to_back_shot_is_kept_as_unknown_not_removed():
    """The 20.2 s shot: its ball moved away, was lost, then came back towards the hoop."""
    x, y = _xy(SHOT3)
    shot = _pose_shot(20.233)
    action, found = attempt_check(shot, x, y, NIGHT_HOOP, FPS, **_attempt_args())
    assert action == "flag"                                   # never "drop"
    assert not clearly_not_a_shot(shot, x, y, NIGHT_HOOP, FPS, **_attempt_args())
    shot.made, shot.reason = None, "ball lost before it could reach the rim"
    flag_possible_non_shot(shot, found)
    assert shot.made is None and shot.reason.startswith("possible non-shot")
    assert any(e.startswith("judge alone") for e in shot.evidence)


def test_flagged_shot_that_went_through_the_net_stays_a_make():
    shot = _pose_shot(20.233)
    shot.made, shot.reason, shot.evidence = True, "through the net", ["left the net"]
    flag_possible_non_shot(shot, ("not a shot: ...", ["moved away"]))
    assert shot.made is True and shot.reason == "through the net"


def test_only_an_unambiguous_gather_is_removed():
    # Moves steadily and far away below the rim, never comes back: the stricter test holds.
    pts = {19.95 + k / FPS: (200 - 4 * k, 200 - k) for k in range(30)}
    x, y = _xy(pts)
    action, _ = attempt_check(_pose_shot(20.233), x, y, NIGHT_HOOP, FPS, **_attempt_args())
    assert action == "drop"


def test_ball_release_shots_are_always_kept():
    x, y = _xy(SHOT3)
    shot = _pose_shot(20.233)
    shot.release_method = "ball"
    assert attempt_check(shot, x, y, NIGHT_HOOP, FPS, **_attempt_args()) == ("keep", None)


# ── 2. back-to-back shots, the second off the first one's rebound ─────────────

@pytest.fixture(scope="module")
def pair_clip(tmp_path_factory):
    """A rim-out, then 2.2 s later (hands up to hands up) a make with the rebound."""
    d = tmp_path_factory.mktemp("pair")
    old = (synthetic.SHOTS, synthetic.DURATION)
    synthetic.SHOTS = [(1.6, 420.0, 240.0, "rimout"), (3.8, 430.0, 238.0, "make")]
    synthetic.DURATION = 7.0
    try:
        clip = sample.make_sample(d / "pair.mp4")
    finally:
        synthetic.SHOTS, synthetic.DURATION = old
    return clip, d


def _run_pair(pair_clip, name, s):
    from basketball.src.cache import Cache
    from basketball.src.providers.fixture import FixtureProvider

    clip, d = pair_clip
    old = (synthetic.SHOTS, synthetic.DURATION)
    synthetic.SHOTS = [(1.6, 420.0, 240.0, "rimout"), (3.8, 430.0, 238.0, "make")]
    synthetic.DURATION = 7.0
    try:
        return pipeline.run(clip, d / name, settings=s, gateway=settings.Gateway(),
                            provider=FixtureProvider(), cache=Cache(d / "cache"))
    finally:
        synthetic.SHOTS, synthetic.DURATION = old


def test_back_to_back_pair_both_judged(pair_clip):
    r = _run_pair(pair_clip, "new", settings.Settings())
    assert [(s["result"], s["reason"]) for s in r["shots"]] == [("miss", "off the rim"),
                                                                ("make", "through the net")]
    # The second shot's ball window starts before the first's ends: its own lead-in.
    (s1, l1), (s2, _) = r["meta"]["ball"]["windows"][:2]
    assert s2 < s1 + l1
    assert (s1 + l1) - s2 <= int(settings.Settings().ball_max_overlap_seconds * FPS)


def test_old_gap_rule_swallowed_the_second_shot(pair_clip):
    """Control: the reference's 1.5 s gap merged the pair into one shot."""
    old = dataclasses.replace(settings.Settings(), shot_min_gap_seconds=1.5)
    r = _run_pair(pair_clip, "old", old)
    assert len(r["shots"]) == 1


# ── 3. impossible jumps are not rim crossings ────────────────────────────────

def test_jump_between_rebound_and_next_shooters_hands_is_not_a_crossing():
    hoop = Hoop(cx=900.5, cy=260.0, left=865.0, right=936.0, top=254.0, net_bottom=333.0,
                score=0.9, frame=0)
    pts = {}
    for k in range(30):                                   # up and over, down to the rim edge
        t = k / 30
        pts[40 + k] = (450 + 15 * k, 330 - 900 * t + 1300 * t * t)
    pts.update({70: (912, 255), 71: (924, 261), 72: (929, 253), 73: (935, 246), 74: (940, 240),
                75: (945, 236), 76: (950, 232),           # bounced off the back of the rim
                81: (466, 489), 82: (466, 489)})          # 5 frames later: the next shooter's hands
    n = 200
    x, y, d = (np.full(n, np.nan) for _ in range(3))
    for f, (px, py) in pts.items():
        x[f], y[f], d[f] = px, py, 34.0
    shot = Shot(number=1, start=30, end=35, release=40, set_frame=30, elbow_set=np.nan,
                elbow_release=np.nan)
    judge(shot, x, y, d, hoop, FPS, until=n, inner_fraction=0.8, net_min_seconds=0.15,
          net_side_tolerance=0.35, rebound_margin=0.15, max_gap=15, evidence_gap=9,
          min_flight=24, ball_to_rim=0.239 / 0.49, front_size=1.2)
    assert shot.made is False and shot.reason in ("off the rim", "rimmed out")


# ── 4. motion blur in flight ─────────────────────────────────────────────────

def _flight(blur_frames, *, box, score=0.4, offset=(0.0, 0.0)):
    """A 22 px ball rising fast to the right: round boxes, except *blur_frames*,
    whose box is *box* (w, h) at *offset* from the true position."""
    raw = {}
    for f in range(10, 40):
        cx, cy = 200 + 10 * (f - 10), 300 - 12 * (f - 10) + 0.45 * (f - 10) ** 2
        if f in blur_frames:
            raw[f] = [(cx + offset[0], cy + offset[1], box[0], box[1], score)]
        else:
            raw[f] = [(cx, cy, 22.0, 22.0, 0.9)]
    return raw


def _pick(raw, rescue=True):
    return ball.select(raw, 60, [(0, 60)], size_tolerance=0.35, min_score=0.5,
                       blur_rescue=rescue, max_stretch=2.5)


def test_blurred_ball_in_flight_is_recovered():
    # Near the apex the ball moves mostly sideways (10 px/frame), so its blur is wide.
    gap = set(range(20, 28))                                       # lost mid-flight
    t = _pick(_flight(gap, box=(38.0, 24.0)))                      # stretched along the motion
    assert all(np.isfinite(t.x[f]) for f in gap)
    assert t.blurred[list(gap)].all() and not t.blurred[:20].any()
    assert np.nanmax(np.abs(t.d[list(gap)] - 24.0)) < 1e-6         # the short side is the ball


def test_blur_rescue_is_off_when_disabled():
    gap = set(range(20, 28))
    t = _pick(_flight(gap, box=(38.0, 24.0)), rescue=False)
    assert not any(np.isfinite(t.x[f]) for f in gap)


def test_box_off_the_predicted_path_is_not_rescued():
    gap = set(range(20, 28))
    t = _pick(_flight(gap, box=(38.0, 24.0), offset=(-80.0, 60.0)))   # e.g. a dark sleeve
    assert not any(np.isfinite(t.x[f]) for f in gap)


def test_box_stretched_across_the_motion_is_not_rescued():
    gap = set(range(20, 22))
    # Here the ball moves mostly sideways: a TALL box (an arm, a sleeve) is not its blur.
    t = _pick(_flight(gap, box=(24.0, 44.0)))
    assert not any(np.isfinite(t.x[f]) for f in gap)


def test_slow_ball_is_not_rescued():
    raw = {f: [(300.0, 300.0, 22.0, 22.0, 0.9)] for f in range(10, 20)}   # held still
    raw.update({f: [(300.0, 300.0, 22.0, 40.0, 0.4)] for f in range(20, 24)})
    t = _pick(raw)
    assert not any(np.isfinite(t.x[f]) for f in range(20, 24))


def test_main_pass_unchanged_without_low_score_boxes():
    """Identical to the old selection when every box is round and confident."""
    raw = _flight(set(), box=(22.0, 22.0))
    new = _pick(raw)
    old = ball.select(raw, 60, [(0, 60)], size_tolerance=0.35)
    assert np.allclose(np.nan_to_num(new.x), np.nan_to_num(old.x))
    assert not new.blurred.any()

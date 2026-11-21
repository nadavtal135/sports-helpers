"""Squat and standing jump on the imperfect pose tracks real footage gives:
keypoint jitter, pose dropouts and a hidden far ankle. The synthetic samples
supply the ground truth; perturbations are seeded, so results are repeatable."""

import numpy as np
import pytest

from running.src.pose import unwrap
from running.src.side_view_sample import J, fixture_poses
from squat import analysis as squat_analysis, sample as squat_sample, settings as squat_settings
from standing_jump import analysis as jump_analysis, sample as jump_sample, settings as jump_settings


def _frames(module, *, noise_px=0.0, drop=(), hidden=lambda t: (), seed=0, fps=30.0):
    rng = np.random.default_rng(seed)
    n = int(round(module.DURATION * fps))

    def pose(t):
        p = module.pose_at(t)
        return p + rng.normal(0, noise_px, p.shape) if noise_px else p

    payload = fixture_poses(0, n, fps, pose, (module.W, module.H), hidden)
    payload["content"]["items"] = [it for it in payload["content"]["items"] if it["frame_id"] not in drop]
    return unwrap(payload), n


def _squat(**kw):
    frames, n = _frames(squat_sample, **kw)
    return squat_analysis.analyze(frames, 30.0, n, width=squat_sample.W, height=squat_sample.H,
                                  settings=squat_settings.Settings())


def _jump(**kw):
    frames, n = _frames(jump_sample, **kw)
    return jump_analysis.analyze(frames, 30.0, n, width=jump_sample.W, height=jump_sample.H,
                                 settings=jump_settings.Settings())


# ── squat: bottom measurements survive keypoint jitter ────────────────────────

SQUAT_DEPTH = ["below", "roughly level", "above", "below"]   # the sample's 4 reps (no gap here)


def test_squat_depth_band_is_stable_under_keypoint_jitter():
    """Rep 2 (thigh horizontal) is 2.7% of leg length from the 4% band edge; read off a
    single bottom frame, 2 px of jitter flipped it ~1 time in 10."""
    right = total = 0
    lean_error = []
    for seed in range(20):
        result = _squat(noise_px=2, seed=seed)
        assert [a.status for a in result.attempts] == ["completed"] * 4
        for rep, depth in zip(result.attempts, SQUAT_DEPTH):
            total += 1
            right += rep.depth == depth
            lean_error.append(abs(rep.torso_lean - 33.0))
    assert right / total >= 0.97
    assert np.median(lean_error) <= 0.8


def test_squat_held_bottom_uses_the_whole_pause_and_clean_values_are_exact():
    clean = _squat()
    held = clean.attempts[1]                       # the 0.8 s pause at 4.3-5.1 s
    a, b = held.bottom_span
    assert (b - a) / 30.0 >= 0.6
    assert [r.torso_lean for r in clean.attempts] == [33.0] * 4
    touch_and_go = clean.attempts[0]
    assert touch_and_go.bottom_span[1] - touch_and_go.bottom_span[0] < 9


# ── standing jump: jitter, dropouts and a hidden far ankle ────────────────────

JUMP_TIMES = [(s + jump_sample.LEAVE + 1, s + jump_sample.LAND) for s in jump_sample.STARTS]
JUMP_BEND = [59.8, 57.2, 17.6, 54.5]


def test_clean_jump_track_is_read_frame_by_frame():
    clean = _jump()
    s = jump_analysis.make_signals(clean.points, clean.side, jump_sample.W, jump_sample.H, 30.0,
                                   jump_settings.Settings())
    assert not s.noisy and s.jitter < 0.6
    assert [a.metrics["landing_bend_deg"] for a in clean.attempts] == JUMP_BEND


def test_jump_recovery_and_timing_survive_keypoint_jitter():
    """Raw per-frame stillness checks failed on ~55% of jumps at 1-3 px of jitter."""
    errors = []
    for seed in range(10):
        for px in (2, 3):
            attempts = _jump(noise_px=px, seed=seed).attempts
            assert len(attempts) == 4
            for a, (takeoff, contact), bend in zip(attempts, JUMP_TIMES, JUMP_BEND):
                assert abs(a.frames["takeoff"] - takeoff) <= 1 and a.frames["contact"] == contact
                assert a.frames["recovery"] is not None
                assert not any(i["phase"] == "recovery" for i in a.issues)
                errors.append(a.metrics["landing_bend_deg"] - bend)
    e = np.array(errors)
    assert abs(e.mean()) < 1.0 and np.sqrt((e ** 2).mean()) < 3.5


def test_jittery_track_turns_on_fitted_readings():
    noisy = _jump(noise_px=2, seed=1)
    s = jump_analysis.make_signals(noisy.points, noisy.side, jump_sample.W, jump_sample.H, 30.0,
                                   jump_settings.Settings())
    assert s.noisy and 1.0 < s.jitter < 3.5


@pytest.mark.parametrize("phase", ["takeoff", "contact"])
def test_single_dropped_pose_frame_at_takeoff_or_contact_is_bridged(phase):
    i = 0 if phase == "takeoff" else 1
    frame = JUMP_TIMES[0][i]
    a = _jump(drop={frame}).attempts[0]
    assert abs(a.frames[phase] - frame) <= 1     # interpolated across the gap, as the note says
    assert "dropout" in a.methods[phase] and "±1 frame" in a.methods[phase]
    assert a.metrics["flight_time_estimate_s"] is not None


def test_scattered_dropped_frames_keep_every_flight_time():
    rng = np.random.default_rng(5)
    drop = set(np.flatnonzero(rng.random(375) < 0.05).tolist())
    attempts = _jump(drop=drop).attempts
    assert all(a.metrics["flight_time_estimate_s"] is not None for a in attempts[:3])


def test_long_dropout_still_stays_a_gap():
    """The sample's 9-frame hidden-shoe interval is far longer than a bridged dropout."""
    hidden = _jump(hidden=jump_sample.hidden).attempts[3]
    assert hidden.frames["contact"] is None


def test_hidden_far_ankle_is_explained_once_for_the_clip():
    result = _jump(hidden=lambda t: (J["left_ankle"],))
    warning = [w for w in result.warnings if "far ankle" in w]
    assert len(warning) == 1 and "100%" in warning[0] and "both shoes" in warning[0]
    assert all(a.frames["takeoff"] is None for a in result.attempts)    # two-foot rule unchanged
    assert not any("far ankle" in w for w in _jump().warnings)

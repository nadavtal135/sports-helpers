"""Gait analysis on synthetic pose replies with known cadence and controlled gaps."""

import json
import math

import numpy as np
import pytest

from running.settings import Settings
from running.src import gait, pose, synthetic
from running.src.skeleton import KPT_NAMES

FPS = 30.0
J = {n: i for i, n in enumerate(KPT_NAMES)}
LEG = [f"{s}_{j}" for s in ("left", "right") for j in ("hip", "knee", "ankle")]


def reply(n=300, *, drop=None, offset=0, noise=0.0):
    """Synthetic-runner pose reply with joints set to (0, 0) where *drop(frame) -> names*."""
    payload = synthetic.pose_payload(offset, n, FPS, noise=noise)
    for item in payload["content"]["items"]:
        if drop:
            for name in drop(item["frame_id"] + offset) or ():
                item["kpts_xy"][J[name]] = [0, 0]
        t = (item["frame_id"] + offset) / FPS
        if synthetic.GAP[0] <= t < synthetic.GAP[1]:   # undo the sample's built-in gap
            pts = synthetic.pose_at(t) / (synthetic.W, synthetic.H)
            for name in ("right_hip", "right_knee", "right_ankle"):
                if not (drop and name in (drop(item["frame_id"] + offset) or ())):
                    item["kpts_xy"][J[name]] = [round(float(v), 4) for v in pts[J[name]]]
    return payload


def analyze(payload, n=300, aspect=synthetic.W / synthetic.H, **kw):
    frames = pose.unwrap(payload)
    return gait.analyze(frames, FPS, n, settings=Settings(**kw), aspect=aspect)


def test_cadence_from_known_periodic_signal():
    a = analyze(reply())
    assert a.mean_cadence == pytest.approx(synthetic.CADENCE, abs=0.3)
    assert a.mean_stride == pytest.approx(synthetic.STRIDE, abs=0.005)
    assert a.cadence_confidence == "ok"
    # Feet alternate.
    feet = [s.foot for s in a.steps]
    assert all(x != y for x, y in zip(feet, feet[1:]))


def test_cadence_with_pose_noise():
    a = analyze(reply(noise=0.0015))
    assert a.mean_cadence == pytest.approx(synthetic.CADENCE, abs=1.0)


def test_aspect_ratio_correction():
    """Knee flexion and leg length come out right only with x scaled by width/height."""
    a = analyze(reply())
    hip, knee, ankle = (synthetic.pose_at(1.0)[J[f"right_{j}"]] for j in ("hip", "knee", "ankle"))
    v1, v2 = hip - knee, ankle - knee
    truth = 180 - math.degrees(math.acos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))
    assert a.knee_flexion["right"][30] == pytest.approx(truth, abs=1.0)
    assert a.leg_length == pytest.approx(synthetic.L / synthetic.H, rel=0.03)
    wrong = analyze(reply(), aspect=1.0)   # the stretched space the reference warned about
    assert abs(wrong.knee_flexion["right"][30] - truth) > 3


def test_observed_mask_is_model_returns_not_interpolation():
    """Regression: the reference took the mask after interpolation, so it was all True."""
    drop = lambda f: ["right_knee"] if f in (100, 101) else []
    frames = pose.unwrap(reply(drop=drop))
    idx = np.array([f["index"] for f in frames], float)
    x, y, observed = gait._joint_xy(frames, "right_knee", idx, 300, max_gap=3)
    assert not observed[100] and not observed[101] and observed[99] and observed[102]
    assert np.isfinite(x[100]) and np.isfinite(y[101])     # short gap filled for drawing...
    a = analyze(reply(drop=drop))
    assert a.observed["right"][100] is np.False_ or not a.observed["right"][100]


def test_interpolated_frames_never_become_knee_landmarks():
    # Knee missing for two frames around every right strike: the landmark window
    # leans on interpolated frames, so no knee value may be reported there.
    base = analyze(reply())
    near = {s.frame + k for s in base.steps_for("right") for k in (-1, 0)}
    a = analyze(reply(drop=lambda f: ["right_knee"] if f in near else []))
    assert all(np.isnan(s.knee_flexion) for s in a.steps_for("right"))
    assert any(np.isfinite(s.knee_flexion) for s in a.steps_for("left"))
    assert a.warnings["knee_landmarks_skipped_unobserved"] >= len(a.steps_for("right"))


def test_long_tracking_gap_is_not_interpolated_and_breaks_strides():
    drop = lambda f: LEG[3:] if 120 <= f < 150 else []   # right leg gone for 1 s
    a = analyze(reply(drop=drop))
    assert np.isnan(a.clearance["right"][120:150]).all()
    assert a.warnings["frames_in_long_gaps"]["right"] >= 30
    for s in a.steps_for("right"):
        if np.isfinite(s.stride):
            start = s.mid_stance - s.stride
            assert not (start * FPS < 150 and s.mid_stance * FPS > 120), "stride spans the gap"
    assert a.mean_cadence == pytest.approx(synthetic.CADENCE, abs=0.5)


def test_short_gap_is_bridged():
    a = analyze(reply(drop=lambda f: ["right_ankle"] if f in (60, 61) else []))
    assert np.isfinite(a.clearance["right"][60:62]).all()
    assert a.warnings["frames_in_long_gaps"]["right"] == 0


def test_gap_is_never_counted_as_ground_contact():
    """Regression: the reference labelled NaN frames as a peakless swing and folded them into stance."""
    drop = lambda f: LEG[3:] if 120 <= f < 150 else []
    a = analyze(reply(drop=drop))
    for s in a.steps_for("right"):
        if np.isfinite(s.contact_seconds):
            lo, hi = s.time * FPS, (s.time + s.contact_seconds) * FPS
            assert hi <= 120 or lo >= 150


def test_missing_limb_is_reported_and_other_foot_still_gives_cadence():
    a = analyze(reply(drop=lambda f: LEG[3:]))
    assert a.warnings["absent_feet"] == ["right"]
    assert not a.steps_for("right")
    assert a.mean_cadence == pytest.approx(synthetic.CADENCE, abs=0.5)


def test_both_limbs_missing_raises_clear_error():
    with pytest.raises(RuntimeError, match="Neither ankle"):
        analyze(reply(drop=lambda f: LEG))


def test_truncated_final_stance_is_counted_but_untimed():
    base = analyze(reply())
    last = base.steps[-1]
    n = int(last.time * FPS) + 3   # cut the clip a few frames into the last stance
    a = analyze(reply(n=n), n=n)
    final = a.steps[-1]
    assert final.time == pytest.approx(last.time, abs=0.01)
    assert np.isnan(final.mid_stance) and np.isnan(final.contact_seconds) and np.isnan(final.stride)
    assert a.warnings["untimed_strikes"] >= 1


def test_insufficient_strides():
    a = analyze(reply(n=24), n=24)        # under one stride
    assert np.isnan(a.mean_cadence) and a.cadence_confidence == "none"
    b = analyze(reply(n=60), n=60)        # a couple of strides
    assert b.cadence_confidence in ("low", "none")


def test_summary_is_json_safe():
    a = analyze(reply(drop=lambda f: LEG[3:] if 120 <= f < 150 else []))
    json.dumps({**a.summary(), "series": a.series()}, allow_nan=False)
    text = gait.summary_text(a, video_seconds=10)
    assert "heel" in text and "Cadence = 120" in text


# ── failure modes found on real footage (docs/EVALUATION.md) ─────────────────

def _signal(swing_heights, *, fps=60, stance_s=0.12, swing_s=0.30, dip_depth=0.0):
    """Clearance in leg lengths: stance at ~0, then a half-sine swing per height."""
    out = [np.zeros(int(stance_s * fps)) + dip_depth]
    for h in swing_heights:
        k = int(swing_s * fps)
        out.append(h * np.sin(np.linspace(0, np.pi, k + 2)[1:-1]))
        out.append(np.zeros(int(stance_s * fps)) + dip_depth)
    return np.concatenate(out)


def test_accelerating_runner_keeps_small_early_swings():
    """Real clip: swings grew from ~0.22 to ~0.6 leg lengths; a clip-wide gate dropped the early ones."""
    heights = [0.22, 0.24, 0.23, 0.25, 0.3, 0.45, 0.55, 0.6, 0.62, 0.6, 0.61, 0.6]
    sig = _signal(heights)
    amp = gait._robust_peak(sig)
    low, high = 0.18 * amp, 0.55 * amp
    adaptive, _ = gait._strikes(sig, "left", 60, high=high, low=low, min_frames=6, fraction=0.55)
    clipwide, _ = gait._strikes(sig, "left", 60, high=high, low=low, min_frames=6, fraction=None)
    assert len(adaptive) == len(heights)          # every swing ends in a strike
    assert len(clipwide) < len(heights)           # the reference gate loses the early ones


def test_wobble_at_bottom_of_stance_is_still_not_a_swing():
    sig = _signal([0.5, 0.5, 0.5])
    a = int(0.12 * 60) + int(0.30 * 60) + 3
    sig[a:a + 3] = 0.12    # a bump inside a stance: above the contact threshold, far below a swing
    amp = gait._robust_peak(sig)
    steps, _ = gait._strikes(sig, "left", 60, high=0.55 * amp, low=0.18 * amp, min_frames=6,
                             fraction=0.55)
    assert len(steps) == 3


def test_short_deep_contact_kept_shallow_short_dip_absorbed():
    deep = _signal([0.5] * 4, stance_s=0.07)                  # 4-frame real contacts
    amp = gait._robust_peak(deep)
    kept, _ = gait._strikes(deep, "left", 60, high=0.55 * amp, low=0.18 * amp, min_frames=6, fraction=0.55)
    assert len(kept) == 4
    # Swings touching end to end: each "stance" is a few frames of the sine's tails,
    # never near the ground (mid-swing threshold clipping), so it must be absorbed.
    shallow = _signal([0.5] * 4, stance_s=0.0)
    merged, _ = gait._strikes(shallow, "left", 60, high=0.55 * amp, low=0.18 * amp, min_frames=6, fraction=0.55)
    assert len(merged) < 4


def test_pause_mid_clip_does_not_drag_cadence_down():
    """Real clip: standing still mid-clip made 1-2 s 'strides' and read 168 instead of ~250 spm."""
    frozen = synthetic.pose_at(4.0 + synthetic.STRIDE * 0.1)
    payload = reply(n=420)
    for item in payload["content"]["items"]:
        if 150 <= item["frame_id"] < 240:      # 3 s standing still
            pts = frozen / (synthetic.W, synthetic.H)
            for name in LEG:
                item["kpts_xy"][J[name]] = [round(float(v), 4) for v in pts[J[name]]]
    a = analyze(payload, n=420)
    assert a.mean_cadence == pytest.approx(synthetic.CADENCE, abs=2.0)
    assert a.warnings["irregular_intervals_dropped"] >= 2

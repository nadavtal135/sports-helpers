"""Check which claims are supported by actual fixture landmarks and a lifter's targets."""

import copy

import pytest

from running.src.pose import unwrap
from squat import analysis, review, sample
from squat.pipeline import rep_rows
from squat.settings import Settings


@pytest.fixture()
def report():
    fps = sample.FPS
    offset = 15  # the original frame clock stays intact after a 0.5 s trim
    n = int(fps * sample.DURATION) - offset
    frames = unwrap(sample.pose_payload(offset, n, fps))
    result = analysis.analyze(frames, fps, n, width=sample.W, height=sample.H, settings=Settings())
    return {"meta": {"analysis_video": {"fps": fps}}, "reps": rep_rows(result, fps, offset)}


def test_two_default_observations_cite_measured_source_frames(report):
    feedback = review.for_set(report)
    assert feedback["targets"] == {}
    strength, comparison = feedback["strength"], feedback["closer_look"]
    assert strength["kind"] == "Consistency" and "torso lean" in strength["title"]
    assert [x["source_frame"] for x in strength["evidence"]] == [54, 141, 222]
    assert "Rep 3 was shallower than rep 1" in comparison["title"]
    assert [(x["source_frame"], x["measurement"]) for x in comparison["evidence"]] == [
        (222, "Hip above knee"), (54, "Hip below knee")]
    gap, = feedback["quality"]["4"]["evidence"]
    assert 9.55 <= gap["source_frame"] / sample.FPS < 10.25
    assert gap["phase"] == "tracking gap" and "ankle visible" in feedback["quality"]["4"]["detail"]


def test_targets_use_personal_depth_band_and_phase_specific_tolerance(report):
    feedback = review.for_set(report, depth="above", descent=0.6, ascent=1.2, tolerance=0.05)
    assert feedback["strength"]["kind"] == "Consistency"
    assert feedback["closer_look"]["kind"] == "Your target"
    assert [x["rep"] for x in feedback["closer_look"]["evidence"]] == [1, 3]
    checks = feedback["checks"]
    assert [c["status"] for c in checks["3"]] == ["met", "met", "different"]
    assert checks["3"][2]["difference"] == "0.60 s shorter"
    assert checks["2"][2]["status"] == "met" and checks["1"][0]["status"] == "different"
    assert all(c["status"] == "unknown" for c in checks["4"])
    for r in report["reps"][:3]:
        assert r["source_frame_start"] < r["source_frame_descent"] < r["source_frame_bottom"]
        assert r["source_frame_bottom"] < r["source_frame_ascent"] < r["source_frame_end"]
    assert review.for_set(report, descent=0.8, tolerance=0.2)["checks"]["3"][0]["status"] == "met"


def test_consistency_and_missing_target_can_coexist(report):
    rows = report["reps"][:3]
    for row in rows:
        row.update(depth_2d="above", bottom_hip_minus_knee_leg_lengths=-0.2)
    report["reps"] = rows
    feedback = review.for_set(report, depth="below")
    assert "same visible depth band" in feedback["strength"]["title"]
    assert all(checks[0]["status"] == "different" for checks in feedback["checks"].values())


def test_a_barely_crossed_depth_band_is_not_featured_as_a_depth_change(report):
    report["reps"] = report["reps"][:2]
    a, b = report["reps"]
    a.update(depth_2d="above", bottom_hip_minus_knee_leg_lengths=-0.041)
    b.update(depth_2d="roughly level", bottom_hip_minus_knee_leg_lengths=-0.039)
    assert review.for_set(report)["closer_look"]["kind"] != "Depth comparison"


def test_repeated_depth_and_a_shallower_fourth_rep_cite_the_whole_reference_group(report):
    template = report["reps"][0]
    references = [dict(template, attempt=i, rep=i, source_frame_bottom=60 * i) for i in (1, 2, 3)]
    shallow = dict(report["reps"][2], attempt=4, rep=4, source_frame_bottom=245, torso_lean_at_bottom_deg=44)
    report["reps"] = [*references, shallow]
    feedback = review.for_set(report)
    assert "same visible depth band on reps 1–3" in feedback["strength"]["title"]
    look = feedback["closer_look"]
    assert look["title"] == "Rep 4 was shallower than reps 1–3."
    assert [e["source_frame"] for e in look["evidence"]] == [245, 60, 120, 180]


def test_oblique_framing_identifies_the_position_comparison_and_a_frame():
    n = 90
    p = sample.pose_payload(0, n, sample.FPS)
    for person in p["content"]["items"]:
        for name in ("left_hip", "left_shoulder"):
            person["kpts_xy"][sample.J[name]][0] -= 0.23
    result = analysis.analyze(unwrap(p), sample.FPS, n, width=sample.W, height=sample.H, settings=Settings())
    assert result.side == "right" and result.attempts[0].status == "completed"
    assert 0 <= result.capture_frame < n
    assert "depth and angle comparisons uncertain" in result.capture_note
    assert "directly from beside" in result.capture_note


def test_incomplete_or_unseen_landmarks_are_excluded_from_comparisons(report):
    bad, missing = report["reps"][:2]
    bad.update(status="incomplete", rep=None, source_frame_start=None)
    missing.update(depth_2d="unknown", torso_lean_at_bottom_deg=None)
    report["reps"] = [bad, missing]
    feedback = review.for_set(report, depth="below")
    assert all(r[0]["status"] == "unknown" for r in feedback["checks"].values())
    assert feedback["strength"]["kind"] == "Rep reference"
    assert "standing pause" in feedback["quality"]["1"]["detail"]
    assert "bottom torso lean" in feedback["quality"]["2"]["detail"]


def test_poor_camera_view_suppresses_spatial_claims_but_keeps_observed_tempo(report):
    report["capture"] = {"source_frame": 45, "message": "Film directly from the side."}
    feedback = review.for_set(report, depth="above", descent=0.6, tolerance=1)
    assert all(c[0]["status"] == "unknown" for c in feedback["checks"].values())
    assert feedback["checks"]["3"][1]["status"] == "met"
    assert feedback["closer_look"]["kind"] == "Camera view"
    assert feedback["closer_look"]["evidence"][0]["source_frame"] == 45
    other_target = review.for_set(report, depth="above", descent=8)
    assert other_target["closer_look"]["kind"] == "Camera view"
    assert "Camera view" in other_target["checks"]["3"][0]["difference"]
    assert other_target["checks"]["3"][1]["status"] == "different"


def test_saved_pre_review_jobs_keep_source_clock_and_unknown_peaks(report):
    legacy = copy.deepcopy(report)
    for r in legacy["reps"]:
        for phase in ("descent", "ascent", "gap", "peak_flexion"):
            r.pop("source_frame_" + phase)
    review.legacy_landmarks(legacy)
    first = legacy["reps"][0]
    assert first["source_frame_start"] < first["source_frame_descent"] < first["source_frame_bottom"]
    assert "source_frame_peak_flexion" not in first
    assert review.for_set(legacy)["closer_look"]["evidence"][0]["source_frame"] == 222


def test_saved_camera_warning_links_to_clip_context_and_suppresses_spatial_targets(report):
    report["meta"]["trim"] = {"source_frame_offset": 15}
    report["warnings"] = ["The ankle shifts substantially in the image. Check camera movement and framing."]
    review.legacy_landmarks(report)
    feedback = review.for_set(report, depth="below")
    assert feedback["checks"]["1"][0]["status"] == "unknown"
    cue, = feedback["closer_look"]["evidence"]
    assert cue["phase"] == "clip start" and cue["source_frame"] == 15


@pytest.mark.parametrize("invalid", [dict(depth="ideal"), dict(ascent=float("nan")), dict(descent=-1),
                                     dict(tolerance=float("inf")), dict(tolerance=-0.1)])
def test_invalid_targets(report, invalid):
    with pytest.raises(ValueError):
        review.for_set(report, **invalid)

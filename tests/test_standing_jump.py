"""Frame-level expectations use the drawn motion, independently of the detector's thresholds."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from running.src.pose import unwrap
from running.src.side_view_sample import J, fixture_poses
from standing_jump import analysis, pipeline, review, sample, settings
from standing_jump.foot_contact import ShoeContact


def detect(start=0, stop=375, scene=sample.pose_at, hidden=sample.hidden, fps=30):
    payload = fixture_poses(start, stop - start, fps, scene, (sample.W, sample.H), hidden)
    found = analysis.analyze(unwrap(payload), fps, stop - start, width=sample.W, height=sample.H, settings=settings.Settings())
    return pipeline.attempt_rows(found, fps, start), found


def test_clean_jump_matches_visible_crossings_and_joint_change():
    rows, result = detect(15, 101)
    a, = rows
    f, m = a["frames"], a["metrics"]
    assert result.side == "right" and a["status"] == "confirmed" and not a["issues"]
    assert f["dip_start"] < f["countermovement"] == 36 < f["takeoff"] == 46
    assert f["takeoff"] < f["flight"] < f["contact"] == 60 < f["landing_bottom"] == 67 < f["recovery"]
    assert a["times_s"]["contact"] == 2.0
    assert m["flight_time_estimate_s"] == pytest.approx((60 - 46) / 30, abs=0.001)
    assert m["landing_bend_deg"] == pytest.approx(m["landing_knee_deg"] - m["contact_knee_deg"], abs=0.1)
    assert m["landing_torso_deg"] == pytest.approx(24.5, abs=0.2)
    insight = review.for_set(rows)["strength"]
    assert [c["source_frame"] for c in insight["evidence"]] == [60, 67]
    assert "after contact" in insight["title"]


def test_repeated_jumps_compare_specific_measured_frames_after_trim():
    rows, _ = detect(15, 374)
    assert [a["status"] for a in rows] == ["confirmed", "confirmed", "confirmed", "partial"]
    assert [a["frames"]["takeoff"] for a in rows] == [46, 130, 217, 310]
    assert [a["frames"]["contact"] for a in rows] == [60, 144, 231, None]
    feedback = review.for_set(rows)
    assert "descent timing was similar" in feedback["strength"]["title"]
    assert [c["source_frame"] for c in feedback["strength"]["evidence"]] == [36, 120, 207, 300]
    look = feedback["closer_look"]
    assert "attempt 3" in look["title"] and "attempts 1 and 2" in look["title"]
    assert [c["source_frame"] for c in look["evidence"]] == [238, 67, 151, 231, 60, 144]


def test_shallow_grounded_dip_and_arm_swing_are_not_a_jump():
    standing = sample.pose_at(0)

    def dip_only(t):
        p = sample.pose_at(t).copy()
        # Remove whole-body flight and retain an abbreviated countermovement and arm swing.
        for side in ("left", "right"):
            indices = [j for name, j in J.items() if name.startswith(side)]
            p[indices, 1] += standing[J[f"{side}_ankle"], 1] - p[J[f"{side}_ankle"], 1]
        return standing + 0.6 * (p - standing)

    rows, _ = detect(0, 100, scene=dip_only)
    assert rows == []


def test_recovery_step_requires_lift_and_displaced_replant():
    rows, _ = detect(86, 187)
    a, = rows
    assert a["step"] == "observed" and a["frames"]["contact"] < a["frames"]["step_lift"] < a["frames"]["step_plant"]
    assert a["frames"]["step_plant"] < a["frames"]["recovery"]
    closer = review.for_set(rows)["closer_look"]
    assert "step" in closer["title"]
    assert [c["source_frame"] for c in closer["evidence"]] == [164, 169]
    other, _ = detect(0, 101)
    assert other[0]["step"] == "no visible step"


def test_hidden_contact_keeps_preparation_and_recovery_independent():
    rows, _ = detect(269, 375)
    a, = rows
    assert a["frames"]["takeoff"] == 310 and a["frames"]["flight"] is not None
    assert a["frames"]["contact"] is a["frames"]["landing_bottom"] is None
    assert a["visible_landing_bend"] is a["metrics"]["landing_bend_deg"] is a["metrics"]["flight_time_estimate_s"] is None
    assert a["frames"]["recovery"] is not None and a["status"] == "partial"
    issue = next(q for q in a["issues"] if q["phase"] == "contact")
    assert sample.HIDDEN[0] <= issue["source_frame"] < sample.HIDDEN[1]
    assert "obscured" in issue["detail"]


def test_clip_ends_in_flight_and_clip_begins_in_flight():
    rows, _ = detect(0, 58)
    a, = rows
    assert a["cues"]["takeoff"]["source_frame"] == 46
    assert "contact" not in a["cues"] and "recovery" not in a["cues"]
    assert any("trim ends" in q["detail"] for q in a["issues"])
    rows, _ = detect(49, 107)
    a, = rows
    assert a["frames"]["takeoff"] is a["metrics"]["flight_time_estimate_s"] is None
    assert a["frames"]["contact"] == 60 and a["metrics"]["landing_bend_deg"] is not None


def test_only_one_visible_ankle_cannot_confirm_two_foot_contacts():
    rows, _ = detect(0, 101, hidden=lambda t: (J["left_ankle"],))
    a, = rows
    assert a["status"] == "partial" and a["frames"]["flight"] is not None
    assert a["frames"]["takeoff"] is a["frames"]["contact"] is None
    assert any("one ankle" in q["detail"] for q in a["issues"])


def test_personal_target_uses_each_available_phase_and_can_be_unknown():
    rows, _ = detect()
    feedback = review.for_set(rows, "landing_bend_deg", 45, 65)
    assert [check["status"] for check in feedback["checks"].values()] == ["met", "met", "below", "unknown"]
    assert "attempts 1 and 2" in feedback["strength"]["title"]
    assert [c["source_frame"] for c in feedback["closer_look"]["evidence"]] == [231, 238]
    assert review.for_set(rows, "descent_s", 0.25, 0.40)["checks"]["4"]["status"] == "met"
    for params in [(None, 1, 4), ("height", 0, 2), ("descent_s", 2, 1), ("landing_bend_deg", 0, 200),
                   ("descent_s", np.nan, 1)]:
        with pytest.raises(ValueError):
            review.for_set(rows, *params)


def test_oblique_view_keeps_landing_evidence_and_withholds_angle_comparisons():
    def oblique(t):
        xy = sample.pose_at(t)
        for name, j in J.items():
            if name.startswith("left_"):
                xy[j, 0] -= 60
        return xy

    rows, result = detect(0, 101, scene=oblique)
    a, = rows
    assert any("directly beside" in msg for msg in result.warnings)
    assert a["status"] == "confirmed" and a["visible_landing_bend"]
    assert all(v is None for k, v in a["metrics"].items() if k.endswith("_deg"))
    assert a["metrics"]["descent_s"] is not None
    note = review.for_set(rows)["strength"]
    assert "continued to bend" in note["title"]
    assert [c["source_frame"] for c in note["evidence"]] == [60, 67]
    assert review.for_set(rows, "landing_bend_deg", 45, 65)["checks"]["1"]["status"] == "unknown"


def test_connected_jumps_share_a_lowest_position_without_a_standing_recovery():
    def rebound(t):
        frame = t * 30
        return sample.pose_at(t if frame <= 67 else (36 + frame - 67) / 30)

    rows, _ = detect(0, 136, scene=rebound, hidden=lambda _: ())
    first, second = rows
    assert first["frames"]["recovery"] is None and second["frames"]["dip_start"] is None
    assert first["frames"]["contact"] < first["frames"]["landing_bottom"] < second["frames"]["takeoff"]
    assert abs(first["frames"]["landing_bottom"] - second["frames"]["countermovement"]) <= 2
    assert first["rebounds_into_next"] and second["frames"]["recovery"] is not None
    note = review.for_set(rows)["closer_look"]
    assert note["kind"] == "Recovery" and "flowed into the next dip" in note["title"]
    assert [c["source_frame"] for c in note["evidence"]] == [60, 68]


@pytest.mark.parametrize("false_contact", [False, True])
def test_shoe_pixels_refine_an_ankle_estimate_only_across_a_clear_transition(tmp_path, false_contact):
    path = tmp_path / "shoes.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (180, 140))
    assert writer.isOpened()
    for frame in range(50):
        img = np.full((140, 180, 3), 38, np.uint8)
        lift = 30 if 18 <= frame < 31 and not (false_contact and frame == 21) else 0
        for x in (55, 110):
            cv2.rectangle(img, (x - 7, 98 - lift), (x + 11, 110 - lift), (220, 220, 220), -1)
        writer.write(img)
    writer.release()
    s = SimpleNamespace(fps=60, scale=100, ankles=np.tile([[55., 100.], [110., 100.]], (50, 1, 1)),
                        clearance=np.zeros((50, 2)), grounded=np.ones((50, 2), dtype=bool))
    shoes = ShoeContact(path)
    try:
        assert shoes("takeoff", 15, 25, s) == (None if false_contact else 18)
        assert shoes("contact", 35, 25, s) == 31
        s.clearance[28, 1] = np.nan
        assert shoes("contact", 35, 25, s) is None
    finally:
        shoes.close()

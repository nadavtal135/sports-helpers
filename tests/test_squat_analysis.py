"""One-person side-view cycles with independently specified timing and joint angles."""

import numpy as np
import pytest

from running.src.pose import unwrap
from squat import analysis, sample
from squat.pipeline import rep_rows
from squat.settings import Settings

FPS = sample.FPS
J = sample.J


def analyze(start, end, edit=None):
    offset = round(start * FPS)
    n = round((end - start) * FPS)
    payload = sample.pose_payload(offset, n, FPS)
    if edit:
        for person in payload["content"]["items"]:
            edit(person, (person["frame_id"] + offset) / FPS)
    return analysis.analyze(unwrap(payload), FPS, n, width=sample.W, height=sample.H, settings=Settings())


def test_clean_rep_has_ordered_phases_and_projected_metrics():
    a = analyze(0, 3.1)
    rep, = a.attempts
    assert a.side == "right" and rep.status == "completed"
    assert rep.start < rep.bottom < rep.end
    assert rep.start / FPS == pytest.approx(0.8, abs=0.2)
    assert rep.bottom / FPS == pytest.approx(1.8, abs=0.04)
    assert rep.end / FPS == pytest.approx(2.8, abs=0.2)
    # 26 degree shank + 107 degree thigh. Using normalized xy without accounting
    # for the portrait aspect ratio would give a different angle.
    assert rep.peak_flexion == pytest.approx(133, abs=1)
    assert rep.torso_lean == pytest.approx(33, abs=0.2)
    assert rep.depth == "below"
    phases = [p for i, p in enumerate(a.phases) if i == 0 or p != a.phases[i - 1]]
    assert phases == ["standing", "descent", "bottom", "ascent", "standing"]


@pytest.mark.parametrize("bounce", [False, True])
def test_paused_rep_counts_once_even_with_a_bottom_bounce(bounce):
    # During the 0.8 s hold, replay a few near-bottom positions up then down.
    def edit(person, t):
        if bounce and 4.45 <= t < 4.95:
            earlier = 4.3 - 0.18 * np.sin(np.pi * (t - 4.45) / 0.5)
            person["kpts_xy"] = (sample.pose_at(earlier) / (sample.W, sample.H)).tolist()
    a = analyze(3.1, 6.4, edit)
    rep, = a.attempts
    assert rep.status == "completed" and rep.depth == "roughly level"
    assert 4.25 <= rep.bottom / FPS + 3.1 <= 5.15
    assert set(a.phases[round((4.3 - 3.1) * FPS):round((5.1 - 3.1) * FPS)]) == {"bottom"}


def test_shallow_rep_counts_after_return_to_standing():
    rep, = analyze(6.3, 8.55).attempts
    assert rep.status == "completed" and rep.depth == "above"
    assert rep.peak_flexion == pytest.approx(79, abs=1)


def test_clip_starting_mid_rep_requires_a_fresh_standing_start():
    a = analyze(1.5, 6.4)
    first, second = a.attempts
    assert first.status == "incomplete" and first.start is None
    assert first.bottom is not None and first.end is not None
    assert second.status == "completed"
    partial, full = rep_rows(a, FPS, 45)
    assert partial["start_s"] is partial["descent_s"] is partial["rep"] is None
    assert partial["review_start_s"] == 1.5
    assert partial["bottom_s"] == pytest.approx(1.8)
    assert full["rep"] == 1 and full["bottom_s"] == pytest.approx(4.7, abs=0.05)


@pytest.mark.parametrize("start,end", [(0, 1.5), (1.8, 3.1), (1.95, 2.25)])
def test_cut_descent_or_ascent_has_no_invented_bottom(start, end):
    a = analyze(start, end)
    rep, = a.attempts
    assert rep.status == "incomplete"
    assert rep.bottom is rep.peak_flexion is rep.torso_lean is None
    assert rep.depth == "unknown"
    assert (rep.start is None) == (start != 0)
    assert (rep.end is None) == (end < 2.8)


def test_tracking_gap_around_bottom_is_one_uncertain_interval():
    a = analyze(0, 12)
    assert [r.status for r in a.attempts] == ["completed"] * 3 + ["uncertain"]
    gap = a.attempts[-1]
    assert gap.start / FPS < sample.GAP[0] < sample.GAP[1] < gap.end / FPS
    assert gap.bottom is gap.peak_flexion is None and gap.depth == "unknown"
    assert set(a.phases[287:308]) == {"tracking gap"}
    assert any("0.70 s" in w for w in a.warnings)
    row = rep_rows(a, FPS, 0)[-1]
    assert row["descent_s"] is row["ascent_s"] is None


@pytest.mark.parametrize("start,end", [(0.9, 3.1), (0, 2.733)])
def test_moving_leg_near_extension_at_clip_boundary_is_not_a_standing_pause(start, end):
    rep, = analyze(start, end).attempts
    assert rep.status == "incomplete"
    assert (rep.start is None) == (start > 0)
    assert (rep.end is None) == (end < 2.8)


def test_side_is_selected_once_and_uses_keypoint_confidence():
    def edit(person, t):
        for name in ("hip", "knee", "ankle", "shoulder"):
            person["kpts_score"][J[f"right_{name}"]] = 0.08 if t > 0.5 else 0.99
        if 1.55 <= t <= 2.1:
            person["kpts_score"][J["left_knee"]] = 0.0
            person["kpts_score"][J["right_knee"]] = 0.99
            person["kpts_score"][J["right_hip"]] = 0.99
            person["kpts_score"][J["right_ankle"]] = 0.99
    a = analyze(0, 3.1, edit)
    assert a.side == "left" and a.coverage["left"] > a.coverage["right"]
    assert a.attempts[0].status == "uncertain" and a.phases[54] == "tracking gap"


@pytest.mark.parametrize("joint,expected_depth,expected_lean", [
    ("hip", "unknown", None), ("knee", "unknown", 33), ("shoulder", "below", None)])
def test_low_confidence_bottom_metrics_are_withheld_independently(joint, expected_depth, expected_lean):
    def edit(person, t):
        if 1.73 < t < 1.87:
            person["kpts_score"][J[f"right_{joint}"]] = 0.40
    rep, = analyze(0, 3.1, edit).attempts
    assert rep.status == "completed" and rep.depth == expected_depth
    assert rep.torso_lean == expected_lean


def test_interpolated_bottom_has_no_depth_or_torso_measurement():
    def edit(person, t):
        if abs(t - 1.8) < 0.04:
            person["kpts_xy"][J["right_hip"]] = [0, 0]
    rep, = analyze(0, 3.1, edit).attempts
    assert rep.status == "completed"
    assert rep.depth == "unknown" and rep.torso_lean is None


def test_standing_alone_and_untracked_footage_produce_no_reps():
    assert analyze(0, 0.8).attempts == []
    assert analyze(4.35, 5.0).attempts == []   # stationary bottom hold, no excursion

    def no_pose(person, t):
        person["kpts_score"] = [0] * 17
    a = analyze(0, 3.1, no_pose)
    assert a.side is None and not a.attempts and a.warnings
    assert set(a.phases) == {"tracking gap"}

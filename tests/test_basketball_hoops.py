"""Hoop choice, per-camera-segment hoops, scene cuts and shooter tracking across
tracker id changes: the failure modes of a real portrait workout clip
(wrong far-court hoop, camera cuts, shooter posed on 17% of frames)."""

import base64
import dataclasses

import cv2
import numpy as np
import pytest

from basketball import pipeline, sample, settings
from basketball.src import hoop as hoop_mod
from basketball.src import pose, scenes, synthetic
from basketball.src.cache import Cache
from basketball.src.providers.fixture import FixtureProvider

W, H = 576, 1024   # portrait, like the clip that failed


def two_hoop_reply(*, near_score=0.62, far_score=0.95):
    """A big near rim at the top of the frame and a small far-court hoop with a HIGHER score."""
    labels = np.zeros((H, W), np.uint8)
    cv2.ellipse(labels, (288, 140), (150, 18), 0, 0, 360, 1, -1)                # near rim
    cv2.fillPoly(labels, [np.array([[160, 140], [416, 140], [360, 300], [216, 300]])], 1)
    cv2.ellipse(labels, (470, 520), (14, 3), 0, 0, 360, 2, -1)                  # far rim
    cv2.fillPoly(labels, [np.array([[458, 520], [482, 520], [478, 540], [462, 540]])], 2)
    ok, png = cv2.imencode(".png", labels)

    def box(inst):
        ys, xs = np.nonzero(labels == inst)
        return [xs.min() / W, ys.min() / H, (np.ptp(xs) + 1) / W, (np.ptp(ys) + 1) / H]

    return {"content": {"object": "img.segment.masks",
                        "items": [{"instance_id": 2, "score": far_score, "bbox_xywh": box(2)},
                                  {"instance_id": 1, "score": near_score, "bbox_xywh": box(1)}],
                        "mask": {"format": "png",
                                 "data": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()}}}


# ── which hoop ───────────────────────────────────────────────────────────────

def test_two_hoops_largest_rim_wins_over_higher_score():
    hoops = hoop_mod.candidates(two_hoop_reply(), frame=0, width=W, height=H)
    assert len(hoops) == 2
    chosen, why = hoop_mod.choose(hoops, "auto")
    assert chosen.cx == pytest.approx(288, abs=3) and chosen.width > 250
    assert "largest" in why
    # locate() (the old single-hoop entry point) agrees; the reference took the far one.
    assert hoop_mod.locate(two_hoop_reply(), frame=0, width=W, height=H).width > 250


@pytest.mark.parametrize("mode, expect_cx", [("highest_score", 470), ("left", 288),
                                             ("right", 470), ("largest", 288)])
def test_hoop_override_setting_is_respected(mode, expect_cx):
    s = settings.from_user({"hoop_choice": mode})
    hoops = hoop_mod.candidates(two_hoop_reply(), frame=0, width=W, height=H)
    chosen, _ = hoop_mod.choose(hoops, s.hoop_choice)
    assert chosen.cx == pytest.approx(expect_cx, abs=3)


def test_hoop_override_setting_is_validated():
    with pytest.raises(ValueError):
        settings.from_user({"hoop_choice": "middle"})
    assert "hoop_choice" in settings.USER_SETTINGS     # shown in the web UI


def test_single_hoop_behaviour_unchanged():
    reply = synthetic.segment_payload(1280, 720)
    hoops = hoop_mod.candidates(reply, frame=0, width=1280, height=720)
    assert len(hoops) == 1
    assert hoop_mod.choose(hoops, "highest_score")[0] == hoop_mod.choose(hoops, "auto")[0]
    assert hoop_mod.choose(hoops)[1] == "only hoop found"


def _drop_through(hoop, n=60, start=10):
    """A ball falling straight through *hoop*'s rim line."""
    x, y = np.full(n, np.nan), np.full(n, np.nan)
    for k in range(start, start + 20):
        x[k] = hoop.cx
        y[k] = hoop.cy - 3 * hoop.width + (k - start) * 0.3 * hoop.width
    return x, y


def test_ball_trajectory_overrides_size_in_auto():
    near, far = sorted(hoop_mod.candidates(two_hoop_reply(), frame=0, width=W, height=H),
                       key=lambda h: -h.width)
    x, y = _drop_through(far)
    chosen, switched = hoop_mod.prefer_by_ball([near, far], near, x, y)
    assert switched and chosen is far
    x, y = _drop_through(near)
    chosen, switched = hoop_mod.prefer_by_ball([near, far], near, x, y)
    assert not switched and chosen is near
    # No crossings either way: keep the size-based choice.
    chosen, switched = hoop_mod.prefer_by_ball([near, far], near, np.full(40, np.nan), np.full(40, np.nan))
    assert not switched


def test_timeline_lookup():
    a = hoop_mod.Hoop(100, 100, 80, 120, 95, 140, 0.9, 5)
    b = hoop_mod.Hoop(300, 120, 280, 320, 115, 160, 0.9, 50)
    t = hoop_mod.HoopTimeline([hoop_mod.HoopSegment(0, 40, a, [a]), hoop_mod.HoopSegment(40, 80, b, [b]),
                               hoop_mod.HoopSegment(80, 90, None, [])])
    assert t.at(10) is a and t.at(40) is b and t.at(85) is None and t.at(500) is None
    assert t.segment_index(79) == 1 and t.hoops() == [a, b]


# ── camera cuts ──────────────────────────────────────────────────────────────

def test_cuts_from_differences():
    rng = np.random.default_rng(1)
    diff = rng.uniform(0.5, 2.5, 400)
    diff[180:200] = rng.uniform(4, 7, 20)      # a fast pan / a player close to the lens
    diff[100] = 35.0                           # cut
    diff[250] = 28.0                           # cut
    diff[252] = 20.0                           # same transition, weaker: merged
    diff[190] = 14.0                           # above threshold but not 4x the local motion
    assert scenes.cuts_from_differences(diff, threshold=12, min_gap=15) == [100, 250]


def test_segments_split_at_cuts_and_long_stretches():
    assert scenes.segments(300, [100, 250]) == [(0, 100), (100, 250), (250, 300)]
    segs = scenes.segments(300, [100], max_len=60)
    assert segs[0] == (0, 50) and segs[1] == (50, 100)
    assert all(b - a <= 60 for a, b in segs) and segs[-1][1] == 300


@pytest.fixture(scope="module")
def cut_clip(tmp_path_factory):
    """The synthetic session, cut to a second camera at 8.2 s (shots 4 and 5 after it)."""
    d = tmp_path_factory.mktemp("cut")
    old = synthetic.CUT_AT
    synthetic.CUT_AT = 8.2
    try:
        clip = sample.make_sample(d / "cut.mp4")
    finally:
        synthetic.CUT_AT = old      # never leak the cut into other tests
    return clip, d


def _run(clip, d, name, s):
    old = synthetic.CUT_AT
    synthetic.CUT_AT = 8.2
    try:
        return pipeline.run(clip, d / name, settings=s, gateway=settings.Gateway(),
                            provider=FixtureProvider(), cache=Cache(d / "cache"))
    finally:
        synthetic.CUT_AT = old


def test_per_segment_hoops_are_used_after_a_cut(cut_clip):
    clip, d = cut_clip
    r = _run(clip, d, "per_segment", settings.Settings())
    assert [s["result"] for s in r["shots"]] == synthetic.EXPECTED
    assert len(r["meta"]["camera_cuts"]) == 1
    hoops = [h["hoop"] for h in r["meta"]["hoops"] if h["hoop"]]
    assert len(hoops) == 2 and abs(hoops[0]["cx"] - hoops[1]["cx"]) > 200
    assert [s["camera_segment"] for s in r["shots"]] == [0, 0, 0, 1, 1]
    # Control: one hoop for the whole clip (the old behaviour) misjudges the last shot.
    one = _run(clip, d, "one_hoop", dataclasses.replace(settings.Settings(), hoop_frame_seconds=1.0))
    assert one["shots"][4]["result"] != "make"


def test_bundled_sample_has_no_false_cut(tmp_path):
    clip = sample.make_sample(tmp_path / "s.mp4")
    diff = scenes.frame_differences(clip)
    assert scenes.cuts_from_differences(diff, threshold=settings.Settings().scene_cut_threshold) == []


# ── the shooter across tracker id changes ────────────────────────────────────

def _person(tid, x, y, *, up=False, w=0.2, h=0.5):
    k = np.full((17, 2), 0.0)
    k[:, 0] = x + w / 2
    k[:5, 1] = y + 0.05                         # head
    k[5:9, 1] = y + 0.15
    k[[9, 10], 1] = y if up else y + 0.30       # wrists overhead, or down
    k[11:, 1] = y + 0.35
    return {"track_id": tid, "kpts_xy": k.tolist(), "bbox_xywh": [x, y, w, h]}


def _coverage(frames, near_x):
    return sum(1 for f in frames if f["persons"] and abs(f["persons"][0]["bbox_xywh"][0] - near_x) < 0.1)


def test_shooter_survives_track_id_changes():
    """Regression: keep_track() kept one id, which on a real clip was 461 of 2639 frames."""
    frames = []
    for i in range(300):
        tid = 1 if i < 90 else 7 if i < 200 else 12          # the tracker re-ids twice
        x = 0.30 + 0.0005 * i                                 # the shooter drifts slowly
        up = 40 <= i < 46 or 150 <= i < 156 or 250 <= i < 256
        frames.append({"index": i, "persons": [_person(tid, x, 0.3, up=up)]})
    info = pose.select_shooter(frames, [(0, 300)])
    assert all(len(f["persons"]) == 1 for f in frames)        # nothing thrown away
    assert info["segments"][0] is not None
    # The old rule would have kept only the id with the most overhead frames.
    old = [{"index": f["index"], "persons": list(f["persons"])} for f in frames]
    pose.keep_track(old, pose.pick_shooter(old))
    assert sum(1 for f in old if f["persons"]) <= 110


def test_shooter_followed_through_id_changes_with_a_bystander():
    frames = []
    for i in range(300):
        tid = 3 if i < 120 else 9                             # shooter re-id'd at 120
        shooter = _person(tid, 0.25, 0.3, up=(30 <= i < 38), w=0.15, h=0.45)
        bystander = _person(50, 0.70, 0.2, w=0.25, h=0.7)     # bigger, never shoots
        frames.append({"index": i, "persons": [bystander, shooter]})
    pose.select_shooter(frames, [(0, 300)])
    assert _coverage(frames, 0.25) == 300
    assert not any(p["track_id"] == 50 for f in frames for p in f["persons"])


def test_shooter_repicked_per_camera_segment():
    frames = []
    for i in range(200):
        if i < 100:   # camera 1: shooter on the left
            persons = [_person(1, 0.2, 0.3, up=(20 <= i < 26)), _person(2, 0.7, 0.3)]
        else:         # camera 2: new ids, shooter now on the right, the other on the left
            persons = [_person(5, 0.2, 0.3), _person(6, 0.7, 0.3, up=(130 <= i < 136))]
        frames.append({"index": i, "persons": persons})
    info = pose.select_shooter(frames, [(0, 100), (100, 200)])
    assert info["segments"] == [1, 6]
    assert all(f["persons"][0]["track_id"] == (1 if f["index"] < 100 else 6) for f in frames)


def test_crowd_without_a_shooter_is_left_for_dedupe():
    frames = [{"index": i, "persons": [_person(1, 0.2, 0.3), _person(2, 0.6, 0.3, w=0.3)]}
              for i in range(50)]
    info = pose.select_shooter(frames, [(0, 50)])
    assert info["unresolved_segments"] == 1
    assert all(len(f["persons"]) == 2 for f in frames)        # untouched; dedupe picks the largest

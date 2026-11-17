"""The basketball pipeline end to end on the synthetic clip with fixture replies,
plus parsing of recorded gateway-shaped replies."""

import json
from pathlib import Path

import pytest

from basketball import pipeline, sample, settings
from basketball.src import ball, hoop, pose, synthetic
from basketball.src.cache import Cache
from basketball.src.providers.fixture import FixtureProvider

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("bb")
    src = sample.make_sample(d / "sample.mp4")
    return pipeline.run(src, d / "out", settings=settings.Settings(), gateway=settings.Gateway(),
                        provider=FixtureProvider(), cache=Cache(d / "cache")), d


def test_outcomes_match_scripted_ground_truth(result):
    r, _ = result
    assert [s["result"] for s in r["shots"]] == synthetic.EXPECTED


def test_stats_are_consistent(result):
    r, _ = result
    s = r["stats"]
    results = [x["result"] for x in r["shots"]]
    assert s["makes"] == results.count("make") and s["misses"] == results.count("miss")
    assert s["unknown"] == results.count("unknown") == 1
    assert s["judged"] == s["makes"] + s["misses"]
    assert s["fg_pct"] == pytest.approx(s["makes"] / s["judged"], abs=1e-4)


def test_exports_keep_unknown_distinct(result):
    r, d = result
    out = d / "out"
    for name in ("basketball_annotated.mp4", "shot_chart.png", "ball_paths.png", "shots.csv",
                 "frames.csv", "analysis.json"):
        assert (out / name).is_file(), name
    rows = (out / "shots.csv").read_text().splitlines()[1:]
    assert [row.split(",")[1] for row in rows] == synthetic.EXPECTED
    data = json.loads((out / "analysis.json").read_text())
    assert data["stats"]["unknown"] == 1
    assert "excluded" in data["notes"]["fg_pct"]


def test_trim_offsets_times_to_original_clip(tmp_path):
    src = sample.make_sample(tmp_path / "s.mp4")
    r = pipeline.run(src, tmp_path / "out", settings=settings.Settings(), gateway=settings.Gateway(),
                     provider=FixtureProvider(), cache=Cache(tmp_path / "c"),
                     trim_start=5.0, trim_end=10.5)
    assert r["meta"]["trim"]["source_frame_offset"] == 150
    # Shots 3 (hands up 6.6 s) and 4 (9.1 s) are in the window; times stay in source seconds.
    assert [s["result"] for s in r["shots"]] == ["make", "unknown"]
    assert r["shots"][0]["hands_up_s"] == pytest.approx(6.6, abs=0.15)
    assert r["shots"][0]["source_frame_release"] == round(r["shots"][0]["release_s"] * 30)


def test_recorded_pose_reply_parses():
    payload = json.loads((FIX / "basketball_pose_reply.json").read_text())
    frames = pose.unwrap(payload)
    assert [f["index"] for f in frames] == list(range(5))
    kp = pose.keypoints(frames, 5, 1280, 720)
    assert kp.shape == (5, 17, 2) and kp[0, 0, 0] > 1   # scaled to pixels
    with pytest.raises(RuntimeError):
        pose.unwrap({"content": {"object": "img.segment.masks"}})


def test_recorded_hoop_reply_gives_rim():
    payload = json.loads((FIX / "basketball_hoop_reply.json").read_text())
    h = hoop.locate(payload, frame=0, width=1280, height=720)
    assert h.cx == pytest.approx(synthetic.RIM_CX, abs=2)
    assert h.cy == pytest.approx(synthetic.RIM_CY, abs=3)
    assert h.width == pytest.approx(synthetic.RIM_W, abs=3)
    assert h.net_bottom == pytest.approx(synthetic.NET_BOTTOM, abs=3)


def test_recorded_track_reply_offsets_and_filters():
    payload = json.loads((FIX / "basketball_track_reply.json").read_text())
    det = ball.detections(payload, frame_offset=100, width=1280, height=720, min_score=0.5)
    assert min(det) >= 100   # window-relative frame ids moved to clip frames
    t = ball.select(det, 300, [(100, 128)], size_tolerance=0.35)
    assert t.diameter == pytest.approx(synthetic.BALL_D, rel=0.05)
    # The net-grown boxes (1.9x) never become the ball.
    assert all(abs(d - synthetic.BALL_D) < 3 for d in t.d[t.found])

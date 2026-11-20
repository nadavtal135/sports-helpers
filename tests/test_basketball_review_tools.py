"""Shots that leave the top of the frame, gaps in flight, the 20.2 s night-court
candidate, the ball-detection export and contact-sheet tool, and the owner's
make/miss/not-a-shot review. All synthetic, no network."""

import csv
import io
import time
import zipfile

import numpy as np
import pytest

from basketball import diagnose, pipeline, review, sample, settings
from basketball.src import ball
from basketball.src.cache import Cache
from basketball.src.hoop import Hoop
from basketball.src.providers.fixture import FixtureProvider
from basketball.src.shots import Shot, attempt_check, judge

FPS = 30.0
NIGHT = Hoop(cx=370.0, cy=100.0, left=350.0, right=390.0, top=94.0, net_bottom=135.0,
             score=0.9, frame=0)
G = 9.81 * 40 / 0.49 / FPS ** 2          # gravity in px/frame² at the rim's distance


def _judge(points, *, release=600, n=900):
    x, y, d = (np.full(n, np.nan) for _ in range(3))
    for f, (px, py) in points.items():
        x[f], y[f], d[f] = px, py, 19.5
    shot = Shot(number=1, start=release - 5, end=release, release=release, set_frame=release - 5,
                elbow_set=np.nan, elbow_release=np.nan)
    judge(shot, x, y, d, NIGHT, FPS, until=n, inner_fraction=0.8, net_min_seconds=0.15,
          net_side_tolerance=0.35, rebound_margin=0.15, max_gap=15, evidence_gap=9,
          min_flight=24, ball_to_rim=0.239 / 0.49, front_size=1.2)
    return shot


def _arc(frames, *, x0=160.0, y0=150.0, to=(370.0, 104.0), start=600, visible=lambda f, y: y >= 0):
    vx = (to[0] - x0) / frames
    vy = (to[1] - y0 - 0.5 * G * frames * frames) / frames
    pts = {}
    for k in range(frames + 1):
        y = y0 + vy * k + 0.5 * G * k * k
        if visible(start + k, y):
            pts[start + k] = (x0 + vx * k, y)
    return pts


def _through_net(start):
    pts = {start + k: (370.0, 104 + 31 * k / 9) for k in range(1, 10)}
    pts[start + 10] = (371.0, 140.0)
    return pts


# ── task 4: a shot whose ball leaves the top of the frame ─────────────────────

def test_high_arc_leaving_the_top_of_the_frame_is_judged_a_make():
    pts = _arc(39)                                       # a 1.3 s lob: apex well above y = 0
    assert any(f not in pts for f in range(600, 640))     # it really was out of frame
    shot = _judge({**pts, **_through_net(639)})
    assert shot.made is True and shot.reason == "through the net"


def test_ball_out_of_the_top_then_lost_after_the_rim_is_unknown_not_miss():
    shot = _judge(_arc(39))
    assert shot.made is None


def test_ball_leaving_the_top_and_never_coming_back_is_unknown_not_miss():
    pts = {f: xy for f, xy in _arc(39).items() if f < 612}   # last seen rising at the top edge
    shot = _judge(pts)
    assert shot.made is None and shot.reason != "never above the rim"


def test_out_of_frame_gap_survives_ball_selection():
    pts = {**_arc(39), **_through_net(639)}
    raw = {f: [(x, y, 19.5, 19.5, 0.9)] for f, (x, y) in pts.items()}
    t = ball.select(raw, 900, [(580, 128)], size_tolerance=0.35, min_score=0.5, blur_rescue=True)
    kept = {f for f in range(900) if np.isfinite(t.x[f])}
    assert kept == set(pts)                              # no point dropped, nothing invented


# ── the ball lost for 0.3-0.5 s mid-flight, then found again ─────────────────

@pytest.mark.parametrize("lost", [9, 12, 15])
def test_ball_lost_mid_flight_and_found_again_is_still_judged(lost):
    apex = 12                                            # frames after release
    gap = range(600 + apex - lost // 2, 600 + apex - lost // 2 + lost)
    pts = _arc(30, visible=lambda f, y: f not in gap)
    shot = _judge({**pts, **_through_net(630)})
    assert shot.made is True


# ── the 20.2 s night-court candidate stays an attempt ─────────────────────────

NIGHT_202 = {19.4: (271, 132), 19.7: (235, 321), 19.8: (227, 324), 20.1: (199, 189),
             20.2: (191, 165), 20.3: (181, 150), 20.4: (166, 147), 20.5: (160, 155),
             20.9: (208, 251), 21.0: (228, 247), 21.1: (248, 238), 21.4: (295, 218)}


def _night_202():
    n = 1400
    x, y = np.full(n, np.nan), np.full(n, np.nan)
    for t, (px, py) in NIGHT_202.items():
        f = int(round(t * FPS))
        x[f], y[f] = px, py
    return x, y


def test_night_202_stays_an_attempt():
    assert settings.Settings().shot_attempt_check is False      # nothing is ever dropped by default
    x, y = _night_202()
    f = int(round(20.233 * FPS))
    shot = Shot(number=3, start=f, end=f + 6, release=f, set_frame=f, elbow_set=np.nan,
                elbow_release=np.nan, release_method="pose")
    s = settings.Settings()
    action, _ = attempt_check(shot, x, y, NIGHT, FPS, window_s=s.attempt_window_seconds,
                              min_obs=s.attempt_min_observations, recede_rims=s.attempt_recede_rims,
                              early_s=s.attempt_early_seconds, gap_s=s.evidence_gap_seconds)
    assert action != "drop"                                     # even with the check on
    judged = _judge({int(round(t * FPS)): xy for t, xy in NIGHT_202.items()}, release=f, n=1400)
    assert judged.made is None                                  # no evidence for make or miss


# ── ball_detections.csv and the contact-sheet tool ────────────────────────────

@pytest.fixture(scope="module")
def sample_job(tmp_path_factory):
    d = tmp_path_factory.mktemp("job")
    clip = sample.make_sample(d / "clip.mp4")
    r = pipeline.run(clip, d / "out", settings=settings.Settings(), gateway=settings.Gateway(),
                     provider=FixtureProvider(), cache=Cache(d / "cache"))
    return clip, d / "out", r


def test_every_ball_box_is_exported(sample_job):
    _, out, r = sample_job
    assert "ball_detections.csv" in r["files"]
    rows = list(csv.DictReader((out / "ball_detections.csv").open()))
    assert rows and {"score", "score_band", "selected", "window_start_frame"} <= set(rows[0])
    assert any(row["selected"] == "True" for row in rows)
    # The net-grown boxes (score 0.6, wrong size) are exported too, though never selected.
    assert any(row["selected"] == "False" and float(row["w_px"]) > 50 for row in rows)


def test_contact_sheet_renders_frames_with_detections(sample_job, tmp_path):
    clip, out, _ = sample_job
    res = diagnose.diagnose(out, clip, start=10.0, end=10.5, scale=2, step=3,
                            out=tmp_path / "sheet.png", frames_dir=tmp_path / "frames")
    assert res["frames"] == 6 and res["has_detections"]
    assert (tmp_path / "sheet.png").is_file() and len(list((tmp_path / "frames").glob("*.png"))) == 6
    import cv2
    tile = cv2.imread(str(next((tmp_path / "frames").glob("*.png"))))
    assert tile.shape[1] == 1280 * 2                              # rendered at 2x


# ── the owner's own calls ─────────────────────────────────────────────────────

def test_review_overrides_results_and_keeps_the_auto_call(tmp_path):
    result = {"stats": {"attempts": 3}, "shots": [
        {"shot": 1, "result": "make", "reason": "through the net", "release_s": 6.2},
        {"shot": 2, "result": "unknown", "reason": "ball lost", "release_s": 20.2},
        {"shot": 3, "result": "miss", "reason": "off the rim", "release_s": 30.6}]}
    with pytest.raises(ValueError):
        review.set_mark(tmp_path, 9, "make", [1, 2, 3])
    with pytest.raises(ValueError):
        review.set_mark(tmp_path, 2, "swish", [1, 2, 3])
    review.set_mark(tmp_path, 2, "miss", [1, 2, 3])
    assert review.set_mark(tmp_path, 3, "not_a_shot", [1, 2, 3]) == {"2": "miss", "3": "not_a_shot"}
    out = review.apply(result, review.load(tmp_path))
    s2, s3 = out["shots"][1], out["shots"][2]
    assert (s2["result"], s2["auto_result"], s2["reviewed"]) == ("miss", "unknown", "miss")
    assert (s3["result"], s3["auto_result"]) == ("not_a_shot", "miss")
    st = out["stats"]
    assert (st["attempts"], st["makes"], st["misses"], st["unknown"], st["judged"]) == (2, 1, 1, 0, 2)
    assert st["fg_pct"] == 0.5 and st["reviewed"] == 2 and out["auto_stats"] == {"attempts": 3}
    assert result["shots"][1]["result"] == "unknown"              # the input is not changed
    review.set_mark(tmp_path, 3, None, [1, 2, 3])                 # clearing a mark
    assert review.load(tmp_path) == {"2": "miss"}
    name = review.write_csv(tmp_path, out)
    rows = list(csv.DictReader((tmp_path / name).open()))
    assert rows[1]["auto_result"] == "unknown" and rows[1]["final_result"] == "miss"


def test_review_through_the_web_api(tmp_path, monkeypatch):
    monkeypatch.setenv("SPORTS_HELPERS_DATA", str(tmp_path))
    monkeypatch.setenv("SPORTS_HELPERS_MODE", "fixture")
    from fastapi.testclient import TestClient

    from webapp import server
    with TestClient(server.app) as client:
        up = client.post("/api/samples/basketball").json()
        job = client.post("/api/jobs", json={"upload_id": up["id"], "sport": "basketball"}).json()
        for _ in range(360):
            st = client.get(f"/api/jobs/{job['id']}").json()
            if st["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.5)
        assert st["status"] == "completed", st
        before = client.get(f"/api/jobs/{job['id']}/result").json()
        assert before["stats"]["unknown"] == 1
        unknown = next(s for s in before["shots"] if s["result"] == "unknown")
        res = client.post(f"/api/jobs/{job['id']}/shot-review",
                          json={"shot": unknown["shot"], "outcome": "make"})
        assert res.status_code == 200
        after = res.json()
        shot = next(s for s in after["shots"] if s["shot"] == unknown["shot"])
        assert (shot["result"], shot["auto_result"], shot["reviewed"]) == ("make", "unknown", "make")
        assert after["stats"]["makes"] == before["stats"]["makes"] + 1
        assert after["stats"]["unknown"] == 0
        names = [f["name"] for f in after["files"]]
        assert "shots_reviewed.csv" in names
        csv_text = client.get(f"/api/jobs/{job['id']}/files/shots_reviewed.csv").text
        assert "auto_result" in csv_text and ",unknown," in csv_text
        z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job['id']}/bundle.zip").content))
        assert "shots_reviewed.csv" in z.namelist()
        assert client.post(f"/api/jobs/{job['id']}/shot-review",
                           json={"shot": 99, "outcome": "make"}).status_code == 400
        assert client.post(f"/api/jobs/{job['id']}/shot-review",
                           json={"shot": 1, "outcome": "swish"}).status_code == 400
        # Clearing restores the automatic call.
        cleared = client.post(f"/api/jobs/{job['id']}/shot-review",
                              json={"shot": unknown["shot"], "outcome": None}).json()
        assert next(s for s in cleared["shots"] if s["shot"] == unknown["shot"])["result"] == "unknown"

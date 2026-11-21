"""Upload-to-results through the HTTP API in fixture mode (real worker processes)."""

import csv
import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPORTS_HELPERS_DATA", str(tmp_path))
    monkeypatch.setenv("SPORTS_HELPERS_MODE", "fixture")
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    from webapp import server
    with TestClient(server.app) as c:
        yield c


def wait(client, job_id, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_config_reports_fixture_mode_without_secrets(client):
    c = client.get("/api/config").json()
    assert c["mode"] == "fixture" and c["key_configured"] is False and c["ffmpeg"]["ok"]
    assert "VLMRUN_API_KEY" not in str(c.get("key", ""))


@pytest.mark.parametrize("sport", ["basketball", "running", "squat"])
def test_sample_upload_to_results(client, sport):
    up = client.post(f"/api/samples/{sport}").json()
    assert up["sample_of"] == sport and "source" not in up
    assert client.get(f"/api/uploads/{up['id']}/preview").status_code == 200
    trim = {"trim_start": 1.5, "trim_end": 11.5} if sport == "squat" else {"trim_start": 0.5}
    job = client.post("/api/jobs", json={"upload_id": up["id"], "sport": sport, **trim}).json()
    assert job["provider"] == "fixture"
    done = wait(client, job["id"])
    assert done["status"] == "completed", done
    r = client.get(f"/api/jobs/{job['id']}/result").json()
    assert r["meta"]["simulated"] is True and r["meta"]["provider"] == "fixture"
    assert r["meta"]["trim"]["start_s"] == trim["trim_start"]
    names = [f["name"] for f in r["files"]]
    assert any(n.endswith(".mp4") for n in names) and any(n.endswith(".json") for n in names)
    for f in r["files"]:
        res = client.get(f["url"])
        assert res.status_code == 200 and len(res.content) > 0
    if sport == "basketball":
        assert r["stats"]["unknown"] >= 1 and r["stats"]["fg_denominator"]
    elif sport == "running":
        assert abs(r["summary"]["cadence_spm"] - 172) < 1
    else:
        assert set(names) == {"squat_annotated.mp4", "squat_clean.mp4", "reps.csv", "reps.json"}
        assert r["summary"]["completed_reps"] == 2
        assert r["summary"]["incomplete_attempts"] == r["summary"]["uncertain_attempts"] == 1
        assert [a["status"] for a in r["reps"]] == ["incomplete", "completed", "completed", "uncertain"]
        assert [a["depth_2d"] for a in r["reps"]] == ["below", "roughly level", "above", "unknown"]
        assert r["meta"]["trim"]["source_frame_offset"] == 45
        assert r["meta"]["analysis_video"]["n_frames"] == 300
        base = f"/api/jobs/{job['id']}"
        assert client.get(f"{base}/files/reps.json").json()["reps"] == r["reps"]
        bundle = client.get(f"{base}/bundle.zip")
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert set(archive.namelist()) == set(names)
            rows = list(csv.DictReader(archive.read("reps.csv").decode().splitlines()))
        assert [row["rep"] for row in rows] == ["", "1", "2", ""]
        assert rows[0]["start_s"] == "" and float(rows[0]["bottom_s"]) == pytest.approx(1.8)
        assert rows[-1]["bottom_s"] == rows[-1]["descent_s"] == ""
        for row, rep in zip(rows, r["reps"], strict=True):
            for name in ("start", "bottom", "end"):
                seconds = rep[f"{name}_s"]
                if seconds is not None:
                    assert float(row[f"{name}_s"]) == seconds
                    assert int(row[f"source_frame_{name}"]) == round(seconds * 30)
            if rep["status"] == "completed":
                assert float(row["descent_s"]) == pytest.approx(rep["bottom_s"] - rep["start_s"], abs=0.001)
                assert float(row["ascent_s"]) == pytest.approx(rep["end_s"] - rep["bottom_s"], abs=0.001)


def test_fixture_mode_refuses_real_uploads(client, tmp_path):
    from running.sample import make_sample
    import subprocess
    src = make_sample(tmp_path / "x.mp4")
    other = tmp_path / "other.mp4"   # same picture, different bytes: not the sample
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-t", "4", "-crf", "35",
                    str(other)], check=True)
    with other.open("rb") as fh:
        up = client.post("/api/uploads", files={"file": ("mine.mp4", fh, "video/mp4")}).json()
    assert up["sample_of"] is None
    for sport in ("running", "squat"):
        res = client.post("/api/jobs", json={"upload_id": up["id"], "sport": sport})
        assert res.status_code == 403 and "sample" in res.json()["detail"]


def test_bad_inputs(client):
    res = client.post("/api/uploads", files={"file": ("x.mp4", b"not a video", "video/mp4")})
    assert res.status_code == 400 and "/" not in res.json()["detail"]   # no server paths leak
    up = client.post("/api/samples/running").json()
    assert client.post("/api/jobs", json={"upload_id": up["id"], "sport": "basketball"}).status_code == 400
    assert client.post("/api/jobs", json={"upload_id": up["id"], "sport": "squat"}).status_code == 400
    assert client.post("/api/jobs", json={"upload_id": up["id"], "sport": "running",
                                          "trim_start": 9.5}).status_code == 400
    assert client.post("/api/jobs", json={"upload_id": up["id"], "sport": "running",
                                          "settings": {"foot_reference": "x"}}).status_code == 400
    assert client.post("/api/jobs", json={"upload_id": "nope", "sport": "running"}).status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


@pytest.mark.parametrize("sport,csv_name", [("basketball", "shots.csv"), ("squat", "reps.csv")])
def test_cancel_deletes_partial_outputs(client, sport, csv_name):
    up = client.post(f"/api/samples/{sport}").json()
    job = client.post("/api/jobs", json={"upload_id": up["id"], "sport": sport}).json()
    time.sleep(1.5)
    client.post(f"/api/jobs/{job['id']}/cancel")
    done = wait(client, job["id"])
    assert done["status"] == "cancelled"
    assert client.get(f"/api/jobs/{job['id']}/result").status_code == 409
    assert client.get(f"/api/jobs/{job['id']}/files/{csv_name}").status_code == 404


def test_concurrent_jobs_are_isolated(client):
    ups = [client.post(f"/api/samples/{s}").json() for s in ("running", "running")]
    jobs = [client.post("/api/jobs", json={"upload_id": u["id"], "sport": "running",
                                           "trim_start": t}).json()
            for u, t in zip(ups, (0.0, 2.0))]
    results = []
    for j in jobs:
        assert wait(client, j["id"])["status"] == "completed"
        results.append(client.get(f"/api/jobs/{j['id']}/result").json())
    assert results[0]["meta"]["trim"]["start_s"] == 0.0 and results[1]["meta"]["trim"]["start_s"] == 2.0
    assert results[0]["steps"][0]["time"] != results[1]["steps"][0]["time"]


def test_bundle_zip_and_restart_persistence(client, tmp_path):
    from webapp import jobs as jobs_mod
    up = client.post("/api/samples/running").json()
    job = client.post("/api/jobs", json={"upload_id": up["id"], "sport": "running"}).json()
    assert wait(client, job["id"])["status"] == "completed"
    res = client.get(f"/api/jobs/{job['id']}/bundle.zip")
    assert res.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(res.content)).namelist()
    assert "running_annotated.mp4" in names and "gait.json" in names
    # A new manager (= a server restart) still knows the finished job.
    fresh = jobs_mod.JobManager()
    assert fresh.get(job["id"]).status == "completed"
    assert fresh.result(job["id"])["summary"]["cadence_spm"]


def test_job_interrupted_by_restart_is_reported_failed(client):
    import json

    from webapp import jobs as jobs_mod
    m = jobs_mod.JobManager()
    folder = m.jobs_dir() / ("a" * 32)
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps(
        {"id": "a" * 32, "sport": "running", "upload_id": "x", "provider": "fixture",
         "status": "running"}))
    job = jobs_mod.JobManager().get("a" * 32)
    assert job.status == "failed" and "server stopped" in job.error

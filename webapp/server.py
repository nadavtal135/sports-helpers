"""HTTP API and static UI.

    python -m webapp.server            # http://127.0.0.1:8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import jobs

STATIC = Path(__file__).resolve().parent / "static"
manager = jobs.JobManager()


@asynccontextmanager
async def lifespan(_app):
    jobs.sweep_old()
    yield
    manager.shutdown()


app = FastAPI(title="Sports Helpers", lifespan=lifespan)


def _settings_schema():
    return {sport: {"fields": cfg.USER_SETTINGS, "guidance": cfg.GUIDANCE}
            for sport, cfg in jobs.SPORT_SETTINGS.items()}


@app.get("/api/config")
def config():
    from basketball.src import video

    status = video.ffmpeg_status()
    return {"mode": jobs.mode(), "key_configured": bool(jobs.api_key()),
            "provider": "VLM Run Gateway (gateway.vlm.run)",
            "ffmpeg": {"ok": status["ok"], "version": (status["ffmpeg"] or {}).get("version")},
            "max_upload_mb": jobs.MAX_UPLOAD_BYTES // 2**20,
            "max_live_seconds": jobs.MAX_SECONDS_LIVE,
            "sports": _settings_schema()}


@app.post("/api/uploads")
async def upload(file: UploadFile = File(...)):
    try:
        record = await run_in_threadpool(jobs.new_upload, file.filename or "clip.mp4", file.file)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(400, "Could not read this file as a video. Try an MP4 or MOV "
                                 "straight from the camera.") from exc
    return _public_upload(record)


@app.post("/api/samples/{sport}")
async def use_sample(sport: str):
    if sport not in jobs.STAGES:
        raise HTTPException(404, "unknown sport")
    path, _ = await run_in_threadpool(jobs.sample, sport)
    with path.open("rb") as fh:
        record = await run_in_threadpool(jobs.new_upload, path.name, fh)
    return _public_upload(record)


def _public_upload(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "source"}


@app.get("/api/uploads/{upload_id}/preview")
def preview(upload_id: str):
    record = jobs.get_upload(upload_id)
    if record is None:
        raise HTTPException(404, "upload not found")
    return FileResponse(Path(record["source"]).parent / "preview.mp4", media_type="video/mp4")


@app.post("/api/jobs")
async def create_job(request: Request):
    body = await request.json()
    record = jobs.get_upload(str(body.get("upload_id", "")))
    if record is None:
        raise HTTPException(404, "Upload not found; upload the clip again.")
    try:
        job = manager.create(sport=body.get("sport"), upload=record,
                             settings=body.get("settings") or {},
                             trim_start=body.get("trim_start") or 0.0, trim_end=body.get("trim_end"),
                             consent=bool(body.get("consent")))
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return job.public()


def _job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job(job_id).public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    _job(job_id)
    return manager.cancel(job_id).public()


@app.get("/api/jobs/{job_id}/result")
def result(job_id: str):
    job = _job(job_id)
    data = manager.result(job_id) if job.status == "completed" else None
    if data is None:
        raise HTTPException(409, f"job is {job.status}")
    data["files"] = [{"name": n, "url": f"/api/jobs/{job_id}/files/{n}",
                      "bytes": (manager.folder(job_id) / "output" / n).stat().st_size}
                     for n in data["files"] if (manager.folder(job_id) / "output" / n).is_file()]
    return JSONResponse(data)


@app.get("/api/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str, inline: bool = False):
    job = _job(job_id)
    data = manager.result(job_id) if job.status == "completed" else None
    if data is None or name not in data["files"]:
        raise HTTPException(404, "file not found")
    path = manager.folder(job_id) / "output" / name
    if inline:
        return FileResponse(path)
    return FileResponse(path, filename=name)


@app.get("/api/jobs/{job_id}/bundle.zip")
def bundle(job_id: str):
    """Every output file of a completed job, in one zip (built once, then reused)."""
    job = _job(job_id)
    data = manager.result(job_id) if job.status == "completed" else None
    if data is None:
        raise HTTPException(404, "no results to download")
    out = manager.folder(job_id) / "output"
    path = manager.folder(job_id) / f"{job.sport}_analysis.zip"
    if not path.is_file():
        import zipfile

        tmp = path.with_suffix(".partial")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in data["files"]:
                if (out / name).is_file():
                    # Video is already compressed; storing it is much faster.
                    zf.write(out / name, name, compress_type=zipfile.ZIP_STORED
                             if name.endswith(".mp4") else zipfile.ZIP_DEFLATED)
        tmp.replace(path)
    return FileResponse(path, filename=path.name)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main():
    import uvicorn

    from basketball.src import video

    if not video.ffmpeg_status()["ok"]:
        print("WARNING: ffmpeg/ffprobe not found on PATH; uploads and analysis will fail.")
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"Sports Helpers: http://{host}:{port}  (mode: {jobs.mode()})")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

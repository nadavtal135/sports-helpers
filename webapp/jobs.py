"""Uploads, samples and analysis jobs: storage, validation, process management."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
STAGES = {
    "basketball": [("convert", "Convert video", 0.08), ("pose", "Pose (ViTPose)", 0.22),
                   ("hoop", "Hoop (SAM 3.1)", 0.05), ("ball", "Ball tracking (SAM 3.1)", 0.30),
                   ("analyze", "Judge shots", 0.03), ("render", "Render video", 0.29),
                   ("export", "Charts & exports", 0.03)],
    "running": [("convert", "Convert video", 0.10), ("pose", "Pose (ViTPose)", 0.40),
                ("analyze", "Foot strikes & cadence", 0.05), ("render", "Render video", 0.40),
                ("export", "Charts & exports", 0.05)],
}
MAX_UPLOAD_BYTES = int(os.getenv("SPORTS_HELPERS_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
MAX_SECONDS_LIVE = float(os.getenv("SPORTS_HELPERS_MAX_SECONDS", "180"))
MAX_CONCURRENT = int(os.getenv("SPORTS_HELPERS_MAX_JOBS", "2"))
JOB_TIMEOUT = float(os.getenv("SPORTS_HELPERS_JOB_TIMEOUT", "3600"))


def data_dir() -> Path:
    path = Path(os.getenv("SPORTS_HELPERS_DATA", REPO_DIR / "data")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def api_key() -> str:
    """The server's key: the environment, else ``.env`` at the repo root. Never sent to the browser."""
    key = os.getenv("VLMRUN_API_KEY", "").strip()
    if key:
        return key
    env = REPO_DIR / ".env"
    if env.is_file():
        try:
            from dotenv import dotenv_values

            return (dotenv_values(env).get("VLMRUN_API_KEY") or "").strip()
        except ImportError:
            pass
    return ""


def mode() -> str:
    """"fixture" or "live". SPORTS_HELPERS_MODE forces one; otherwise live iff a key is set."""
    forced = os.getenv("SPORTS_HELPERS_MODE", "").lower()
    if forced in ("fixture", "live"):
        return forced
    return "live" if api_key() else "fixture"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


# ── samples ──────────────────────────────────────────────────────────────────

_sample_lock = threading.Lock()


def sample(sport: str) -> tuple[Path, str]:
    """The synthetic fixture clip for *sport* (generated once) and its sha256."""
    with _sample_lock:
        path = data_dir() / "samples" / f"{sport}_sample.mp4"
        digest = path.with_suffix(".sha256")
        if not path.is_file() or not digest.is_file():
            if sport == "basketball":
                from basketball.sample import make_sample
            else:
                from running.sample import make_sample
            make_sample(path)
            digest.write_text(_sha256(path))
        return path, digest.read_text().strip()


def sample_hashes() -> dict[str, str]:
    out = {}
    for sport in STAGES:
        digest = data_dir() / "samples" / f"{sport}_sample.sha256"
        if digest.is_file():
            out[sport] = digest.read_text().strip()
    return out


# ── uploads ──────────────────────────────────────────────────────────────────

def uploads_dir() -> Path:
    return data_dir() / "uploads"


def new_upload(filename: str, stream, *, sample_of: str | None = None) -> dict:
    from basketball.src import video

    upload_id = uuid.uuid4().hex
    folder = uploads_dir() / upload_id
    folder.mkdir(parents=True)
    suffix = Path(filename).suffix.lower()[:8] or ".mp4"
    dst = folder / f"source{suffix}"
    size = 0
    h = hashlib.sha256()
    try:
        with dst.open("wb") as fh:
            while block := stream.read(1 << 20):
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError(f"The file is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
                h.update(block)
                fh.write(block)
        meta = video.probe(dst)
        if meta["duration"] <= 0 or not meta["width"]:
            raise ValueError("Could not read a duration or frame size from this file.")
        video.preview(dst, folder / "preview.mp4")
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    digest = h.hexdigest()
    samples = sample_hashes()
    record = {"id": upload_id, "filename": Path(filename).name[:120], "size": size, "sha256": digest,
              "source": str(dst), **{k: meta[k] for k in ("duration", "fps", "width", "height", "codec")},
              "sample_of": next((s for s, d in samples.items() if d == digest), None)}
    (folder / "upload.json").write_text(json.dumps(record))
    return record


def get_upload(upload_id: str) -> dict | None:
    if not upload_id.isalnum():
        return None
    path = uploads_dir() / upload_id / "upload.json"
    return json.loads(path.read_text()) if path.is_file() else None


# ── jobs ─────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    sport: str
    upload_id: str
    provider: str
    status: str = "queued"          # queued, running, completed, failed, cancelled
    stage: str | None = None
    fraction: float = 0.0
    message: str = "Waiting for a free worker"
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    trim_start: float = 0.0
    trim_end: float | None = None

    def public(self) -> dict:
        d = asdict(self)
        d["progress"] = self.progress()
        d["stages"] = [{"id": s, "label": label} for s, label, _ in STAGES[self.sport]]
        return d

    def progress(self) -> float:
        if self.status == "completed":
            return 1.0
        done = 0.0
        for stage, _, weight in STAGES[self.sport]:
            if stage == self.stage:
                return round(done + weight * self.fraction, 3)
            done += weight
        return 0.0


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.cancelled: set[str] = set()
        self.lock = threading.Lock()
        self.slots = threading.Semaphore(MAX_CONCURRENT)
        self.load()

    def load(self) -> None:
        """Jobs from earlier server runs, from each job folder's status.json. A job
        that was still queued or running when the server stopped is marked failed:
        its worker died with the server."""
        root = self.jobs_dir()
        if not root.is_dir():
            return
        for folder in root.iterdir():
            path = folder / "status.json"
            try:
                job = Job(**json.loads(path.read_text()))
            except (OSError, ValueError, TypeError):
                continue
            if job.sport not in STAGES:
                continue
            if job.status in ("queued", "running"):
                job.status, job.message = "failed", "Analysis failed"
                job.error = "The server stopped while this analysis was running. Start it again."
                job.finished = job.finished or time.time()
                self._save(job)
            self.jobs[job.id] = job

    def _save(self, job: Job) -> None:
        folder = self.folder(job.id)
        if not folder.is_dir():
            return
        tmp = folder / "status.partial.json"
        tmp.write_text(json.dumps(asdict(job)))
        tmp.replace(folder / "status.json")

    def jobs_dir(self) -> Path:
        return data_dir() / "jobs"

    def create(self, *, sport: str, upload: dict, settings: dict, trim_start: float,
               trim_end: float | None, consent: bool) -> Job:
        if sport not in STAGES:
            raise ValueError("Choose Basketball or Running.")
        # Validate settings early, in this process, so the user sees the error now.
        if sport == "basketball":
            from basketball.settings import from_user
        else:
            from running.settings import from_user
        from_user(settings)

        duration = upload["duration"]
        trim_start = max(0.0, float(trim_start or 0.0))
        trim_end = None if trim_end in (None, "") else float(trim_end)
        if trim_end is not None and trim_end >= duration - 1e-3:
            trim_end = None
        if trim_start >= (trim_end or duration) - 2.0:
            raise ValueError("The trimmed clip must be at least 2 seconds long.")

        if upload.get("sample_of"):
            if upload["sample_of"] != sport:
                raise ValueError(f"This is the {upload['sample_of']} sample clip; choose "
                                 f"{upload['sample_of'].title()} to analyze it.")
            provider = "fixture"
        elif mode() == "fixture":
            raise PermissionError(
                "Fixture mode can only analyze the bundled synthetic sample clip. Its results are "
                "simulated and would say nothing about your video. Set VLMRUN_API_KEY on the "
                "server to analyze your own clips.")
        else:
            if not api_key():
                raise PermissionError("Live analysis needs VLMRUN_API_KEY set on the server.")
            if not consent:
                raise PermissionError("Confirm that the clip may be uploaded to the model provider.")
            if (trim_end or duration) - trim_start > MAX_SECONDS_LIVE:
                raise ValueError(f"Trim the clip to at most {MAX_SECONDS_LIVE:.0f} seconds for live "
                                 "analysis (model calls are billed per frame).")
            provider = "vlmrun"

        job = Job(id=uuid.uuid4().hex, sport=sport, upload_id=upload["id"], provider=provider,
                  trim_start=trim_start, trim_end=trim_end)
        folder = self.jobs_dir() / job.id
        folder.mkdir(parents=True)
        (folder / "job.json").write_text(json.dumps({
            "sport": sport, "source": upload["source"], "provider": provider,
            "settings": settings, "trim_start": trim_start, "trim_end": trim_end,
            "cache_dir": str(data_dir() / "cache"), "upload": {k: upload[k] for k in (
                "filename", "sha256", "duration", "fps", "width", "height")},
        }))
        with self.lock:
            self.jobs[job.id] = job
        self._save(job)
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def folder(self, job_id: str) -> Path:
        return self.jobs_dir() / job_id

    def result(self, job_id: str) -> dict | None:
        path = self.folder(job_id) / "result.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is None or job.status not in ("queued", "running"):
            return job
        self.cancelled.add(job_id)
        proc = self.procs.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if job.status == "queued":
            self._finish(job, "cancelled", "Cancelled before it started.")
        return job

    def _finish(self, job: Job, status: str, message: str, error: str | None = None) -> None:
        job.status, job.message, job.error, job.finished = status, message, error, time.time()
        self._save(job)
        out = self.folder(job.id) / "output"
        if status != "completed":
            shutil.rmtree(out, ignore_errors=True)   # nothing half-written is offered for download

    def _run(self, job: Job) -> None:
        with self.slots:
            if job.id in self.cancelled:
                if job.status == "queued":
                    self._finish(job, "cancelled", "Cancelled before it started.")
                return
            job.status, job.message = "running", "Starting"
            self._save(job)
            env = {k: v for k, v in os.environ.items() if k != "VLMRUN_API_KEY"}
            env.update({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
                        "PYTHONPATH": str(REPO_DIR)})
            key = api_key()
            if job.provider == "vlmrun":
                env["VLMRUN_API_KEY"] = key
            log = (self.folder(job.id) / "worker.log").open("w")
            proc = subprocess.Popen([sys.executable, "-m", "webapp.worker", str(self.folder(job.id))],
                                    stdout=subprocess.PIPE, stderr=log, text=True, cwd=str(REPO_DIR),
                                    env=env)
            self.procs[job.id] = proc
            timer = threading.Timer(JOB_TIMEOUT, proc.kill)
            timer.daemon = True
            timer.start()
            final = None
            try:
                for line in proc.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "progress":
                        job.stage, job.fraction = event["stage"], event["fraction"]
                        job.message = event.get("message", "")
                    else:
                        final = event
                proc.wait()
            finally:
                timer.cancel()
                log.close()
                self.procs.pop(job.id, None)
            if key:   # belt and braces: the worker never prints the key, but scrub its log anyway
                path = self.folder(job.id) / "worker.log"
                path.write_text(path.read_text().replace(key, "[redacted]"))
            if job.id in self.cancelled:
                self._finish(job, "cancelled", "Cancelled. Partial outputs were deleted.")
            elif final and final.get("type") == "done" and proc.returncode == 0:
                self._finish(job, "completed", "Analysis complete")
            elif final and final.get("type") == "error":
                msg = final.get("message", "Analysis failed")
                self._finish(job, "failed", "Analysis failed", msg.replace(key, "[redacted]") if key else msg)
            else:
                self._finish(job, "failed", "Analysis failed",
                             "The analysis process stopped unexpectedly (timeout or crash).")

    def shutdown(self) -> None:
        for proc in list(self.procs.values()):
            if proc.poll() is None:
                proc.kill()


def sweep_old(days: float = 7.0) -> None:
    """Delete uploads and jobs older than *days* (the cache is kept)."""
    cutoff = time.time() - days * 86400
    for root in (uploads_dir(), data_dir() / "jobs"):
        if not root.is_dir():
            continue
        for item in root.iterdir():
            try:
                if item.is_dir() and item.stat().st_mtime < cutoff:
                    shutil.rmtree(item, ignore_errors=True)
            except OSError:
                pass

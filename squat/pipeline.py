"""Squat upload to annotated H.264, per-attempt timing and 2D landmark estimates."""

import csv
import json
import tempfile
import time
from pathlib import Path

from running.src import video
from running.src.cache import Cache
from running.src.pose_clip import Cancelled, check_cancelled, prepare
from . import analysis, render
from .settings import PIPELINE_VERSION, Gateway, Settings

LANDMARKS = ("start", "bottom", "end")
COLUMNS = ("attempt", "rep", "status", "reason", "side", "review_start_s", "review_end_s",
           *(f"{name}_s" for name in LANDMARKS), *(f"source_frame_{name}" for name in LANDMARKS),
           "descent_s", "ascent_s", "peak_knee_flexion_projected_deg", "torso_lean_at_bottom_deg", "depth_2d")


def rep_rows(result: analysis.Analysis, fps: float, offset: int) -> list[dict]:
    def source_time(frame):
        return round((offset + frame) / fps, 4) if frame is not None else None

    def seconds(a, b):
        return round((b - a) / fps, 4) if a is not None and b is not None else None

    rows = []
    completed = 0
    for i, a in enumerate(result.attempts, start=1):
        completed += a.status == "completed"
        row = {"attempt": i, "rep": completed if a.status == "completed" else None,
               "status": a.status, "reason": a.reason, "side": result.side,
               "review_start_s": source_time(a.first), "review_end_s": source_time(a.last),
               "descent_s": seconds(a.start, a.bottom), "ascent_s": seconds(a.bottom, a.end),
               "peak_knee_flexion_projected_deg": a.peak_flexion,
               "torso_lean_at_bottom_deg": a.torso_lean, "depth_2d": a.depth}
        for name in LANDMARKS:
            frame = getattr(a, name)
            row[f"{name}_s"] = source_time(frame)
            row[f"source_frame_{name}"] = frame + offset if frame is not None else None
        rows.append(row)
    return rows


def run(source: Path, out_dir: Path, *, settings: Settings, gateway: Gateway, provider,
        cache: Cache, trim_start: float = 0.0, trim_end: float | None = None,
        progress=lambda *_: None, cancelled=lambda: False) -> dict:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = prepare(source, settings=settings, gateway=gateway, provider=provider, cache=cache,
                   trim_start=trim_start, trim_end=trim_end, subject="lifter",
                   progress=progress, cancelled=cancelled)
    info = clip.info
    progress("analyze", 0.0, "Finding squat reps")
    result = analysis.analyze(clip.frames, info.fps, info.n_frames, width=info.width, height=info.height,
                              settings=settings)
    check_cancelled(cancelled)
    progress("render", 0.0, "Rendering the annotated video")
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        raw = Path(tmp) / "overlay.mp4"
        render.render(info, result, raw, export_height=settings.export_height,
                      progress=lambda f: progress("render", f, "Rendering the annotated video"),
                      cancelled=cancelled)
        check_cancelled(cancelled)
        video.encode_h264(raw, out_dir / "squat_annotated.mp4", crf=settings.output_crf)

    check_cancelled(cancelled)
    progress("export", 0.0, "Writing rep exports")
    rows = rep_rows(result, info.fps, clip.offset)
    summary = {"completed_reps": sum(a.status == "completed" for a in result.attempts),
               "incomplete_attempts": sum(a.status == "incomplete" for a in result.attempts),
               "uncertain_attempts": sum(a.status == "uncertain" for a in result.attempts),
               "selected_side": result.side, "leg_coverage": result.coverage}
    meta = {**clip.meta, "sport": "squat", "pipeline_version": PIPELINE_VERSION,
            "seconds": round(time.perf_counter() - started, 2)}
    report = {"sport": "squat", "summary": summary, "reps": rows, "warnings": result.warnings, "meta": meta}
    with (out_dir / "reps.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    notes = {"times": "Landmarks and review bounds are in original clip seconds. Durations are elapsed seconds.",
             "depth_2d": "Visible hip keypoint height relative to the visible knee at the bottom, in the image plane.",
             "angles": "Projected angles in degrees. A straight knee has zero flexion. An upright torso has zero lean."}
    (out_dir / "reps.json").write_text(json.dumps({**report, "notes": notes}, indent=2, allow_nan=False))
    progress("export", 1.0, "Done")
    return {**report, "files": ["squat_annotated.mp4", "reps.csv", "reps.json"]}

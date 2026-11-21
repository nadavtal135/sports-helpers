"""Turn one uploaded jump set into a technique review and frame-aligned video."""

import csv
import json
from pathlib import Path

from running.src.pose_clip import Cancelled, check_cancelled, prepare
from running.src.side_view_video import export_videos, render
from . import analysis, review
from .foot_contact import ShoeContact
from .settings import PIPELINE_VERSION


def attempt_rows(result, fps, offset):
    rows = []
    for number, a in enumerate(result.attempts, 1):
        frames = {phase: None if frame is None else offset + frame for phase, frame in a.frames.items()}
        row = {"attempt": number, "status": "confirmed" if a.confirmed else "partial", "side": result.side,
               "first_frame": offset + a.first, "last_frame": offset + a.last, "frames": frames,
               "times_s": {k: round(v / fps, 4) if v is not None else None for k, v in frames.items()},
               "metrics": a.metrics, "step": a.step, "timing_evidence": a.methods,
               "visible_landing_bend": a.visible_landing_bend, "rebounds_into_next": a.rebounds_into_next,
               "issues": [{"phase": q["phase"], "source_frame": offset + q["frame"], "detail": q["detail"]} for q in a.issues]}
        row["cues"] = {phase: review.cue(row, phase) for phase, frame in frames.items() if frame is not None}
        rows.append(row)
    return rows


def run(source: Path, out_dir: Path, *, settings, gateway, provider, cache,
        trim_start=0.0, trim_end=None, progress=lambda *_: None, cancelled=lambda: False):
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = prepare(source, settings=settings, gateway=gateway, provider=provider, cache=cache,
                   trim_start=trim_start, trim_end=trim_end, subject="jumper", progress=progress, cancelled=cancelled)
    info = clip.info
    progress("analyze", 0, "Finding takeoffs and reviewing landings")
    shoe_contact = ShoeContact(info.path)
    try:
        result = analysis.analyze(clip.frames, info.fps, info.n_frames, width=info.width, height=info.height,
                                  settings=settings, verify_contact=shoe_contact)
    finally:
        shoe_contact.close()
    check_cancelled(cancelled)
    ends = {a.frames["contact"] for a in result.attempts if a.confirmed}

    def draw(raw):
        render(info, result.points, result.phases, result.side, ends, raw, export_height=settings.export_height,
               progress=lambda f: progress("render", f, "Rendering jump phases and pose"), cancelled=cancelled)

    progress("render", 0, "Rendering jump phases and pose")
    export_videos(info, out_dir, "standing_jump", draw, settings.output_crf, cancelled)
    progress("export", 0, "Saving jump observations and evidence frames")
    rows = attempt_rows(result, info.fps, clip.offset)
    report = {"sport": "standing_jump", "attempts": rows, "review": review.for_set(rows),
              "summary": {"confirmed_jumps": len(ends), "partial_attempts": len(rows) - len(ends), "side": result.side},
              "warnings": result.warnings,
              "meta": {**clip.meta, "sport": "standing_jump", "pipeline_version": PIPELINE_VERSION}}
    (out_dir / "jumps.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    fields = ("attempt", "status", "side", "step", *(f"{k}_frame" for k in analysis.LANDMARKS),
              *(f"{k}_s" for k in analysis.LANDMARKS), *analysis.METRICS)
    with (out_dir / "jumps.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in ("attempt", "status", "side", "step")}, **r["metrics"],
                             **{f"{k}_frame": v for k, v in r["frames"].items()},
                             **{f"{k}_s": v for k, v in r["times_s"].items()}})
    progress("export", 1, "Done")
    return {**report, "files": ["standing_jump_annotated.mp4", "standing_jump_clean.mp4", "jumps.csv", "jumps.json"]}

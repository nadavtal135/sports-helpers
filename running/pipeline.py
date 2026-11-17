"""The running pipeline: video -> poses -> gait -> annotated video, charts, exports.

Adapted from jeremyipark/vision-demos ``running/main.py`` (Apache-2.0, commit
c06cfb78ccd8e2c63629222695aac710fe4c199c). Modifications: a function with
explicit settings, provider and cache instead of a script over a mutated
``config``; content-hash cache keys; trimming with source-frame offsets in all
outputs; progress/cancel callbacks; user-facing warnings; no console UI and no
timing/metrics files.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .settings import PIPELINE_VERSION, Gateway, Settings
from .src import gait, pose, render, video
from .src.cache import Cache, file_sha256, key_of

POSE_CONTRACT = "flat-envelope-v1"


class Cancelled(Exception):
    pass


def _noop(*_a, **_k):
    return None


def run(source: Path, out_dir: Path, *, settings: Settings, gateway: Gateway, provider,
        cache: Cache, trim_start: float = 0.0, trim_end: float | None = None,
        progress=_noop, cancelled=lambda: False) -> dict:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp"
    tmp.mkdir(exist_ok=True)

    def check():
        if cancelled():
            raise Cancelled()

    progress("convert", 0.0, "Converting video")
    src = video.probe(source)
    fps_src = src["fps"] or 30.0
    duration = (trim_end - trim_start) if trim_end else None
    src_hash = file_sha256(source)
    conv_key = key_of(src=src_hash, start=round(trim_start, 3), duration=duration and round(duration, 3),
                      height=settings.inference_height, crf=settings.convert_crf, fps=fps_src, v=2)
    mp4 = cache.path("videos", conv_key, ".mp4")
    if not mp4.is_file():
        video.convert(source, mp4, target_height=settings.inference_height, start=trim_start,
                      duration=duration, crf=settings.convert_crf, fps=fps_src,
                      progress=lambda f: progress("convert", f, "Converting video"), cancel_check=cancelled)
    check()
    info = video.inspect(mp4)
    fps, n = info.fps, info.n_frames
    offset = int(round(trim_start * fps))
    if n < fps * 2:
        raise RuntimeError("The clip is shorter than 2 seconds after trimming; nothing to analyze.")
    video_hash = file_sha256(mp4)

    progress("pose", 0.0, "Estimating the runner's pose (ViTPose)")
    # Every frame, as the reference's EVERY_FRAME: detector stride 1, all frames billed.
    extra_body = {"method": "pose", "video_fps": fps, "precision": gateway.precision,
                  "video_max_frames": n}
    key = key_of(video=video_hash, provider=provider.name, kind="pose", model=gateway.pose_model,
                 extra_body=extra_body, contract=POSE_CONTRACT)
    hit = cache.get_json("pose", key)
    if hit is not None:
        payload, cost = hit["payload"], 0.0
    else:
        payload, usage = provider.pose(mp4, model=gateway.pose_model, extra_body=extra_body,
                                       ctx={"source_offset": offset, "n_frames": n, "fps": fps})
        cache.put_json("pose", key, {"payload": payload, "usage": usage})
        cost = float((usage or {}).get("cost") or 0)
    check()
    frames = pose.unwrap(payload)
    pose.dedupe(frames, max_persons=settings.max_persons, iou=settings.person_iou)
    progress("pose", 1.0, "Poses received")

    progress("analyze", 0.0, "Finding foot strikes")
    analysis = gait.analyze(frames, fps, n, settings=settings, aspect=info.width / info.height)
    notices = warnings_for(analysis, n, fps)
    check()

    progress("render", 0.0, "Rendering the annotated video")
    kp = pose.keypoints(frames, n, info.width, info.height)
    raw_mp4 = tmp / "_raw.mp4"
    render.render(info, kp=kp, analysis=analysis, dst=raw_mp4, settings=settings,
                  progress=lambda f: progress("render", f, "Rendering the annotated video"),
                  cancelled=cancelled)
    check()
    out_video = out_dir / "running_annotated.mp4"
    video.encode_h264(raw_mp4, out_video, crf=settings.output_crf)
    progress("export", 0.0, "Writing charts and exports")
    render.plot_clearance(analysis, out_dir / "clearance.png", offset=offset)

    meta = {
        "sport": "running", "pipeline_version": PIPELINE_VERSION,
        "provider": provider.name, "simulated": provider.simulated,
        "source": {**src, "sha256": src_hash},
        "trim": {"start_s": trim_start, "end_s": trim_end, "source_frame_offset": offset},
        "analysis_video": {"width": info.width, "height": info.height, "fps": fps, "n_frames": n},
        "model": gateway.pose_model, "extra_body": extra_body,
        "frames_returned": len(frames), "frames_with_person": sum(1 for f in frames if f["persons"]),
        "settings": settings.as_dict(), "signal": gait.signal_config(settings),
        "cost_usd": round(cost, 6), "seconds": round(time.perf_counter() - t_start, 2),
    }
    files = write_exports(analysis, out_dir, offset=offset, meta=meta, notices=notices,
                          video_seconds=info.duration)
    shutil.rmtree(tmp, ignore_errors=True)
    progress("export", 1.0, "Done")
    summary = analysis.summary()
    summary.pop("steps_detail", None)
    return {"sport": "running", "summary": summary, "warnings": notices,
            "files": ["running_annotated.mp4", "clearance.png", *files],
            "steps": [_step_row(s, offset, fps) for s in analysis.steps],
            "chart": chart_data(analysis, offset),
            "meta": meta}


def warnings_for(a: gait.GaitAnalysis, n: int, fps: float) -> list[str]:
    q = a.warnings
    out = []
    for foot in q["absent_feet"]:
        out.append(f"The {foot} leg was tracked on only {q['ankle_coverage'][foot]:.0%} of frames "
                   "and is left out. Film from the side with both ankles visible.")
    if a.cadence_confidence == "none":
        out.append("No complete same-foot stride was measured, so there is no cadence. "
                   "Use a longer clip of steady running.")
    elif a.cadence_confidence == "low":
        out.append(f"Cadence rests on only {a.stride_count} stride(s); treat it as low confidence.")
    gaps = sum(q["frames_in_long_gaps"][f] for f in a.present)
    if gaps:
        out.append(f"{gaps / fps:.1f}s of leg tracking was lost (gaps longer than "
                   f"{q['max_interp_gap_frames']} frames are not filled in). Strides across "
                   f"those gaps are not measured ({q['intervals_dropped_for_gaps']} dropped).")
    if q["untimed_strikes"]:
        out.append(f"{q['untimed_strikes']} stance(s) were cut off by a gap or the clip end: "
                   "counted, but not timed.")
    if q["consecutive_same_foot_strikes"]:
        out.append(f"{q['consecutive_same_foot_strikes']} strike(s) on the other foot were missed; "
                   "the affected step intervals are left out.")
    if q["implausible_stance_flexion_frames"]:
        out.append(f"{q['implausible_stance_flexion_frames']} frame(s) show a planted knee bent past "
                   "the plausible limit, often a left/right keypoint swap. Check the knee values.")
    if q["heading_disagrees_with_facing"]:
        out.append("Direction of travel disagrees with the way the body faces; the pose may be "
                   "flipped.")
    if a.count < 4:
        out.append("Very few foot strikes were detected. Is the runner side-on with both ankles visible?")
    return out


def _step_row(s, offset: int, fps: float) -> dict:
    d = s.as_dict()
    shift = offset / fps
    for k in ("time", "mid_stance", "peak_time"):
        if d.get(k) is not None and np.isfinite(d[k]):
            d[k] = round(d[k] + shift, 4)
    d["source_frame"] = s.frame + offset
    for k, v in list(d.items()):
        if isinstance(v, float) and not np.isfinite(v):
            d[k] = None
    return d


def chart_data(a: gait.GaitAnalysis, offset: int, max_points: int = 1500) -> dict:
    step = max(1, a.n_frames // max_points)
    shift = offset / a.fps

    def series(values):
        return [None if not np.isfinite(v) else round(float(v), 4) for v in values[::step]]

    return {"time": [round(t + shift, 3) for t in a.times[::step]],
            "clearance": {f: series(a.clearance[f]) for f in gait.FEET},
            "cadence": series(a.cadence_live),
            "contact_level": round(a.contact_level, 4)}


def write_exports(a: gait.GaitAnalysis, out_dir: Path, *, offset: int, meta: dict,
                  notices: list[str], video_seconds: float) -> list[str]:
    fps = a.fps
    with (out_dir / "steps.csv").open("w", newline="") as fh:
        rows = [_step_row(s, offset, fps) for s in a.steps]
        cols = ["number", "foot", "time", "source_frame", "mid_stance", "step_interval",
                "stride_seconds", "contact_seconds", "peak_clearance", "knee_flexion_deg",
                "knee_sample_offset_ms"]
        out = csv.writer(fh)
        out.writerow([{"time": "strike_s", "mid_stance": "mid_stance_s",
                       "knee_flexion_deg": "knee_flexion_projected_deg",
                       "contact_seconds": "contact_proxy_s",
                       "peak_clearance": "peak_clearance_leg_lengths"}.get(c, c) for c in cols])
        for r in rows:
            out.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
    with (out_dir / "frames.csv").open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["frame", "source_frame", "t_s", "clearance_left", "clearance_right",
                      "observed_left", "observed_right", "knee_left_projected_deg",
                      "knee_right_projected_deg", "cadence_spm"])

        def v(x, d=4):
            return "" if not np.isfinite(x) else round(float(x), d)

        for k in range(a.n_frames):
            out.writerow([k, k + offset, round((k + offset) / fps, 4),
                          v(a.clearance["left"][k]), v(a.clearance["right"][k]),
                          bool(a.observed["left"][k]), bool(a.observed["right"][k]),
                          v(a.knee_flexion["left"][k], 1), v(a.knee_flexion["right"][k], 1),
                          v(a.cadence_live[k], 1)])
    summary = a.summary()
    summary["steps_detail"] = [_step_row(s, offset, fps) for s in a.steps]
    (out_dir / "gait.json").write_text(json.dumps({
        **summary,
        "warnings": notices,
        "notes": {
            "cadence": "120 / mean same-foot stride seconds.",
            "knee": "Projected 2D knee flexion near foot strike; not an anatomical measurement.",
            "contact": "Time the ankle spends under the clearance threshold: a proxy for contact.",
            "not_measured": "COCO-17 has no heel or toe keypoints: no heel/forefoot strike "
                            "classification, pronation, or treadmill speed.",
            "times": "Seconds in the original uploaded clip.",
        },
        "leg_length_frame_heights": round(a.leg_length, 5),
        "frame_aspect": round(a.frame_aspect, 5),
        "contact_level_leg_lengths": round(a.contact_level, 5),
        "meta": meta,
        "series": a.series(),
    }, indent=1, allow_nan=False, default=lambda o: None))
    (out_dir / "summary.txt").write_text(gait.summary_text(
        a, video_seconds=video_seconds))
    return ["steps.csv", "frames.csv", "gait.json", "summary.txt"]

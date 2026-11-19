"""Prepare a trimmed CFR video and fetch COCO-17 poses for a single subject.

Conversion and gateway orchestration extracted from the Running pipeline
(adapted from jeremyipark/vision-demos, Apache-2.0).
"""

from dataclasses import dataclass
from pathlib import Path

from ..settings import Gateway, PoseVideoSettings
from . import pose, video
from .cache import Cache, file_sha256, key_of

POSE_CONTRACT = "flat-envelope-v1"


class Cancelled(Exception):
    pass


def check_cancelled(cancelled):
    if cancelled():
        raise Cancelled()


@dataclass
class PoseClip:
    info: video.VideoInfo
    frames: list[dict]
    offset: int
    meta: dict


def prepare(source: Path, *, settings: PoseVideoSettings, gateway: Gateway, provider,
            cache: Cache, trim_start: float, trim_end: float | None, subject: str,
            progress, cancelled) -> PoseClip:
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
    check_cancelled(cancelled)
    info = video.inspect(mp4)
    fps, n = info.fps, info.n_frames
    offset = int(round(trim_start * fps))
    if n < fps * 2:
        raise RuntimeError("The clip is shorter than 2 seconds after trimming; nothing to analyze.")
    video_hash = file_sha256(mp4)

    progress("pose", 0.0, f"Estimating the {subject}'s pose (ViTPose)")
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
    check_cancelled(cancelled)
    frames = pose.unwrap(payload)
    pose.dedupe(frames, max_persons=settings.max_persons, iou=settings.person_iou)
    progress("pose", 1.0, "Poses received")

    meta = {
        "provider": provider.name, "simulated": provider.simulated,
        "source": {**src, "sha256": src_hash},
        "trim": {"start_s": trim_start, "end_s": trim_end, "source_frame_offset": offset},
        "analysis_video": {"width": info.width, "height": info.height, "fps": fps, "n_frames": n},
        "model": gateway.pose_model, "extra_body": extra_body,
        "frames_returned": len(frames), "frames_with_person": sum(1 for f in frames if f["persons"]),
        "settings": settings.as_dict(),
        "cost_usd": round(cost, 6),
    }
    return PoseClip(info=info, frames=frames, offset=offset, meta=meta)

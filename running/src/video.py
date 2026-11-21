"""FFmpeg/OpenCV video helpers: checks, probing, cached conversion, cutting, encoding.

Adapted from jeremyipark/vision-demos ``basketball/src/video.py`` (Apache-2.0,
commit c06cfb78ccd8e2c63629222695aac710fe4c199c). Modifications: no rich
console (progress goes through a callback), a start offset for trimming,
constant-frame-rate output so frame index and time stay in lockstep, a browser
preview rendition, an explicit FFmpeg availability check, and HDR tone-mapping
dropped (a clip is converted as-is; see README limitations).
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


class FFmpegMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    """A decoded clip, as OpenCV sees it."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


@functools.lru_cache(maxsize=None)
def ffmpeg_status() -> dict:
    """Which FFmpeg tools are available, and their version line."""
    out = {}
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        version = None
        if path:
            try:
                proc = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=10)
                version = proc.stdout.splitlines()[0] if proc.stdout else None
            except (OSError, subprocess.TimeoutExpired):
                path = None
        out[name] = {"path": path, "version": version}
    out["ok"] = all(out[n]["path"] for n in ("ffmpeg", "ffprobe"))
    return out


def require_ffmpeg() -> None:
    status = ffmpeg_status()
    if not status["ok"]:
        missing = [n for n in ("ffmpeg", "ffprobe") if not status[n]["path"]]
        raise FFmpegMissing(
            f"{' and '.join(missing)} not found on PATH. Install FFmpeg "
            "(macOS: `brew install ffmpeg`; Debian/Ubuntu: `apt install ffmpeg`).")


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed: {proc.stderr.strip()[-1500:]}")
    return proc


def _fraction(text: str | None) -> float:
    if not text or text in ("0/0", "N/A"):
        return 0.0
    if "/" in text:
        a, b = text.split("/", 1)
        return float(a) / float(b) if float(b) else 0.0
    return float(text)


def probe(path: Path) -> dict:
    """Container/stream facts from ffprobe. Raises RuntimeError if there is no video stream."""
    require_ffmpeg()
    proc = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=codec_name,width,height,nb_frames,duration,avg_frame_rate,r_frame_rate,"
                 "pix_fmt,color_transfer:stream_side_data=rotation:stream_tags=rotate"
                 ":format=duration", "-of", "json", str(path)], timeout=60)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("No video stream found. Is this a video file?")
    s = streams[0]
    rotation = 0
    for side in s.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(float(side["rotation"]))
    if not rotation and (s.get("tags") or {}).get("rotate"):
        rotation = -int(float(s["tags"]["rotate"]))
    width, height = int(s.get("width") or 0), int(s.get("height") or 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    fps = _fraction(s.get("avg_frame_rate")) or _fraction(s.get("r_frame_rate"))
    duration = float(s.get("duration") or (data.get("format") or {}).get("duration") or 0.0)
    return {"codec": s.get("codec_name"), "width": width, "height": height,
            "fps": round(fps, 3), "duration": round(duration, 3),
            "n_frames": int(s.get("nb_frames") or 0) or int(round(duration * fps)),
            "pix_fmt": s.get("pix_fmt"), "color_transfer": s.get("color_transfer"),
            "rotation": rotation}


def convert(src: Path, dst: Path, *, target_height: int | None, start: float = 0.0,
            duration: float | None = None, crf: int = 23, fps: float | None = None,
            progress=None, cancel_check=None) -> None:
    """Transcode to 8-bit H.264 at constant frame rate, rotation applied, no audio.

    *start*/*duration* trim the clip (seconds). The output's frame k is source
    frame ``round(start * fps) + k``; that offset is what results report against.
    """
    require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent jobs may convert the same content-addressed clip.
    tmp = dst.with_suffix(f".{uuid.uuid4().hex}.partial.mp4")
    cmd = ["ffmpeg", "-y", "-nostats", "-loglevel", "error", "-progress", "pipe:1"]
    if start:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration:.6f}"]
    chain = []
    if fps:
        chain.append(f"fps={fps:.6f}")
    if target_height:
        chain.append(f"scale=-2:'min({int(target_height)},ih)'")
    if chain:
        cmd += ["-vf", ",".join(chain)]
    cmd += ["-fps_mode", "cfr", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-an", "-map_metadata", "-1", str(tmp)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for line in proc.stdout:
            if cancel_check and cancel_check():
                proc.kill()
                break
            key, _, value = line.strip().partition("=")
            if progress and key == "out_time_us" and value.isdigit() and duration:
                progress(min(1.0, int(value) / 1e6 / duration))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg conversion failed: {proc.stderr.read().strip()[-1500:]}")
    tmp.replace(dst)   # atomic: a killed run never leaves half a file in the cache


def preview(src: Path, dst: Path, *, height: int = 480) -> None:
    """A small H.264 copy any browser can play, for the preview/trim step."""
    convert(src, dst, target_height=height, crf=28)


def inspect(path: Path) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path.name}")
    info = VideoInfo(path=path, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                     n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def read_frame(path: Path, index: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} of {path.name}")
    return frame


def encode_h264(src: Path, dst: Path, *, crf: int = 20) -> None:
    """Re-encode OpenCV's MPEG-4 output to H.264 tagged Rec. 709, which browsers play."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
          "-vf", "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
          "-movflags", "+faststart", str(dst)])


def segment(src: Path, dst: Path, *, start_frame: int, n_frames: int, fps: float,
            crf: int = 20) -> None:
    """Frames ``[start_frame, start_frame + n_frames)`` into their own MP4, frame-exact."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
          "-vf", f"select='between(n\\,{start_frame}\\,{start_frame + n_frames - 1})',setpts=N/({fps:.6f}*TB)",
          "-frames:v", str(int(n_frames)), "-fps_mode", "passthrough", "-c:v", "libx264",
          "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(tmp)])
    tmp.replace(dst)

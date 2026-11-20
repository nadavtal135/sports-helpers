"""Write the synthetic fixture-mode sample clip (see src/synthetic.py)."""

from __future__ import annotations

from pathlib import Path

import cv2

from .src import synthetic, video


def write_sample(dst: Path, *, size: tuple[int, int], fps: float, duration: float, draw_frame) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = dst.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for k in range(int(duration * fps)):
        writer.write(draw_frame(k / fps))
    writer.release()
    video.encode_h264(raw, dst, crf=26)
    raw.unlink(missing_ok=True)
    return dst


def make_sample(dst: Path) -> Path:
    return write_sample(dst, size=(synthetic.W, synthetic.H), fps=synthetic.FPS,
                        duration=synthetic.DURATION, draw_frame=synthetic.draw_frame)

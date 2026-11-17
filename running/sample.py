"""Write the synthetic fixture-mode sample clip (see src/synthetic.py)."""

from __future__ import annotations

from pathlib import Path

import cv2

from .src import synthetic, video


def make_sample(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = dst.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), synthetic.FPS,
                             (synthetic.W, synthetic.H))
    for k in range(int(synthetic.DURATION * synthetic.FPS)):
        writer.write(synthetic.draw_frame(k / synthetic.FPS))
    writer.release()
    video.encode_h264(raw, dst, crf=26)
    raw.unlink(missing_ok=True)
    return dst

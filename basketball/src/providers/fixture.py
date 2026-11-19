"""Fixture provider: simulated gateway replies for the bundled synthetic sample.

It answers in exactly the gateway's reply shapes, so the parsing and analysis
code under test is the code that runs live. It can only describe the synthetic
scene, so the job layer refuses to run it on anything but that sample clip
(checked by content hash) — it must never appear to analyze a real upload.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .. import synthetic


class FixtureProvider:
    name = "fixture"
    simulated = True

    def pose(self, video: Path, *, model: str, extra_body: dict, ctx: dict):
        return synthetic.pose_payload(ctx["source_offset"], ctx["n_frames"], ctx["fps"]), {"cost": 0}

    def segment(self, image_png: bytes, *, model: str, prompt: str, ctx: dict):
        img = cv2.imdecode(np.frombuffer(image_png, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        return synthetic.segment_payload(w, h, ctx.get("source_offset", 0) / synthetic.FPS), {"cost": 0}

    def track(self, video: Path, *, model: str, prompt: str, stride: int, ctx: dict):
        return synthetic.track_payload(ctx["source_offset"], ctx["n_frames"], ctx["fps"]), {"cost": 0}

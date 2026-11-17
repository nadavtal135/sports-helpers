"""Fixture provider: simulated ViTPose replies for the bundled synthetic sample.

Same reply shape as the gateway, so the live parsing/analysis path is what runs.
The job layer only allows it on the sample clip (checked by content hash), so it
never appears to analyze a real upload.
"""

from __future__ import annotations

from pathlib import Path

from .. import synthetic


class FixtureProvider:
    name = "fixture"
    simulated = True

    def pose(self, video: Path, *, model: str, extra_body: dict, ctx: dict):
        return synthetic.pose_payload(ctx["source_offset"], ctx["n_frames"], ctx["fps"]), {"cost": 0}

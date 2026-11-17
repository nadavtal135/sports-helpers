"""VLM Run Gateway calls (OpenAI-compatible chat completions).

The request shapes are copied from jeremyipark/vision-demos (Apache-2.0, commit
c06cfb78ccd8e2c63629222695aac710fe4c199c):

* pose — ``running/src/pose.py`` ``request_poses``

Only this module talks to the network. It never logs the API key, and the key
is read from the environment by the worker process, never from the browser.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path


class GatewayProvider:
    name = "vlmrun"
    simulated = False

    def __init__(self, api_key: str, *, base_url: str, timeout: float):
        if not api_key:
            raise RuntimeError("VLMRUN_API_KEY is not set on the server.")
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _create(self, attempts: int = 4, **kwargs):
        from openai import APIStatusError

        for attempt in range(attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except APIStatusError as exc:   # the gateway sometimes answers 502 under load
                if exc.status_code < 500 or attempt == attempts - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")

    @staticmethod
    def _reply(response) -> tuple[dict, dict | None]:
        payload = json.loads(response.choices[0].message.content)
        usage = response.usage.model_dump() if response.usage else None
        return payload, usage

    @staticmethod
    def _video_url(video: Path) -> dict:
        b64 = base64.b64encode(video.read_bytes()).decode("ascii")
        return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}

    # ── the three calls ──────────────────────────────────────────────────────
    def pose(self, video: Path, *, model: str, extra_body: dict, ctx: dict) -> tuple[dict, dict | None]:
        """ViTPose on every decoded frame. ``extra_body``: method=pose, video_fps
        (detector cadence), precision, video_max_frames (caps billed frames)."""
        response = self._create(
            model=model,
            messages=[{"role": "user", "content": [self._video_url(video)]}],
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        return self._reply(response)

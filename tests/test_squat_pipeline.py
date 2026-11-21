"""Replay a gateway response through the live provider without a key or network service."""

import base64
import functools
from importlib import import_module
import json

import httpx
import openai
import pytest

from running.src import video
from running.src.cache import Cache
from running.src.providers.gateway import GatewayProvider


@pytest.mark.parametrize("sport", ["squat", "standing_jump"])
def test_trimmed_mov_through_gateway_client_and_pose_cache(tmp_path, monkeypatch, sport):
    pipeline, sample, settings = (import_module(f"{sport}.{module}") for module in ("pipeline", "sample", "settings"))
    src = sample.make_sample(tmp_path / "side-view.mov")
    cfg = settings.Gateway(base_url="https://example.invalid/v1")
    calls = []

    def respond(request):
        body = json.loads(request.content)
        assert body["model"] == cfg.pose_model and body["method"] == "pose"
        assert body["video_max_frames"] == 99 and body["video_fps"] == pytest.approx(30)
        url = body["messages"][0]["content"][0]["video_url"]["url"]
        mime, data = url.split(",", 1)
        assert mime == "data:video/mp4;base64"
        sent = tmp_path / "sent.mp4"
        sent.write_bytes(base64.b64decode(data))
        assert video.probe(sent)["duration"] == pytest.approx(3.3)
        calls.append(request.url.path)
        payload = sample.pose_payload(93, 99, 30)
        return httpx.Response(200, json={
            "id": "side-view-test", "object": "chat.completion", "created": 0, "model": cfg.pose_model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(payload)}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.01},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(openai, "OpenAI", functools.partial(openai.OpenAI, http_client=client))
        provider = GatewayProvider("local-test", base_url=cfg.base_url, timeout=1)
        options = dict(provider=provider, cache=Cache(tmp_path / "cache"), gateway=cfg,
                       settings=settings.Settings(), trim_start=3.1, trim_end=6.4)
        live = pipeline.run(src, tmp_path / "first", **options)
        cached = pipeline.run(src, tmp_path / "repeat", **options)

    assert calls == ["/v1/chat/completions"]
    assert live["meta"]["provider"] == "vlmrun" and live["meta"]["simulated"] is False
    assert live["meta"]["cost_usd"] == 0.01 and cached["meta"]["cost_usd"] == 0
    if sport == "squat":
        assert live["reps"] == cached["reps"]
        rep, = live["reps"]
        assert rep["status"] == "completed" and rep["bottom_s"] == pytest.approx(4.7, abs=0.05)
    else:
        assert live["attempts"] == cached["attempts"]
        jump, = live["attempts"]
        assert jump["frames"]["takeoff"] == 130 and jump["times_s"]["contact"] == 4.8
        assert live["review"]["closer_look"]["kind"] == "Recovery"
    out = video.probe(tmp_path / f"first/{sport}_annotated.mp4")
    assert out["codec"] == "h264" and out["n_frames"] == 99

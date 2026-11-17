"""Live-mode request shapes (with a fake client, no network) and the live-mode gates."""

import json
import types

import pytest

from basketball.src.providers.gateway import TRACK_MAX_FRAMES, GatewayProvider


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        msg = types.SimpleNamespace(content=json.dumps({"content": {"object": "x"}}))
        usage = types.SimpleNamespace(model_dump=lambda: {"cost": 0.01})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


@pytest.fixture()
def provider():
    p = GatewayProvider("test-key", base_url="https://example.invalid/v1", timeout=5)
    p._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))
    return p


def test_pose_request_shape(provider, tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"\x00\x01")
    body = {"method": "pose", "video_fps": 15.0, "precision": 4, "video_max_frames": 90}
    payload, usage = provider.pose(clip, model="usyd-community/vitpose-plus-large", extra_body=body, ctx={})
    call = provider._client.chat.completions.calls[0]
    assert call["extra_body"] == body and call["response_format"] == {"type": "json_object"}
    part = call["messages"][0]["content"][0]
    assert part["type"] == "video_url" and part["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert usage == {"cost": 0.01}


def test_segment_and_track_request_shapes(provider, tmp_path):
    provider.segment(b"png", model="facebook/sam3.1", prompt="basketball hoop", ctx={})
    clip = tmp_path / "w.mp4"
    clip.write_bytes(b"\x00")
    provider.track(clip, model="facebook/sam3.1", prompt="basketball", stride=1, ctx={})
    seg, trk = provider._client.chat.completions.calls
    assert seg["extra_body"] == {"method": "segment",
                                 "method_params": {"prompt": "basketball hoop", "mask_format": "png"}}
    assert seg["messages"][0]["content"][1]["type"] == "image_url"
    assert trk["extra_body"]["method"] == "track"
    assert trk["extra_body"]["method_params"] == {"prompt": "basketball", "video_skip_frames": 1,
                                                  "video_max_frames": TRACK_MAX_FRAMES,
                                                  "mask_format": "png", "polygons": False}


def test_missing_key_is_a_clear_error():
    with pytest.raises(RuntimeError, match="VLMRUN_API_KEY"):
        GatewayProvider("", base_url="https://example.invalid", timeout=1)


def test_live_mode_requires_consent_and_bounds_length(tmp_path, monkeypatch):
    monkeypatch.setenv("SPORTS_HELPERS_DATA", str(tmp_path))
    monkeypatch.setenv("SPORTS_HELPERS_MODE", "live")
    monkeypatch.setenv("VLMRUN_API_KEY", "test-key")
    from webapp import jobs
    upload = {"id": "u1", "source": str(tmp_path / "x.mp4"), "duration": 600.0, "sample_of": None,
              "filename": "x.mp4", "sha256": "0", "fps": 30.0, "width": 1, "height": 1}
    m = jobs.JobManager()
    with pytest.raises(PermissionError, match="Confirm"):
        m.create(sport="running", upload=upload, settings={}, trim_start=0, trim_end=60, consent=False)
    with pytest.raises(ValueError, match="at most"):
        m.create(sport="running", upload=upload, settings={}, trim_start=0, trim_end=None, consent=True)

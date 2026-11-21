"""Run one analysis job in its own process.

    python -m webapp.worker <job_dir>

Reads ``job.json``, runs the sport's pipeline, writes ``result.json``, and
reports progress as JSON lines on stdout. One process per job means a cancel
is a hard kill, a crash cannot take the server down, and nothing global (like
the reference's per-run config mutation) can leak between jobs.

The API key, when needed, arrives in this process's environment only. It is
never written to disk, logged, or echoed.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def emit(**event) -> None:
    print(json.dumps(event), flush=True)


def main() -> int:
    job_dir = Path(sys.argv[1]).resolve()
    job = json.loads((job_dir / "job.json").read_text())
    sport = job["sport"]
    if sport == "basketball":
        from basketball import pipeline, settings as cfg
        from basketball.src.cache import Cache
    elif sport == "running":
        from running import pipeline, settings as cfg
        from running.src.cache import Cache
    elif sport == "standing_jump":
        from standing_jump import pipeline, settings as cfg
        from running.src.cache import Cache
    elif sport == "squat":
        from squat import pipeline, settings as cfg
        from running.src.cache import Cache
    else:
        emit(type="error", message=f"unknown sport {sport!r}")
        return 2

    if job["provider"] == "fixture":
        if sport == "basketball":
            from basketball.src.providers.fixture import FixtureProvider
        elif sport == "squat":
            from squat.sample import FixtureProvider
        elif sport == "standing_jump":
            from standing_jump.sample import FixtureProvider
        else:
            from running.src.providers.fixture import FixtureProvider
        provider = FixtureProvider()
    else:
        if sport == "basketball":
            from basketball.src.providers.gateway import GatewayProvider
        else:
            from running.src.providers.gateway import GatewayProvider
        gw = cfg.Gateway()
        provider = GatewayProvider(os.environ.get("VLMRUN_API_KEY", ""), base_url=gw.base_url,
                                   timeout=gw.request_timeout)

    settings = cfg.from_user(job.get("settings"))
    try:
        result = pipeline.run(
            Path(job["source"]), job_dir / "output", settings=settings, gateway=cfg.Gateway(),
            provider=provider, cache=Cache(Path(job["cache_dir"]) / sport),
            trim_start=float(job.get("trim_start") or 0.0), trim_end=job.get("trim_end"),
            progress=lambda stage, frac, msg: emit(type="progress", stage=stage,
                                                   fraction=round(float(frac), 3), message=msg))
    except pipeline.Cancelled:
        emit(type="cancelled")
        return 130
    except Exception as exc:   # noqa: BLE001 - reported to the user, traceback to stderr
        traceback.print_exc()
        emit(type="error", message=_friendly(exc))
        return 1
    tmp = job_dir / "result.partial.json"
    tmp.write_text(json.dumps(result, allow_nan=False, default=_nan_safe))
    tmp.replace(job_dir / "result.json")
    emit(type="done")
    return 0


def _nan_safe(obj):
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return None


def _friendly(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    name = exc.__class__.__name__
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return "The model provider rejected the API key. Check VLMRUN_API_KEY on the server."
    if name == "RateLimitError":
        return "The model provider is rate-limiting requests. Try again in a few minutes."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Could not reach the model provider (network error or timeout)."
    if name == "APIStatusError":
        return f"The model provider returned an error: {text[:300]}"
    return text[:600]


if __name__ == "__main__":
    sys.exit(main())

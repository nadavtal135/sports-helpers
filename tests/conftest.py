import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def fixture_url(tmp_path, monkeypatch):
    import uvicorn
    from webapp import server

    monkeypatch.setenv("SPORTS_HELPERS_DATA", str(tmp_path))
    monkeypatch.setenv("SPORTS_HELPERS_MODE", "fixture")
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        app = uvicorn.Server(uvicorn.Config(server.app, log_level="error"))
        thread = threading.Thread(target=app.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not app.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert app.started
            yield f"http://{host}:{port}"
        finally:
            app.should_exit = True
            thread.join(timeout=10)

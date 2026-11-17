"""Content-addressed cache for converted videos and model replies.

Keys hash the *content* of the video (not its name or mtime, as the reference
did), the model, every request parameter, and a contract version, so a renamed
file still hits and an edited one never does. Writes are atomic, so two jobs
racing on one key both end with a whole file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def key_of(**parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


class Cache:
    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = root
        self.enabled = enabled
        root.mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, key: str, suffix: str = ".json") -> Path:
        return self.root / kind / f"{key}{suffix}"

    def get_json(self, kind: str, key: str):
        p = self.path(kind, key)
        if not self.enabled or not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def put_json(self, kind: str, key: str, value) -> None:
        p = self.path(kind, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(value, fh)
        os.replace(tmp, p)

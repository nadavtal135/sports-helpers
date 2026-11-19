"""Camera cuts, so the hoop (and the shooter) can be located per camera segment.

New (not in the reference, which assumed one fixed camera for the whole clip).
``frame_differences`` reads the video (I/O); ``cuts_from_differences`` and
``segments`` are pure, and are what the tests exercise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def frame_differences(mp4: Path, *, size: tuple[int, int] = (64, 36)) -> np.ndarray:
    """Mean absolute difference (0-255) between each frame and the one before it,
    on small grayscale thumbnails. Element 0 is 0."""
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    out, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size,
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        out.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    return np.asarray(out, float)


def cuts_from_differences(diff: np.ndarray, *, threshold: float, ratio: float = 4.0,
                          window: int = 15, min_gap: int = 15) -> list[int]:
    """Frames that start a new shot of the camera.

    A cut is a difference spike that is both large in absolute terms (above
    *threshold*) and large against the clip's own motion around it (*ratio*
    times the median difference within ±*window* frames), so a fast-moving
    player or a pan does not count. Cuts closer than *min_gap* frames keep the
    stronger one.
    """
    diff = np.asarray(diff, float)
    found: list[int] = []
    for k in range(1, len(diff)):
        lo, hi = max(1, k - window), min(len(diff), k + window + 1)
        local = np.delete(diff[lo:hi], k - lo)
        base = float(np.median(local)) if local.size else 0.0
        if diff[k] > threshold and diff[k] > ratio * max(base, 0.5):
            if found and k - found[-1] < min_gap:
                if diff[k] > diff[found[-1]]:
                    found[-1] = k
                continue
            found.append(k)
    return found


def segments(n_frames: int, cuts: list[int], *, max_len: int | None = None) -> list[tuple[int, int]]:
    """``[(start, end), ...]`` covering ``[0, n_frames)``, split at *cuts* and,
    within a long camera segment, into pieces of at most *max_len* frames (so a
    slowly moving camera still gets its hoop re-located)."""
    bounds = [0, *sorted(c for c in cuts if 0 < c < n_frames), n_frames]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if max_len and b - a > max_len:
            pieces = int(np.ceil((b - a) / max_len))
            edges = np.linspace(a, b, pieces + 1).round().astype(int)
            out += list(zip(edges[:-1].tolist(), edges[1:].tolist()))
        elif b > a:
            out.append((a, b))
    return out

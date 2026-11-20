# Adapted from jeremyipark/vision-demos running/src/gait.py (Apache-2.0).
"""Gap-aware pose signal processing shared by Running and Squat Analysis."""

import numpy as np


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Kill isolated spikes without rounding off the peaks."""
    if k < 3:
        return x
    k += (k + 1) % 2   # force odd
    pad = np.pad(x, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average. Centered and edge-padded, so it does not shift peak times."""
    if k < 2:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation across NaN gaps of at most *max_gap* frames.

    Replaces the reference's ``_interp_nan``, which filled every gap however
    long. A long gap is missing evidence, and interpolating across it would turn
    a guess into a smooth, confident-looking signal; it stays NaN instead, and
    the clip's ends are never extrapolated.
    """
    out = np.array(x, dtype=float, copy=True)
    idx = np.flatnonzero(np.isfinite(out))
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [out[a], out[b]])
    return out


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs as ``(start, end_exclusive, label)``."""
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, int(labels[start])))
            start = i
    return runs


def smooth_runs(x: np.ndarray, median_k: int, mean_k: int) -> np.ndarray:
    """The reference's median-then-mean smoothing, applied within each finite run
    so a gap neither spreads NaN nor gets smoothed over."""
    out = np.full(len(x), np.nan)
    for start, end, label in segments(np.isfinite(x).astype(int)):
        if label:
            out[start:end] = _box_filter(_median_filter(x[start:end], median_k), mean_k)
    return out


def knee_flexion(hip, knee, ankle) -> np.ndarray:
    """Projected knee flexion (straight = 0 degrees) with both axes in the same unit."""
    v1 = np.vstack([hip[0] - knee[0], hip[1] - knee[1]])
    v2 = np.vstack([ankle[0] - knee[0], ankle[1] - knee[1]])
    n1, n2 = np.hypot(*v1), np.hypot(*v2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(axis=0) / (n1 * n2)
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

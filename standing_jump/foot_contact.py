"""Check a pose-estimated transition against shoe pixels in a fixed view.

The airborne frame supplies an image of the exposed floor. Each supported shoe
supplies a local sole level, including perspective differences between feet.
When the floor or shoe cannot be isolated, the caller keeps the ankle estimate.
"""

import cv2
import numpy as np

from running.src.signals import segments


def _sole_band(ground, empty, ankle, scale):
    """Find the bottom edge of the foreground component touching the tracked shoe."""
    height, width = ground.shape[:2]
    x, y = np.round(ankle).astype(int)
    dx = max(8, round(0.23 * scale))
    x0, x1 = max(0, x - dx), min(width, x + dx + 1)
    y0, y1 = max(0, y - round(.04 * scale)), min(height, y + round(.19 * scale) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    a, b = ground[y0:y1, x0:x1], empty[y0:y1, x0:x1]
    diff = cv2.absdiff(a, b).max(axis=2)
    mask = np.uint8(diff >= 32)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    # The component should reach the ankle neighborhood and form a shoe-width edge.
    choices = []
    for label in range(1, count):
        left, top, w, h, area = stats[label]
        if area < .003 * scale**2 or w < .07 * scale or h < .03 * scale:
            continue
        if left + x0 > x + .06 * scale or left + w + x0 < x - .06 * scale:
            continue
        if top + y0 > y + .07 * scale or top + h + y0 < y:
            continue
        choices.append((area, label))
    if not choices:
        return None
    component = labels == max(choices)[1]
    # A cast shadow or compression noise with just a pixel or two of support
    # must not set the floor reference.
    bottom = np.flatnonzero(component.sum(axis=1) >= max(4, round(.024 * scale)))
    if not bottom.size or bottom[-1] >= y1 - y0 - 2:
        return None
    sole_y = int(bottom[-1]) + y0
    if sole_y <= y + .012 * scale:
        return None
    xs = np.flatnonzero(component.any(axis=0))
    pad = max(2, round(.04 * scale))
    return (slice(sole_y - max(1, round(.007 * scale)), sole_y + 2),
            slice(max(0, x0 + int(xs[0]) - pad), min(width, x0 + int(xs[-1]) + pad)))


class ShoeContact:
    def __init__(self, path):
        self.reader = cv2.VideoCapture(str(path))

    def close(self):
        self.reader.release()

    def frame(self, i):
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.reader.read()
        return frame if ok else None

    def __call__(self, phase, estimate, peak, signals):
        if peak is None or not self.reader.isOpened():
            return None
        fps, scale = signals.fps, signals.scale
        leaving = phase == "takeoff"
        pad = max(3, round(.14 * fps))
        ref = estimate - pad if leaving else estimate + pad
        n = len(signals.ankles)
        if not 0 <= ref < n or not signals.grounded[ref].all():
            return None
        lo, hi = (ref, peak) if leaving else (peak, ref)
        if hi - lo < 3 or hi - lo > fps or not np.isfinite(signals.clearance[lo:hi + 1]).all():
            return None
        ground, empty = self.frame(ref), self.frame(peak)
        if ground is None or empty is None:
            return None
        bands = [_sole_band(ground, empty, p, scale) for p in signals.ankles[ref]]
        if any(b is None for b in bands):
            return None
        # Unstable floor texture during the air phase is a poor background reference.
        other_air = self.frame(peak + (1 if leaving else -1))
        if other_air is None or any(np.percentile(cv2.absdiff(empty[b], other_air[b]), 90) > 18 for b in bands):
            return None
        supported = []
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, lo)
        for _ in range(lo, hi + 1):
            ok, frame = self.reader.read()
            if not ok:
                return None
            supported.append(any(self.occupied(frame[b], empty[b], scale) for b in bands))
        # Require a sustained clear interval at the apex and a sustained planted
        # interval at the reference. Reject extra state changes from clutter or motion.
        spans = segments(np.array(supported))
        if len(spans) != 2:
            return None
        ground_span, air_span = spans if leaving else reversed(spans)
        if ground_span[2] != 1 or air_span[2] != 0 or min(b - a for a, b, _ in spans) < 2:
            return None
        found = lo + spans[1][0]
        return found if abs(found - estimate) <= pad else None

    @staticmethod
    def occupied(frame, empty, scale):
        changed = (cv2.absdiff(frame, empty).max(axis=2) >= 30).any(axis=0)
        return any(on and b - a >= max(3, round(.018 * scale)) for a, b, on in segments(changed))

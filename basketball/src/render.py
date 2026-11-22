# Adapted from jeremyipark/vision-demos basketball/src/render.py (Apache-2.0),
# commit c06cfb78ccd8e2c63629222695aac710fe4c199c. Modifications: the skeleton,
# rim and comet-tail ball drawing are kept; the court panel under the video, the
# credit lines, the fire effects and the elbow strip are replaced by a scoreboard
# bar and verdict labels that show UNKNOWN distinctly (amber) and state the FG%
# denominator; the court chart and ball-path plot are drawn with OpenCV.
"""The annotated video, the shot chart, and the ball paths at the hoop."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .ball import smoothed
from .court import CourtView
from .shots import streaks
from .skeleton import FACE_KPTS, SKELETON, edge_part

# BGR
MAKE = (80, 175, 60)
MISS = (60, 60, 225)
UNKNOWN = (40, 175, 240)
RIM = (40, 120, 240)
BALL = (40, 150, 255)
INK = (24, 24, 28)
SKELETON_COLORS = {
    "left_arm": (238, 211, 34), "right_arm": (241, 102, 99), "left_leg": (249, 121, 232),
    "right_leg": (133, 113, 251), "torso": (225, 213, 203),
}


def verdict_color(made):
    return MAKE if made is True else MISS if made is False else UNKNOWN


def _p(v):
    return int(round(v[0])), int(round(v[1]))


def _draw_skeleton(frame, pts, thick, radius):
    ok = np.isfinite(pts).all(axis=1)
    for a, b, _ in SKELETON:
        if a in FACE_KPTS or b in FACE_KPTS or not (ok[a] and ok[b]):
            continue
        cv2.line(frame, _p(pts[a]), _p(pts[b]), SKELETON_COLORS[edge_part(a, b)], thick, cv2.LINE_AA)
    for i in range(len(pts)):
        if ok[i] and i not in FACE_KPTS:
            cv2.circle(frame, _p(pts[i]), radius, (255, 255, 255), -1, cv2.LINE_AA)


def _text(img, text, org, scale, color=(255, 255, 255), thick=2, bg=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    if bg is not None:
        x, y = org
        cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 6, y + base + 4), bg, -1)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)
    return tw


def render(info, *, kp, hoop_at, track, shots, dst: Path, settings, stats_fn, progress=None,
           cancelled=None) -> dict:
    """Overlay the pose, the rim (lit by each verdict), the ball's trail and a scoreboard.

    *hoop_at(frame)* gives that frame's hoop (it can change at a camera cut), or None.
    """
    fps = info.fps
    out_h = min(int(settings.export_height), info.height) // 2 * 2
    sx = out_h / info.height
    width = int(round(info.width * sx / 2)) * 2
    scale = width / 1280
    thick = max(1, round(3 * scale * 1.5))
    radius = max(2, round(3 * scale))
    font = 0.8 * max(0.5, scale)
    flash = int(settings.verdict_seconds * fps)
    trail = int(settings.ball_trail_seconds * fps)
    def rim_of(hoop):
        return ((int(hoop.cx * sx), int(hoop.cy * sx)),
                (max(2, int(hoop.width * sx / 2)), max(2, int((hoop.cy - hoop.top) * sx))))
    streak = streaks(shots, info.n_frames)
    bx, by = smoothed(track, max_gap=int(settings.ball_fill_seconds * fps))
    ball_r = max(3, int(track.diameter * sx / 2)) if np.isfinite(track.diameter) else 6

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, out_h))
    for k in range(info.n_frames):
        ok, src = cap.read()
        if not ok:
            break
        if cancelled and k % 30 == 0 and cancelled():
            break
        frame = cv2.resize(src, (width, out_h), interpolation=cv2.INTER_AREA)
        pts = kp[k] * sx
        if np.isfinite(pts).any():
            _draw_skeleton(frame, pts, thick, radius)

        verdict = next((s for s in shots if s.decided is not None
                        and s.decided <= k < s.decided + flash), None)
        hot = streak[k] >= settings.hot_streak
        color = RIM if verdict is None else verdict_color(verdict.made)
        if hot and verdict is None:
            color = (0, 140, 255)
        hoop = hoop_at(k)
        rim_c, rim_ax = rim_of(hoop) if hoop is not None else ((width // 2, out_h // 4), (0, 0))
        if hoop is not None:
            cv2.ellipse(frame, rim_c, rim_ax, 0, 0, 360, color,
                        max(2, thick if verdict is not None else thick // 2 + 1), cv2.LINE_AA)

        active = next((s for s in shots if s.start - fps <= k <= (s.decided or s.end) + flash), None)
        if active is not None and np.isfinite(bx[k]):
            first = active.motion if active.motion is not None else active.start
            if k >= first:
                lo = max(active.start - int(fps), k - trail, first)
                path = [(f, bx[f] * sx, by[f] * sx) for f in range(lo, k + 1) if np.isfinite(bx[f])]
                for (f0, x0, y0), (f1, x1, y1) in zip(path, path[1:]):
                    if f1 - f0 > 6:
                        continue
                    fade = 0.25 + 0.75 * (f1 - lo) / max(1, k - lo)
                    cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)),
                             tuple(int(c * fade) for c in BALL), max(1, int(thick * fade)), cv2.LINE_AA)
                c = (int(bx[k] * sx), int(by[k] * sx))
                cv2.circle(frame, c, ball_r + 2, INK, 1, cv2.LINE_AA)
                cv2.circle(frame, c, ball_r, BALL, max(2, thick - 1), cv2.LINE_AA)

        if verdict is not None:
            label = f"SHOT {verdict.number}: {verdict.label}"
            y_label = max(int(30 * scale), rim_c[1] - rim_ax[0] - int(30 * scale))
            for text, size, thick_t, bg, dy in ((label, font * 1.1, 2, verdict_color(verdict.made), 0),
                                                (verdict.reason, font * 0.7, 1, INK, int(36 * scale))):
                tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick_t)[0][0]
                # Kept inside the frame: on a portrait clip the rim can sit near an edge.
                x_text = int(min(max(8, rim_c[0] - tw / 2), width - tw - 8))
                _text(frame, text, (x_text, y_label + dy), size, (255, 255, 255), thick_t, bg=bg)

        # Scoreboard: only shots already decided by this frame.
        done = [s for s in shots if s.decided is not None and s.decided <= k]
        t = stats_fn(done)
        fg = f"{t['fg_pct']:.0%} ({t['makes']}/{t['judged']} judged)" if t["judged"] else "-"
        bar = (f"MAKES {t['makes']}   MISSES {t['misses']}   UNKNOWN {t['unknown']}   FG {fg}"
               + (f"   HOT x{streak[k]}" if hot else ""))
        cv2.rectangle(frame, (0, out_h - int(44 * scale)), (width, out_h), INK, -1)
        _text(frame, bar, (int(14 * scale), out_h - int(14 * scale)), font * 0.85)
        writer.write(frame)
        if progress and k % 15 == 0:
            progress(k / max(1, info.n_frames))
    cap.release()
    writer.release()
    return {"width": width, "height": out_h}


def spread_points(points, *, min_dist: float, iterations: int = 60):
    """Nudge 2D points apart until none is closer than *min_dist* (or iterations run
    out), so overlapping shot dots stay readable. Order is preserved."""
    pts = np.array(points, float).reshape(-1, 2)
    rng = np.random.default_rng(0)
    for _ in range(iterations):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = pts[j] - pts[i]
                dist = float(np.hypot(*d))
                if dist >= min_dist:
                    continue
                if dist < 1e-6:
                    d, dist = rng.normal(size=2), 1.0
                    dist = float(np.hypot(*d))
                push = (min_dist - dist) / 2 * d / dist
                pts[i] -= push
                pts[j] += push
                moved = True
        if not moved:
            break
    return [(int(round(x)), int(round(y))) for x, y in pts]


def court_chart(shots, dst: Path, *, settings, stats) -> None:
    """Every shot at its *estimated* spot on a half court, with a legend."""
    view = CourtView(700, 720, three_radius=settings.three_point_radius_m,
                     ft_distance=settings.free_throw_distance_m, floor=(32, 34, 38),
                     paint=(48, 50, 56))
    img = np.vstack([view.base, np.full((150, 700, 3), (32, 34, 38), np.uint8)])
    placed = [(s, view.px(*view.clamp(*s.court_xy))) for s in shots if s.court_xy is not None]
    spread = spread_points([p for _, p in placed], min_dist=32)
    for (s, true), p in zip(placed, spread):
        if p != true:   # nudged apart from a neighbour: a leader line to its estimated spot
            cv2.line(img, true, p, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.circle(img, true, 3, (150, 150, 150), -1, cv2.LINE_AA)
        color = verdict_color(s.made)
        if s.made is None:
            cv2.circle(img, p, 15, color, 3, cv2.LINE_AA)   # hollow: not judged
        else:
            cv2.circle(img, p, 15, color, -1, cv2.LINE_AA)
        cv2.putText(img, str(s.number), (p[0] - 7 * len(str(s.number)), p[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    y = 760
    for color, label, filled in ((MAKE, "make", True), (MISS, "miss", True), (UNKNOWN, "unknown", False)):
        x = {"make": 30, "miss": 160, "unknown": 290}[label]
        cv2.circle(img, (x, y), 10, color, -1 if filled else 3, cv2.LINE_AA)
        _text(img, label, (x + 18, y + 7), 0.6, (230, 230, 230), 1)
    fg = (f"FG {stats['fg_pct']:.0%} = {stats['makes']} makes / {stats['judged']} judged shots"
          if stats["judged"] else "FG: no judged shots")
    _text(img, fg, (30, 800), 0.6, (230, 230, 230), 1)
    _text(img, f"{stats['unknown']} unknown excluded. Spots are ESTIMATES from one camera",
          (30, 830), 0.55, (180, 180, 180), 1)
    _text(img, f"(scaled by shooter height {settings.player_height_m:g} m, "
               f"FOV {settings.camera_hfov_degrees:g} deg).", (30, 855), 0.55, (180, 180, 180), 1)
    cv2.imwrite(str(dst), img)


def ball_paths(shots, hoop, frame_bgr, dst: Path, *, settings) -> None:
    """Every judged ball path over a crop of the hoop: what each verdict was read from."""
    w = hoop.width
    x0, x1 = max(0, int(hoop.cx - 4 * w)), min(frame_bgr.shape[1], int(hoop.cx + 4 * w))
    y0, y1 = max(0, int(hoop.top - 3.5 * w)), min(frame_bgr.shape[0], int(hoop.net_bottom + 2.5 * w))
    crop = (frame_bgr[y0:y1, x0:x1] * 0.55).astype(np.uint8)
    k = 600 / max(1, crop.shape[1])
    crop = cv2.resize(crop, (600, int(crop.shape[0] * k)))

    def P(x, y):
        return int((x - x0) * k), int((y - y0) * k)

    inner = settings.rim_inner_fraction * w / 2
    cv2.line(crop, P(hoop.left, hoop.cy), P(hoop.right, hoop.cy), RIM, 3, cv2.LINE_AA)
    cv2.line(crop, P(hoop.cx - inner, hoop.cy), P(hoop.cx + inner, hoop.cy), (255, 255, 255), 1)
    cv2.line(crop, P(x0, hoop.net_bottom), P(x1, hoop.net_bottom), (150, 150, 150), 1)
    for s in shots:
        if not s.path:
            continue
        color = verdict_color(s.made)
        pts = [P(x, y) for _, x, y in s.path]
        for (f0, *_), (f1, *_), a, b in zip(s.path, s.path[1:], pts, pts[1:]):
            if f1 - f0 <= 6:   # don't draw across tracking gaps
                cv2.line(crop, a, b, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(crop, p, 2, color, -1)
        i = int(np.argmin([abs(x - hoop.cx) + abs(y - hoop.cy) for _, x, y in s.path]))
        _text(crop, str(s.number), (pts[i][0] + 6, pts[i][1] - 6), 0.5, color, 1)
    _text(crop, "Ball paths at the hoop (green make, red miss, amber unknown)", (8, 20), 0.5,
          (255, 255, 255), 1)
    cv2.imwrite(str(dst), crop)


def hoop_check(hoops, frames: dict, dst: Path, *, width: int = 600) -> None:
    """``hoop_check.png``: does the rim outline sit on the rim? One zoomed panel per hoop
    used (at most 4), cropped from the frame SAM looked at, with the rim ellipse, the rim
    line the ball must cross, the net bottom and their pixel numbers."""
    panels = []
    for h in hoops[:4]:
        img = frames.get(h.frame)
        if img is None:
            continue
        w = max(h.width, 4.0)
        x0, x1 = max(0, int(h.cx - 2.5 * w)), min(img.shape[1], int(h.cx + 2.5 * w) + 1)
        y0 = max(0, int(min(h.top, h.cy) - 1.5 * w))
        y1 = min(img.shape[0], int(h.net_bottom + 1.0 * w) + 1)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        k = width / crop.shape[1]
        crop = cv2.resize(crop, (width, max(1, int(round(crop.shape[0] * k)))),
                          interpolation=cv2.INTER_CUBIC)

        def P(x, y):
            return int(round((x - x0) * k)), int(round((y - y0) * k))

        half_h = max(2.0, h.cy - h.top)
        cv2.ellipse(crop, P(h.cx, h.cy), (int(w / 2 * k), max(2, int(half_h * k))), 0, 0, 360,
                    RIM, 2, cv2.LINE_AA)
        cv2.line(crop, P(h.left - 0.6 * w, h.cy), P(h.right + 0.6 * w, h.cy), (255, 255, 255), 1,
                 cv2.LINE_AA)
        cv2.line(crop, P(h.left, h.net_bottom), P(h.right, h.net_bottom), (150, 150, 150), 1,
                 cv2.LINE_AA)
        for x in (h.left, h.right):
            cv2.line(crop, P(x, h.cy - 0.4 * w), P(x, h.cy + 0.4 * w), RIM, 1, cv2.LINE_AA)
        _text(crop, f"rim line y={h.cy:.1f}", (min(P(h.right + 0.6 * w, 0)[0] + 10, width - 150),
                                                P(0, h.cy)[1] + 5), 0.5, (255, 255, 255), 1,
              bg=(0, 0, 0))
        _text(crop, f"net bottom y={h.net_bottom:.0f}", (8, P(0, h.net_bottom)[1] + 18), 0.5,
              (200, 200, 200), 1, bg=(0, 0, 0))
        _text(crop, f"frame {h.frame}: rim x {h.left:.0f}-{h.right:.0f} (center {h.cx:.0f}, "
              f"{h.width:.0f} px wide), top y={h.top:.0f}", (8, 20), 0.5, (255, 255, 255), 1,
              bg=(0, 0, 0))
        if h.notes:
            _text(crop, "mask contaminated: rim read from its top", (8, 42), 0.5, (80, 200, 255), 1,
                  bg=(0, 0, 0))
        panels.append(crop)
    if not panels:
        return
    footer = np.full((34, width, 3), 0, np.uint8)
    _text(footer, "Rim check: does this outline sit on the rim?", (8, 23), 0.55, (255, 255, 255), 1)
    sheet = np.vstack([np.vstack([p, np.full((6, width, 3), 30, np.uint8)]) for p in panels]
                      + [footer])
    cv2.imwrite(str(dst), sheet)

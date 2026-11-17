# Adapted from jeremyipark/vision-demos running/src/render.py (Apache-2.0),
# commit c06cfb78ccd8e2c63629222695aac710fe4c199c. Modifications: the skeleton
# overlay and strike flash are kept; the side panel, comet trails and credit are
# replaced by a compact HUD (live cadence, steps, tracking-gap notice); the
# clearance plot is redrawn with the strike markers and gap shading.
"""The annotated running video and the ankle-clearance chart."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .skeleton import FACE_KPTS, SKELETON, edge_part

COLORS = {"left_arm": (238, 211, 34), "right_arm": (241, 102, 99), "left_leg": (249, 121, 232),
          "right_leg": (133, 113, 251), "torso": (225, 213, 203)}
FOOT = {"left": (249, 121, 232), "right": (133, 113, 251)}
J = {"left_ankle": 15, "right_ankle": 16}


def _text(img, text, org, scale, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(info, *, kp, analysis, dst: Path, settings, progress=None, cancelled=None) -> None:
    fps = info.fps
    out_h = min(int(settings.export_height), info.height) // 2 * 2
    sx = out_h / info.height
    width = int(round(info.width * sx / 2)) * 2
    scale = width / 960
    thick = max(2, round(4 * scale))
    flash = int(settings.flash_seconds * fps)
    strikes = {f: [s.frame for s in analysis.steps_for(f)] for f in ("left", "right")}
    gap = {f: ~np.isfinite(analysis.clearance[f]) for f in analysis.present}

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
        fin = np.isfinite(pts).all(axis=1)
        for a, b, _ in SKELETON:
            if a in FACE_KPTS or b in FACE_KPTS or not (fin[a] and fin[b]):
                continue
            cv2.line(frame, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                     COLORS[edge_part(a, b)], thick, cv2.LINE_AA)
        for foot in ("left", "right"):
            j = J[f"{foot}_ankle"]
            if not fin[j]:
                continue
            c = tuple(int(v) for v in pts[j])
            cv2.circle(frame, c, int(7 * scale), FOOT[foot], -1, cv2.LINE_AA)
            recent = [f for f in strikes[foot] if 0 <= k - f < flash]
            if recent:
                r = int((10 + 14 * (k - recent[-1]) / max(1, flash)) * scale)
                cv2.circle(frame, c, r, (255, 255, 255), max(2, thick // 2), cv2.LINE_AA)

        cad = analysis.cadence_at(k)
        steps = analysis.steps_by(k)
        cv2.rectangle(frame, (0, 0), (int(330 * scale), int(92 * scale)), (24, 24, 28), -1)
        _text(frame, (f"{cad:.0f} spm" if np.isfinite(cad) else "-- spm"), (int(14 * scale), int(48 * scale)),
              1.3 * scale, (255, 255, 255), max(2, int(3 * scale)))
        _text(frame, f"cadence (live)  |  {steps} step{'s' if steps != 1 else ''}",
              (int(14 * scale), int(78 * scale)), 0.55 * scale, (200, 200, 200), 1)
        lost = [f for f in gap if gap[f][k]]
        if lost:
            _text(frame, f"tracking gap: {' & '.join(lost)} leg", (int(14 * scale), out_h - int(18 * scale)),
                  0.7 * scale, (40, 175, 240), 2)
        writer.write(frame)
        if progress and k % 15 == 0:
            progress(k / max(1, info.n_frames))
    cap.release()
    writer.release()


def plot_clearance(a, dst: Path, *, offset: int = 0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = a.times + offset / a.fps
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=110)
    colors = {"left": "#d63384", "right": "#6f42c1"}
    for foot in ("left", "right"):
        c = a.clearance[foot]
        if not np.isfinite(c).any():
            continue
        ax.plot(t, c, color=colors[foot], lw=1.6, label=f"{foot} ankle")
        gap = ~np.isfinite(c)
        if gap.any():
            ax.fill_between(t, 0, 1, where=gap, transform=ax.get_xaxis_transform(),
                            color="#adb5bd", alpha=0.25, lw=0)
        for s in a.steps_for(foot):
            ax.axvline(s.time + offset / a.fps, color=colors[foot], lw=0.8, ls=":", alpha=0.8)
            ax.plot(s.time + offset / a.fps, a.contact_level, "v", color=colors[foot], ms=6)
    ax.axhline(a.contact_level, color="#495057", lw=1, ls="--", label="strike threshold")
    cad = a.mean_cadence
    ax.set_title(f"Ankle clearance: {a.count} strikes, cadence "
                 + (f"{cad:.1f} steps/min" if np.isfinite(cad) else "not measured")
                 + "  (grey = tracking gap, not interpolated)", loc="left")
    ax.set_xlabel("time in clip (s)")
    ax.set_ylabel("ankle height (leg lengths)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)

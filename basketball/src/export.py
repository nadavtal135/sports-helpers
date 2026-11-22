# Adapted from jeremyipark/vision-demos basketball/src/export.py (Apache-2.0),
# commit c06cfb78ccd8e2c63629222695aac710fe4c199c. Modifications: unknown is its
# own result, FG% excludes it and names its denominator, times are in the
# original clip (trim offset added) with source frame numbers, angles and court
# spots are labeled as projected/estimated, and run metadata is included.
"""CSV and JSON exports: one row per shot, one row per frame, and analysis.json."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _num(value, digits: int = 2):
    return "" if value is None or not np.isfinite(value) else round(float(value), digits)


def write(shots, angle, side_on, track, fps: float, offset: int, out_dir: Path, *, stats: dict,
          meta: dict, warnings: list[str], is_three) -> list[str]:
    def t(frame):
        return "" if frame is None else round((frame + offset) / fps, 3)

    with (out_dir / "shots.csv").open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["shot", "result", "reason", "evidence", "hands_up_s", "release_s",
                      "release_source_frame", "release_method", "rim_crossing_s", "decided_s",
                      "side_on", "elbow_set_projected_deg", "elbow_release_projected_deg",
                      "court_x_m_estimate", "court_y_m_estimate", "distance_ft_estimate",
                      "three_point_estimate", "camera_segment"])
        for s in shots:
            x, y = s.court_xy or (None, None)
            out.writerow([s.number, s.result, s.reason, " | ".join(s.evidence), t(s.start),
                          t(s.release), s.release + offset, s.release_method, t(s.crossing),
                          t(s.decided), s.side_on, _num(s.elbow_set, 1), _num(s.elbow_release, 1),
                          _num(x), _num(y),
                          _num(s.distance_m * 3.2808 if s.distance_m is not None else None, 1),
                          "" if s.court_xy is None else is_three(*s.court_xy),
                          "" if s.segment is None else s.segment])

    shot_of = np.zeros(len(angle), int)
    for s in shots:
        shot_of[s.start:(s.decided or s.end) + 1] = s.number
    with (out_dir / "frames.csv").open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["frame", "source_frame", "t_s", "elbow_projected_deg", "side_on",
                      "ball_x_px", "ball_y_px", "ball_tracked", "shot"])
        for k in range(len(angle)):
            out.writerow([k, k + offset, round((k + offset) / fps, 3), _num(angle[k], 1),
                          bool(side_on[k]), _num(track.x[k], 1), _num(track.y[k], 1),
                          bool(track.covered[k]), shot_of[k] or ""])

    (out_dir / "analysis.json").write_text(json.dumps({
        "stats": stats,
        "warnings": warnings,
        "notes": {
            "fg_pct": "makes / (makes + misses). Unknown shots are not misses and are excluded.",
            "court": "Shot spots are estimates from one camera, scaled by the shooter height "
                     "and field of view settings.",
            "elbow": "Projected 2D elbow angles in the image, reported only for side-on shots. "
                     "They are not anatomical joint angles.",
            "times": "Seconds in the original uploaded clip; source_frame is that clip's frame index.",
        },
        "shots": [{**s.summary(fps, offset),
                   "three_point_estimate": None if s.court_xy is None else bool(is_three(*s.court_xy)),
                   "ball_path": [[f + offset, round(x, 1), round(y, 1)] for f, x, y in s.path]}
                  for s in shots],
        "meta": meta,
    }, indent=2))
    return ["shots.csv", "frames.csv", "analysis.json"]


def write_detections(all_boxes: dict, track, fps: float, offset: int, out_dir: Path, *,
                     min_score: float, rescue_min: float) -> str:
    """``ball_detections.csv``: every ball box SAM returned, at any score, for diagnosis.

    ``score_band`` is "main" (at least *min_score*, eligible for the normal pick),
    "rescue" (between *rescue_min* and *min_score*: only the blur rescue may use it)
    or "below" (never used); "on head" for a box on a person's head (not the ball);
    "static" for a ball that never moves (a ball on a
    chair: never the shot ball) and "gated" for a box rejected as off the ball's
    path. Rows with band "net_search" are not SAM boxes: the ball found in the net
    by the local colour search (score empty). ``selected`` marks the box the track
    kept on that frame. Coordinates are pixels of the analyzed video (see
    analysis.json meta).
    """
    from .ball import key
    name = "ball_detections.csv"
    with (out_dir / name).open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["frame", "source_frame", "t_s", "cx_px", "cy_px", "w_px", "h_px", "score",
                      "score_band", "selected", "window_start_frame"])
        for f in sorted(all_boxes):
            for cx, cy, w, h, score, window in all_boxes[f]:
                band = ("on head" if key(f, cx, cy) in track.on_head
                        else "static" if key(f, cx, cy) in track.static
                        else "gated" if key(f, cx, cy) in track.gated
                        else "main" if score >= min_score else "rescue" if score >= rescue_min
                        else "below")
                chosen = (0 <= f < len(track.x) and np.isfinite(track.x[f])
                          and abs(track.x[f] - cx) < 0.5 and abs(track.y[f] - cy) < 0.5)
                out.writerow([f, f + offset, round((f + offset) / fps, 3), round(cx, 1), round(cy, 1),
                              round(w, 1), round(h, 1), round(score, 3), band, bool(chosen),
                              window])
        recovered = track.recovered if track.recovered is not None else []
        for f in np.flatnonzero(recovered):
            out.writerow([f, f + offset, round((f + offset) / fps, 3), round(float(track.x[f]), 1),
                          round(float(track.y[f]), 1), "", "", "", "net_search", True, ""])
    return name

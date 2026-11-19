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

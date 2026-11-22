"""Render a stretch of a basketball clip with every ball detection drawn on it.

    python -m basketball.diagnose JOB_OUTPUT_DIR CLIP --start 20.0 --end 23.0 \\
        --scale 2 --out sheet.png [--frames-dir frames/] [--step 1] [--cols 6] [--crop X0,Y0,X1,Y1]

JOB_OUTPUT_DIR is a finished job's output folder (it needs analysis.json, frames.csv
and ball_detections.csv); CLIP is the original video. Times are seconds in the
original clip, as in the results. Nothing is sent anywhere: this only reads the
job's exported files, which already hold every box SAM returned (cached replies).

On each frame:
  green box   score >= ball_min_score: eligible for the normal pick
  orange box  between ball_rescue_min_score and ball_min_score: only the blur rescue may use it
  red box     below ball_rescue_min_score: never used
  purple box  "static": a ball that never moves (on a chair, a rack): never the shot ball
  magenta box "gated": off the ball's path (too far or too fast to be the same ball)
  cyan box    "net": the ball found in the net by the local colour search (not a SAM box)
  yellow box  "head": on a person's head, not the ball (never used)
  cyan ring   where the ball track was on that frame (thick if the box was kept)
  orange line the rim (center line), grey line the net bottom
  "not tracked" in the corner: SAM did not look at this frame (outside every ball window)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

GREEN, ORANGE, RED, CYAN, RIM, GREY = (60, 200, 60), (0, 165, 255), (60, 60, 230), (255, 220, 0), \
    (40, 120, 240), (160, 160, 160)
PURPLE, MAGENTA, YELLOW = (200, 80, 150), (220, 0, 220), (0, 230, 255)


def load_job(job_dir: Path) -> dict:
    analysis = json.loads((job_dir / "analysis.json").read_text())
    meta = analysis["meta"]
    frames = {}
    with (job_dir / "frames.csv").open() as fh:
        for row in csv.DictReader(fh):
            frames[int(row["frame"])] = row
    boxes: dict[int, list[dict]] = {}
    path = job_dir / "ball_detections.csv"
    if path.is_file():
        with path.open() as fh:
            for row in csv.DictReader(fh):
                boxes.setdefault(int(row["frame"]), []).append(row)
    hoops = meta.get("hoops") or ([{"start_frame": 0, "end_frame": 10 ** 9, "hoop": meta["hoop"]}]
                                  if meta.get("hoop") else [])
    return {"meta": meta, "frames": frames, "boxes": boxes, "hoops": hoops,
            "has_boxes": path.is_file()}


def _hoop_at(hoops: list, frame: int) -> dict | None:
    for seg in hoops:
        if seg["start_frame"] <= frame < seg["end_frame"]:
            return seg.get("hoop")
    return None


def render_frame(img: np.ndarray, job: dict, frame: int, t: float, *, sx: float, sy: float,
                 scale: float) -> np.ndarray:
    """Draw the detections of analyzed frame *frame* on *img* (a frame of the clip)."""
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    k = scale

    def P(x, y):
        return int(round(float(x) * sx * k)), int(round(float(y) * sy * k))

    hoop = _hoop_at(job["hoops"], frame)
    if hoop:
        cv2.line(out, P(hoop["left"], hoop["cy"]), P(hoop["right"], hoop["cy"]), RIM, 2, cv2.LINE_AA)
        cv2.line(out, P(hoop["left"], hoop["net_bottom"]), P(hoop["right"], hoop["net_bottom"]),
                 GREY, 1, cv2.LINE_AA)
    for b in job["boxes"].get(frame, []):
        color = {"main": GREEN, "rescue": ORANGE, "static": PURPLE, "gated": MAGENTA,
                 "net_search": CYAN, "on head": YELLOW}.get(b["score_band"], RED)
        cx, cy = float(b["cx_px"]), float(b["cy_px"])
        w = float(b["w_px"]) if b["w_px"] else 12.0     # net-search points have no size
        h = float(b["h_px"]) if b["h_px"] else 12.0
        cv2.rectangle(out, P(cx - w / 2, cy - h / 2), P(cx + w / 2, cy + h / 2), color, 2)
        tag = {"static": "static", "gated": "gated", "net_search": "net",
               "on head": "head"}.get(b["score_band"])
        text = f"{float(b['score']):.2f}" if b["score"] else ""
        text = f"{text} {tag}".strip() if tag else text
        cv2.putText(out, text, P(cx - w / 2, cy - h / 2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k / 2 + 0.2, color, 1, cv2.LINE_AA)
    row = job["frames"].get(frame)
    if row and row.get("ball_x_px"):
        c = P(row["ball_x_px"], row["ball_y_px"])
        kept = any(b.get("selected") == "True" for b in job["boxes"].get(frame, []))
        cv2.circle(out, c, int(9 * k), CYAN, 3 if kept else 1, cv2.LINE_AA)
    return out


def label(tile: np.ndarray, job: dict, frame: int, t: float) -> np.ndarray:
    """The time, frame and tracking status, top-left (drawn after any crop)."""
    text = f"{t:.2f}s  f{frame}"
    row = job["frames"].get(frame)
    if row is not None and row.get("ball_tracked") == "False":
        text += "  not tracked"
    elif row is not None and not row.get("ball_x_px"):
        text += "  no ball"
    size = 0.5
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
    cv2.rectangle(tile, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(tile, text, (5, th + 4), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def contact_sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    return sheet


def diagnose(job_dir: Path, clip: Path, *, start: float, end: float, scale: float = 2.0,
             step: int = 1, crop: tuple | None = None, out: Path | None = None,
             frames_dir: Path | None = None, cols: int = 6) -> dict:
    job = load_job(job_dir)
    meta = job["meta"]
    fps = meta["analysis_video"]["fps"]
    offset = meta["trim"]["source_frame_offset"]
    aw, ah = meta["analysis_video"]["width"], meta["analysis_video"]["height"]
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")
    tiles, written = [], []
    first = int(round(start * fps)) - offset
    last = int(round(end * fps)) - offset
    for frame in range(max(0, first), last + 1, max(1, step)):
        t = (frame + offset) / fps
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        if not ok:
            break
        sx, sy = img.shape[1] / aw, img.shape[0] / ah
        tile = render_frame(img, job, frame, t, sx=sx, sy=sy, scale=scale)
        if crop:
            x0, y0, x1, y1 = (int(round(v * s * scale)) for v, s in zip(crop, (sx, sy, sx, sy)))
            tile = tile[y0:y1, x0:x1].copy()
        tile = label(tile, job, frame, t)
        tiles.append(tile)
        if frames_dir:
            frames_dir.mkdir(parents=True, exist_ok=True)
            path = frames_dir / f"f{frame + offset:05d}_{t:06.2f}s.png"
            cv2.imwrite(str(path), tile)
            written.append(path)
    cap.release()
    if out and tiles:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), contact_sheet(tiles, cols))
    return {"frames": len(tiles), "frame_files": written, "sheet": out if tiles else None,
            "has_detections": job["has_boxes"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("job_dir", type=Path, help="a finished job's output folder")
    ap.add_argument("clip", type=Path, help="the original video")
    ap.add_argument("--start", type=float, required=True, help="seconds in the original clip")
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--step", type=int, default=1, help="every Nth frame")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--crop", help="X0,Y0,X1,Y1 in analyzed-video pixels")
    ap.add_argument("--out", type=Path, default=Path("ball_sheet.png"))
    ap.add_argument("--frames-dir", type=Path)
    a = ap.parse_args(argv)
    crop = tuple(float(v) for v in a.crop.split(",")) if a.crop else None
    res = diagnose(a.job_dir, a.clip, start=a.start, end=a.end, scale=a.scale, step=a.step,
                   crop=crop, out=a.out, frames_dir=a.frames_dir, cols=a.cols)
    if not res["has_detections"]:
        print("note: no ball_detections.csv in this job (it predates the export); only the "
              "selected track is drawn. Re-run the job: cached replies make it free.")
    print(f"{res['frames']} frames -> {res['sheet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

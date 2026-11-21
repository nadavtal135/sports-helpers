"""Overlay observed pose, completed count and phase on the trimmed clip."""

from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from running.src.pose_clip import check_cancelled
from running.src.skeleton import FACE_KPTS, SKELETON
from . import video
from .side_view import JOINTS


def render(info, points, phases, side, ends, dst: Path, *, export_height: int, progress, cancelled):
    height = min(export_height, info.height) // 2 * 2
    width = max(2, round(info.width * height / info.height / 2) * 2)
    resize = np.array([width / info.width, height / info.height])
    scale = min(width, height) / 540
    thickness = max(1, round(3 * scale))
    joints = JOINTS.get(side, [])
    total = 0
    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (width, height))
    try:
        if not cap.isOpened() or not writer.isOpened():
            raise RuntimeError("Could not open the video for annotation.")
        for i, phase in enumerate(phases):
            if i % 15 == 0:
                check_cancelled(cancelled)
                progress(i / info.n_frames)
            ok, src = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {i} of the video.")
            img = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
            xy = points[i] * resize
            pts = [tuple(p.round().astype(int)) if np.isfinite(p).all() else None for p in xy]
            for a, b, _ in SKELETON:
                if a in FACE_KPTS or b in FACE_KPTS or pts[a] is None or pts[b] is None:
                    continue
                cv2.line(img, pts[a], pts[b], (175, 176, 166), thickness, cv2.LINE_AA)
            # Draw the measured side last so the far side cannot cover it.
            if joints:
                hip, knee, ankle, shoulder = joints
                color = (106, 224, 243)
                for a, b in ((shoulder, hip), (hip, knee), (knee, ankle)):
                    if pts[a] is not None and pts[b] is not None:
                        cv2.line(img, pts[a], pts[b], color, thickness + 1, cv2.LINE_AA)
                for j in joints:
                    if pts[j] is not None:
                        cv2.circle(img, pts[j], thickness + 2, (255, 255, 255), -1, cv2.LINE_AA)

            total += i in ends
            cv2.rectangle(img, (0, 0), (round(265 * scale), round(85 * scale)), (32, 30, 27), -1)
            cv2.putText(img, phase.capitalize(), (round(12 * scale), round(34 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.76 * scale, (255, 255, 255), thickness, cv2.LINE_AA)
            label = f"{total} completed"
            if side:
                label += f"  |  {side} side"
            cv2.putText(img, label, (round(12 * scale), round(66 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.56 * scale, (106, 224, 243), max(1, thickness - 1), cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()
        cap.release()


def export_videos(info, out_dir, sport, draw, crf, cancelled):
    """Publish both renditions on the exact trimmed inference-video clock."""
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        raw = Path(tmp) / "overlay.mp4"
        draw(raw)
        check_cancelled(cancelled)
        video.encode_h264(raw, out_dir / f"{sport}_annotated.mp4", crf=crf)
    check_cancelled(cancelled)
    clean = out_dir / f"{sport}_clean.mp4"
    clean.unlink(missing_ok=True)
    try:
        clean.hardlink_to(info.path)
    except OSError:
        shutil.copyfile(info.path, clean)

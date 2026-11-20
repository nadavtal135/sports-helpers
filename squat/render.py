"""Overlay observed pose, completed count and phase on the trimmed clip."""

from pathlib import Path

import cv2
import numpy as np

from running.src.pose_clip import check_cancelled
from running.src.skeleton import FACE_KPTS, SKELETON
from .analysis import JOINTS, Analysis


def render(info, analysis: Analysis, dst: Path, *, export_height: int, progress, cancelled):
    height = min(export_height, info.height) // 2 * 2
    width = max(2, round(info.width * height / info.height / 2) * 2)
    resize = np.array([width / info.width, height / info.height])
    scale = min(width, height) / 540
    thickness = max(1, round(3 * scale))
    joints = JOINTS.get(analysis.side, [])
    ends = {a.end for a in analysis.attempts if a.status == "completed"}
    total = 0
    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (width, height))
    try:
        if not cap.isOpened() or not writer.isOpened():
            raise RuntimeError("Could not open the squat video for annotation.")
        for i, phase in enumerate(analysis.phases):
            if i % 15 == 0:
                check_cancelled(cancelled)
                progress(i / info.n_frames)
            ok, src = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {i} of the squat video.")
            img = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
            xy = analysis.points[i] * resize
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
            if analysis.side:
                label += f"  |  {analysis.side} side"
            cv2.putText(img, label, (round(12 * scale), round(66 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.56 * scale, (106, 224, 243), max(1, thickness - 1), cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()
        cap.release()

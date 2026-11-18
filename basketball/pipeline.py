"""The basketball pipeline: video -> poses, hoop, ball -> shots -> video, charts, exports.

Adapted from jeremyipark/vision-demos ``basketball/main.py`` (Apache-2.0,
commit c06cfb78ccd8e2c63629222695aac710fe4c199c). Modifications: a function
taking explicit settings, a provider and a cache instead of a script reading a
mutated ``config`` module; content-hash cache keys; trimming with source-frame
offsets carried into every output; progress/cancel callbacks; an "unknown"
outcome end to end; no console UI.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

from .settings import PIPELINE_VERSION, Gateway, Settings
from .src import ball as ball_mod
from .src import court, export, hoop as hoop_mod, pose, render, shots as shots_mod, video
from .src.cache import Cache, file_sha256, key_of
from .src.providers.gateway import TRACK_MAX_FRAMES

POSE_CONTRACT = "flat-envelope-v1"


class Cancelled(Exception):
    pass


def _noop(*_a, **_k):
    return None


def run(source: Path, out_dir: Path, *, settings: Settings, gateway: Gateway, provider,
        cache: Cache, trim_start: float = 0.0, trim_end: float | None = None,
        progress=_noop, cancelled=lambda: False) -> dict:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    warnings_list: list[str] = []
    costs = {"pose": 0.0, "hoop": 0.0, "ball": 0.0}

    def check():
        if cancelled():
            raise Cancelled()

    # ── 1. source and conversion ─────────────────────────────────────────────
    progress("convert", 0.0, "Converting video")
    src = video.probe(source)
    fps_src = src["fps"] or 30.0
    duration = (trim_end - trim_start) if trim_end else None
    src_hash = file_sha256(source)
    conv_key = key_of(src=src_hash, start=round(trim_start, 3), duration=duration and round(duration, 3),
                      height=settings.inference_height, crf=settings.convert_crf, fps=fps_src, v=2)
    mp4 = cache.path("videos", conv_key, ".mp4")
    if not mp4.is_file():
        video.convert(source, mp4, target_height=settings.inference_height, start=trim_start,
                      duration=duration, crf=settings.convert_crf, fps=fps_src,
                      progress=lambda f: progress("convert", f, "Converting video"), cancel_check=cancelled)
    check()
    info = video.inspect(mp4)
    fps = info.fps
    n = info.n_frames
    offset = int(round(trim_start * fps))   # converted frame k == source frame offset + k
    if n < fps * 2:
        raise RuntimeError("The clip is shorter than 2 seconds after trimming; nothing to analyze.")
    video_hash = file_sha256(mp4)

    def cached(kind, parts, call):
        key = key_of(video=video_hash, provider=provider.name, kind=kind, **parts)
        hit = cache.get_json(kind, key)
        if hit is not None:
            return hit["payload"], 0.0, True
        payload, usage = call()
        cache.put_json(kind, key, {"payload": payload, "usage": usage})
        return payload, float((usage or {}).get("cost") or 0), False

    # ── 2. pose ──────────────────────────────────────────────────────────────
    progress("pose", 0.0, "Estimating the shooter's pose (ViTPose)")
    extra_body = {"method": "pose", "video_fps": settings.pose_fps,
                  "precision": gateway.precision, "video_max_frames": n}
    payload, costs["pose"], pose_hit = cached(
        "pose", {"model": gateway.pose_model, "extra_body": extra_body, "contract": POSE_CONTRACT},
        lambda: provider.pose(mp4, model=gateway.pose_model, extra_body=extra_body,
                              ctx={"source_offset": offset, "n_frames": n, "fps": fps}))
    check()
    frames = pose.unwrap(payload)
    people = len({p.get("track_id") for f in frames for p in f["persons"]})
    shooter = pose.pick_shooter(frames) if people > 1 else None
    if shooter is not None:
        pose.keep_track(frames, shooter)
    elif people > 1:
        warnings_list.append(f"{people} people were tracked and none clearly raised both hands "
                             "overhead; analysis follows the largest person.")
    pose.dedupe(frames, max_persons=settings.max_persons, iou=settings.person_iou)
    kp = pose.keypoints(frames, n, info.width, info.height)
    n_posed = int(np.isfinite(kp[:, 5, 0]).sum())
    if n_posed < 0.3 * n:
        warnings_list.append(f"The shooter was posed on only {n_posed} of {n} frames; "
                             "shots may be missed.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spans = shots_mod.candidates(kp, fps, armed_below=settings.shot_armed_below,
                                     min_gap_seconds=settings.shot_min_gap_seconds)
        side = (settings.shooting_hand if settings.shooting_hand in ("left", "right")
                else shots_mod.shooting_side(kp, spans, fps))
    progress("pose", 1.0, f"{len(spans)} candidate shot(s) found")

    # ── 3. hoop ──────────────────────────────────────────────────────────────
    progress("hoop", 0.0, "Finding the hoop (SAM 3.1)")
    hoop_frame = (n // 2 if settings.hoop_frame_seconds is None
                  else min(n - 1, int(settings.hoop_frame_seconds * fps)))
    frame_img = video.read_frame(mp4, hoop_frame)
    ok, png = cv2.imencode(".png", frame_img)
    payload, costs["hoop"], _ = cached(
        "hoop", {"model": gateway.sam_model, "prompt": settings.hoop_prompt, "frame": hoop_frame,
                 "contract": "sam31-hoop-v1"},
        lambda: provider.segment(png.tobytes(), model=gateway.sam_model, prompt=settings.hoop_prompt,
                                 ctx={"source_offset": offset + hoop_frame}))
    hoop = hoop_mod.locate(payload, frame=hoop_frame, width=info.width, height=info.height)
    check()

    # ── 4. ball, around each shot ────────────────────────────────────────────
    raw: dict[int, list] = {}
    windows: list[tuple[int, int]] = []
    calls = 0
    for i, (a, _) in enumerate(spans, 1):
        progress("ball", (i - 1) / max(1, len(spans)), f"Tracking the ball: shot {i} of {len(spans)}")
        start = max(0, a - int(settings.ball_before_seconds * fps))
        end = min(n, a + int(settings.ball_after_seconds * fps))
        if windows and start < sum(windows[-1]):
            start = sum(windows[-1])   # never pay twice for a frame
        for _ext in range(settings.ball_max_extensions + 1):
            length = min(end - start, TRACK_MAX_FRAMES * settings.ball_stride)
            if length <= 0:
                break
            check()
            seg = tmp / f"win{start}_{length}.mp4"

            def call(start=start, length=length, seg=seg):
                video.segment(mp4, seg, start_frame=start, n_frames=length, fps=fps)
                try:
                    return provider.track(seg, model=gateway.sam_model, prompt=settings.ball_prompt,
                                          stride=settings.ball_stride,
                                          ctx={"source_offset": offset + start, "n_frames": length,
                                               "fps": fps})
                finally:
                    seg.unlink(missing_ok=True)

            payload, c, _ = cached("ball", {"model": gateway.sam_model, "prompt": settings.ball_prompt,
                                            "start": start, "length": length,
                                            "stride": settings.ball_stride, "contract": "sam31-ball-track-v1"},
                                   call)
            costs["ball"] += c
            calls += 1
            raw.update(ball_mod.detections(payload, frame_offset=start, width=info.width,
                                           height=info.height, min_score=settings.ball_min_score))
            windows.append((start, length))
            track = ball_mod.select(raw, n, windows, size_tolerance=settings.ball_size_tolerance)
            if not shots_mod.ball_unresolved(track.x, track.y, start, start + length, hoop):
                break
            start, end = start + length, min(n, start + length + int(settings.ball_extend_seconds * fps))
    track = ball_mod.select(raw, n, windows, size_tolerance=settings.ball_size_tolerance)
    track.shown = ball_mod.select(raw, n, windows, local_size=True,
                                  size_tolerance=settings.ball_draw_size_tolerance)
    if spans and not np.isfinite(track.diameter):
        warnings_list.append("SAM never returned a ball-sized detection; every shot is unknown.")
    progress("ball", 1.0, "Ball tracked")
    check()

    # ── 5. shots ─────────────────────────────────────────────────────────────
    progress("analyze", 0.0, "Judging shots")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shot_list = analyze(kp, fps, n, info.width, spans, hoop, track, side, settings)
        angle = shots_mod.elbow_angle(kp, side, min_projected=settings.elbow_min_projected)
        side_on = shots_mod.shoulder_ratio(kp) < settings.side_on_max_shoulder_ratio
    stats = shots_mod.tally(shot_list)
    if not shot_list:
        warnings_list.append("No shots found: a shot needs both wrists above the head, "
                             "after both were below the shoulders.")
    if stats["unknown"]:
        warnings_list.append(f"{stats['unknown']} shot(s) could not be judged from the ball track; "
                             "they are excluded from FG%.")
    if not any(s.side_on for s in shot_list) and shot_list:
        warnings_list.append("No side-on shots: projected elbow angles are not reported.")
    view = court.CourtView(10, 10, three_radius=settings.three_point_radius_m,
                           ft_distance=settings.free_throw_distance_m)
    check()

    # ── 6. render and export ─────────────────────────────────────────────────
    progress("render", 0.0, "Rendering the annotated video")
    raw_mp4 = tmp / "_raw.mp4"
    render.render(info, kp=kp, hoop=hoop, track=track, shots=shot_list, dst=raw_mp4,
                  settings=settings, stats_fn=shots_mod.tally,
                  progress=lambda f: progress("render", f, "Rendering the annotated video"),
                  cancelled=cancelled)
    check()
    out_video = out_dir / "basketball_annotated.mp4"
    video.encode_h264(raw_mp4, out_video, crf=settings.output_crf)
    raw_mp4.unlink(missing_ok=True)
    progress("export", 0.0, "Writing charts and exports")
    render.court_chart(shot_list, out_dir / "shot_chart.png", settings=settings, stats=stats)
    render.ball_paths(shot_list, hoop, frame_img, out_dir / "ball_paths.png", settings=settings)

    total = time.perf_counter() - t_start
    meta = {
        "sport": "basketball", "pipeline_version": PIPELINE_VERSION,
        "provider": provider.name, "simulated": provider.simulated,
        "source": {**src, "sha256": src_hash},
        "trim": {"start_s": trim_start, "end_s": trim_end, "source_frame_offset": offset},
        "analysis_video": {"width": info.width, "height": info.height, "fps": fps, "n_frames": n},
        "models": {"pose": gateway.pose_model, "sam": gateway.sam_model},
        "pose_from_cache": pose_hit,
        "frames_with_person": n_posed, "shooting_hand": side,
        "people_tracked": people, "shooter_track_id": shooter,
        "hoop": hoop.to_dict(),
        "ball": {"windows": windows, "calls": calls, "frames_tracked": int(track.covered.sum()),
                 "frames_found": int(track.found.sum()),
                 "diameter_px": None if not np.isfinite(track.diameter) else round(track.diameter, 1)},
        "settings": settings.as_dict(),
        "cost_usd": {k: round(v, 6) for k, v in {**costs, "total": sum(costs.values())}.items()},
        "seconds": round(total, 2),
    }
    files = export.write(shot_list, angle, side_on, track, fps, offset, out_dir, stats=stats,
                         meta=meta, warnings=warnings_list, is_three=view.is_three)
    shutil_rmtree(tmp)
    progress("export", 1.0, "Done")
    return {"sport": "basketball", "stats": stats, "warnings": warnings_list,
            "files": ["basketball_annotated.mp4", "shot_chart.png", "ball_paths.png", *files],
            "shots": [s.summary(fps, offset) | {"three_point_estimate": (view.is_three(*s.court_xy)
                                                                          if s.court_xy else None)}
                      for s in shot_list],
            "meta": meta}


def analyze(kp, fps, n, width, spans, hoop, track, side, s: Settings) -> list[shots_mod.Shot]:
    angle = shots_mod.elbow_angle(kp, side, min_projected=s.elbow_min_projected)
    wrist_y = kp[:, shots_mod.J[f"{side}_wrist"], 1]
    ratio = shots_mod.shoulder_ratio(kp)
    bx, by = ball_mod.smoothed(track, max_gap=int(s.ball_fill_seconds * fps))
    shown = track.shown or track
    size = np.where(np.isfinite(shown.d), shown.d, track.diameter)
    wrists = kp[:, [shots_mod.J["left_wrist"], shots_mod.J["right_wrist"]], :2]
    out = []
    for i, (a, b) in enumerate(spans, 1):
        nxt = spans[i][0] if i < len(spans) else n
        release, method = shots_mod.find_release(a, b, track.y, wrist_y, track.diameter, fps)
        lo = max(0, a - int(s.set_window_seconds * fps))
        before = angle[lo:release + 1]
        set_frame = lo + int(np.nanargmin(before)) if np.isfinite(before).any() else a
        after = angle[release:min(len(angle), release + int(0.2 * fps)) + 1]
        r = ratio[lo:release + 1]
        side_on = bool(np.isfinite(r).any() and np.nanmedian(r) < s.side_on_max_shoulder_ratio)
        nan = float("nan")
        shot = shots_mod.Shot(
            number=i, start=a, end=b, release=release, set_frame=set_frame, side_on=side_on,
            release_method=method,
            elbow_set=float(angle[set_frame]) if side_on else nan,
            elbow_release=float(np.nanmax(after)) if side_on and np.isfinite(after).any() else nan)
        shot.motion = shots_mod.motion_start(a, bx, by, size, wrists, fps,
                                             lookback=s.ball_before_seconds, reach=s.held_reach)
        covered_until = release
        while covered_until < n and track.covered[covered_until]:
            covered_until += 1
        shots_mod.judge(shot, track.x, track.y, track.d, hoop, fps, until=min(nxt, covered_until),
                        inner_fraction=s.rim_inner_fraction, net_min_seconds=s.net_min_seconds,
                        net_side_tolerance=s.net_side_tolerance,
                        rebound_margin=s.rebound_margin_rims,
                        max_gap=int(s.rim_gap_seconds * fps),
                        evidence_gap=int(s.evidence_gap_seconds * fps),
                        min_flight=int(s.min_flight_seconds * fps),
                        ball_to_rim=s.ball_diameter_m / s.rim_diameter_m,
                        front_size=s.front_of_rim_size)
        where = shots_mod.stance(kp, a, fps)
        if where is not None:
            shot.feet_px, shot.body_px = where
            shot.court_xy = court.project(
                shot.feet_px[0], shot.body_px, hoop, frame_width=width,
                hfov_degrees=s.camera_hfov_degrees, player_height_m=s.player_height_m,
                rim_diameter_m=s.rim_diameter_m, yaw_degrees=s.court_yaw_degrees)
            shot.distance_m = float(np.hypot(*shot.court_xy))
        out.append(shot)
    return out


def shutil_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def summary_json(result: dict) -> str:
    return json.dumps(result, indent=2)

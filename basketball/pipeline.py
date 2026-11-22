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
from .src import ball as ball_mod, net_search
from .src import court, export, hoop as hoop_mod, pose, render, scenes, shots as shots_mod, video
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
    diffs = scenes.frame_differences(mp4)
    cuts = scenes.cuts_from_differences(diffs, threshold=settings.scene_cut_threshold,
                                        min_gap=max(1, int(0.5 * fps)))
    if cuts:
        warnings_list.append(f"{len(cuts)} camera cut(s) detected; the hoop and the shooter are "
                             "located separately in each camera segment, and court spots from "
                             "different camera angles are not comparable.")

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
    # Every person's head, before the shooter is chosen: SAM can take a head for the ball.
    head_map = pose.heads(frames, info.width, info.height)
    people = len({p.get("track_id") for f in frames for p in f["persons"]})
    camera = scenes.segments(n, cuts)
    shooter = pose.select_shooter(frames, camera) if people > 1 else None
    if shooter and shooter["unresolved_segments"]:
        warnings_list.append(f"{people} people were tracked and in {shooter['unresolved_segments']} "
                             "camera segment(s) none clearly raised both hands overhead; there the "
                             "analysis follows the largest person.")
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

    # ── 3. hoop, per camera segment ──────────────────────────────────────────
    progress("hoop", 0.0, "Finding the hoop (SAM 3.1)")
    timeline, costs["hoop"], frame_img = locate_hoops(
        mp4, n, fps, cuts, settings=settings, gateway=gateway, provider=provider,
        cached=cached, offset=offset, width=info.width, height=info.height,
        warnings_list=warnings_list, progress=progress)
    check()

    # ── 4. ball, around each shot ────────────────────────────────────────────
    pick = {"size_tolerance": settings.ball_size_tolerance, "min_score": settings.ball_min_score,
            "blur_rescue": settings.ball_blur_rescue, "max_stretch": settings.ball_rescue_max_stretch,
            "fps": fps, "static_seconds": settings.ball_static_seconds or None,
            "max_speed_mps": settings.ball_max_speed_mps or None}
    detect_min = (min(settings.ball_min_score, settings.ball_rescue_min_score)
                  if settings.ball_blur_rescue else settings.ball_min_score)
    raw: dict[int, list] = {}
    all_boxes: dict[int, list] = {}
    windows: list[tuple[int, int]] = []
    calls = 0
    for i, (a, _) in enumerate(spans, 1):
        progress("ball", (i - 1) / max(1, len(spans)), f"Tracking the ball: shot {i} of {len(spans)}")
        start = max(0, a - int(settings.ball_before_seconds * fps))
        end = min(n, a + int(settings.ball_after_seconds * fps))
        if windows and start < sum(windows[-1]):
            # A quick second shot keeps (most of) its own lead-in: SAM must see the
            # ball in the hands before the release, not start tracking at it. At
            # most ball_max_overlap_seconds are tracked twice (was: never).
            start = max(start, sum(windows[-1]) - int(settings.ball_max_overlap_seconds * fps))
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
            found = ball_mod.detections(payload, frame_offset=start, width=info.width,
                                        height=info.height, min_score=detect_min)
            for f, boxes in found.items():   # merged: overlapping windows add, never replace
                raw.setdefault(f, []).extend(boxes)
            # Every box SAM returned, any score: exported for diagnosis (basketball.diagnose).
            for f, boxes in ball_mod.detections(payload, frame_offset=start, width=info.width,
                                                height=info.height, min_score=0.0).items():
                all_boxes.setdefault(f, []).extend((*b, start) for b in boxes)
            windows.append((start, length))
            track = ball_mod.select(raw, n, windows, **pick)
            at_rim = timeline.at(start)
            if at_rim is None or not shots_mod.ball_unresolved(track.x, track.y, start,
                                                               start + length, at_rim):
                break
            start, end = start + length, min(n, start + length + int(settings.ball_extend_seconds * fps))
    on_head = _head_boxes(raw, head_map, mp4, fps, settings)
    track = ball_mod.select(raw, n, windows, exclude=on_head, **pick)
    track.shown = ball_mod.select(raw, n, windows, local_size=True, exclude=on_head,
                                  **{**pick, "size_tolerance": settings.ball_draw_size_tolerance})
    if on_head:
        warnings_list.append(
            f"{len(on_head)} ball detection(s) sat on a person's head and were not used (SAM's "
            "\"basketball\" can match a round head). They are in ball_detections.csv as \"on head\".")
    rescued = int(track.blurred.sum()) if track.blurred is not None else 0
    if spans and not np.isfinite(track.diameter):
        warnings_list.append("SAM never returned a ball-sized detection; every shot is unknown.")
    progress("ball", 1.0, "Ball tracked")
    check()
    if settings.hoop_choice == "auto":
        rechoose_by_ball(timeline, track, warnings_list)

    # ── 5. shots ─────────────────────────────────────────────────────────────
    progress("analyze", 0.0, "Judging shots")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        not_shots: list[dict] = []
        shot_list = analyze(kp, fps, n, info.width, spans, timeline, track, side, settings,
                            not_shots=not_shots, offset=offset,
                            read_crops=lambda start, count, box: video.read_crops(mp4, start, count, box),
                            height=info.height)
        angle = shots_mod.elbow_angle(kp, side, min_projected=settings.elbow_min_projected)
        side_on = shots_mod.shoulder_ratio(kp) < settings.side_on_max_shoulder_ratio
    stats = shots_mod.tally(shot_list)
    if not_shots:
        warnings_list.append(
            f"{len(not_shots)} candidate(s) were not shot attempts (hands up, but the ball went away "
            "from the hoop and never above the rim): at "
            + ", ".join(f"{d['hands_up_s']:.1f}s" for d in not_shots)
            + ". They are left out of attempts and FG%.")
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
    render.render(info, kp=kp, hoop_at=timeline.at, track=track, shots=shot_list, dst=raw_mp4,
                  settings=settings, stats_fn=shots_mod.tally,
                  progress=lambda f: progress("render", f, "Rendering the annotated video"),
                  cancelled=cancelled)
    check()
    out_video = out_dir / "basketball_annotated.mp4"
    video.encode_h264(raw_mp4, out_video, crf=settings.output_crf)
    raw_mp4.unlink(missing_ok=True)
    progress("export", 0.0, "Writing charts and exports")
    render.court_chart(shot_list, out_dir / "shot_chart.png", settings=settings, stats=stats)
    main_seg = max(timeline.segments, key=lambda sg: (
        sum(1 for sh in shot_list if sg.start <= sh.start < sg.end), sg.hoop is not None))
    if main_seg.hoop is not None:
        render.ball_paths([sh for sh in shot_list if main_seg.start <= sh.start < main_seg.end],
                          main_seg.hoop, frame_img.get(main_seg.hoop.frame), out_dir / "ball_paths.png",
                          settings=settings)
    checked = [sg.hoop for sg in timeline.segments if sg.hoop is not None]
    render.hoop_check(checked, frame_img, out_dir / "hoop_check.png")

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
        "people_tracked": people,
        "shooter": shooter,
        "camera_cuts": [c + offset for c in cuts],
        "hoops": [sg.to_dict() for sg in timeline.segments],
        "ball": {"windows": windows, "calls": calls, "frames_tracked": int(track.covered.sum()),
                 "frames_rescued_from_blur": rescued,
                 "frames_found": int(track.found.sum()),
                 "diameter_px": None if not np.isfinite(track.diameter) else round(track.diameter, 1)},
        "settings": settings.as_dict(),
        "cost_usd": {k: round(v, 6) for k, v in {**costs, "total": sum(costs.values())}.items()},
        "seconds": round(total, 2),
    }
    meta["not_shots"] = not_shots
    files = export.write(shot_list, angle, side_on, track, fps, offset, out_dir, stats=stats,
                         meta=meta, warnings=warnings_list, is_three=view.is_three)
    files.append(export.write_detections(all_boxes, track, fps, offset, out_dir,
                                         min_score=settings.ball_min_score,
                                         rescue_min=settings.ball_rescue_min_score))
    shutil_rmtree(tmp)
    progress("export", 1.0, "Done")
    return {"sport": "basketball", "stats": stats, "warnings": warnings_list,
            "files": ["basketball_annotated.mp4", "shot_chart.png", "ball_paths.png", "hoop_check.png",
                      *files],
            "not_shots": not_shots,
            "shots": [s.summary(fps, offset) | {"three_point_estimate": (view.is_three(*s.court_xy)
                                                                          if s.court_xy else None)}
                      for s in shot_list],
            "meta": meta}


LOST_AT_RIM = ("ball lost at the rim", "ball lost while still above the rim",
               "ball hidden while crossing the rim", "crossed inside the rim")   # prefixes


def analyze(kp, fps, n, width, spans, timeline, track, side, s: Settings, *,
            not_shots: list | None = None, offset: int = 0, read_crops=None,
            height: int | None = None) -> list[shots_mod.Shot]:
    """Every shot, judged against the hoop of the camera segment it starts in.

    Candidates that are clearly not shot attempts (``shots.not_a_shot``) are left
    out and, when *not_shots* is given, recorded there; the rest are numbered 1..n.

    When a shot comes out unknown because the ball was lost at the rim and
    *read_crops(start, count, box)* is given, the ball is looked for in the net
    (``net_search``); sightings found on its continuation are added to the track
    (``track.recovered``) and the shot is judged again by the same rules.
    """
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
        seg_i = timeline.segment_index(a)
        shot.segment = seg_i
        hoop = timeline.at(a)
        if hoop is None:
            shot.made, shot.reason = None, "no hoop located in this camera segment"
            shot.evidence = ["SAM found no hoop in this part of the clip"]
            out.append(shot)
            continue
        possible_non_shot = None
        if s.shot_attempt_check:
            action, found = shots_mod.attempt_check(
                shot, track.x, track.y, hoop, fps, window_s=s.attempt_window_seconds,
                min_obs=s.attempt_min_observations, recede_rims=s.attempt_recede_rims,
                early_s=s.attempt_early_seconds, gap_s=s.evidence_gap_seconds)
            if action == "drop":
                if not_shots is not None:
                    not_shots.append({"hands_up_s": round((a + offset) / fps, 3),
                                      "release_s": round((release + offset) / fps, 3),
                                      "reason": found[0], "evidence": found[1]})
                continue
            possible_non_shot = found if action == "flag" else None
        # Ball positions after a camera cut are in another view: stop the evidence there.
        seg_end = timeline.segments[seg_i].end if seg_i is not None else n
        judge_args = dict(until=min(nxt, covered_until, seg_end),
                          inner_fraction=s.rim_inner_fraction, net_min_seconds=s.net_min_seconds,
                          net_side_tolerance=s.net_side_tolerance,
                          rebound_margin=s.rebound_margin_rims,
                          max_gap=int(s.rim_gap_seconds * fps),
                          evidence_gap=int(s.evidence_gap_seconds * fps),
                          min_flight=int(s.min_flight_seconds * fps),
                          ball_to_rim=s.ball_diameter_m / s.rim_diameter_m,
                          front_size=s.front_of_rim_size, rim_diameter_m=s.rim_diameter_m,
                          fall_speed_tolerance=s.fall_speed_tolerance,
                          net_slow_factor=s.net_slow_factor)
        shots_mod.judge(shot, track.x, track.y, track.d, hoop, fps, **judge_args)
        if (s.ball_net_search and read_crops is not None and shot.made is None
                and shot.reason.startswith(LOST_AT_RIM)):
            # The search reads the video itself (no model call), so it is not limited to the
            # frames SAM tracked: a ball tracking window can end just after the ball enters
            # the net (128 frames is 2.1 s at 60 fps) while the ball is still falling out of it.
            search_until = min(nxt, seg_end)
            found = _search_net(shot, track, hoop, fps, n, width, height, s, read_crops,
                                until=judge_args["until"], fill_until=search_until)
            if found:
                before = f"{shot.result} ({shot.reason})"
                judge_args["until"] = max(judge_args["until"], min(search_until, found[-1] + 1))
                shots_mod.judge(shot, track.x, track.y, track.d, hoop, fps, **judge_args)
                shot.evidence = list(shot.evidence) + [
                    f"ball found in {len(found)} frame(s) around the net by a local colour search "
                    f"({found[0] / fps:.2f}-{found[-1] / fps:.2f}s) after SAM lost it; "
                    f"from SAM's track alone: {before}"]
        if possible_non_shot is not None:
            shots_mod.flag_possible_non_shot(shot, possible_non_shot)
        where = shots_mod.stance(kp, a, fps)
        if where is not None:
            shot.feet_px, shot.body_px = where
            shot.court_xy = court.project(
                shot.feet_px[0], shot.body_px, hoop, frame_width=width,
                hfov_degrees=s.camera_hfov_degrees, player_height_m=s.player_height_m,
                rim_diameter_m=s.rim_diameter_m, yaw_degrees=s.court_yaw_degrees)
            shot.distance_m = float(np.hypot(*shot.court_xy))
        out.append(shot)
    for number, shot in enumerate(out, 1):
        shot.number = number
    return out


def _head_boxes(raw: dict, head_map: dict, mp4: Path, fps: float, s: Settings) -> set:
    """Ball boxes on a person's head (``ball.head_boxes``), with the colour vote read from
    the frames of the boxes that sit on a head."""
    if not s.ball_head_filter or not head_map:
        return set()
    geometric = ball_mod.head_boxes(raw, head_map, fps, reach=s.ball_head_reach,
                                    min_speed=s.ball_head_min_speed)
    if not geometric:
        return set()
    images = video.read_frames(mp4, {f for f, _, _ in geometric})

    def colour(f, cx, cy, size):
        return ball_mod.ball_colour(images.get(f), cx, cy, size)

    return ball_mod.head_boxes(raw, head_map, fps, reach=s.ball_head_reach,
                               min_speed=s.ball_head_min_speed, colour=colour)


def _search_net(shot, track, hoop, fps, n, width, height, s: Settings, read_crops, *,
                until: int, fill_until: int | None = None) -> list[int]:
    """Look for the ball in the net after the last sighting on the shot's path (see
    ``net_search``). Adds what it finds to *track* in place; returns those frames.

    *until* bounds the SAM sightings that count as the shot's path; *fill_until* (default
    *until*) bounds the frames the search may add, which can go past the SAM window."""
    limit = until if fill_until is None else max(until, fill_until)
    seen = [f for f, *_ in shot.path if f < until]
    near = [f for f in seen if abs(track.x[f] - hoop.cx) < 1.5 * hoop.width
            and hoop.cy - 1.5 * hoop.width <= track.y[f] <= hoop.net_bottom + hoop.width]
    if not near or height is None:
        return []
    last = near[-1]
    diameter = float(track.d[last]) if np.isfinite(track.d[last]) else float(track.diameter)
    if not np.isfinite(diameter):
        return []
    box = net_search.crop_box(hoop, width, height)
    count = min(int(round(s.net_search_seconds * fps)), n - last - 1)
    if count < 5:
        return []
    crops = read_crops(last + 1, count, box)
    if len(crops) < 5:
        return []
    path = net_search.follow(net_search.blobs(crops), last + 1,
                             (last, float(track.x[last]), float(track.y[last])), hoop, diameter, fps,
                             max_speed_mps=s.ball_max_speed_mps or 12.0,
                             max_gap_seconds=s.evidence_gap_seconds, offset=box[:2])
    if track.recovered is None:
        track.recovered = np.zeros(len(track.x), bool)
    added = []
    if path:
        # While the ball is being followed through the net, a SAM sighting far off that
        # path is something else (another ball, a head): it would break the evidence.
        frames_on = np.array([p[0] for p in path])
        for f in range(path[0][0], min(path[-1][0] + 1, limit)):
            if not np.isfinite(track.x[f]):
                continue
            j = int(np.argmin(np.abs(frames_on - f)))
            if np.hypot(track.x[f] - path[j][1], track.y[f] - path[j][2]) > 2 * diameter:
                track.gated.add(ball_mod.key(f, track.x[f], track.y[f]))
                track.x[f] = track.y[f] = track.d[f] = np.nan
    for f, x, y in path:
        if f < limit and not np.isfinite(track.x[f]):
            # No size: a blob behind the net strings does not measure the ball, and the
            # judge's "in front of the rim" check must read SAM's boxes only.
            track.x[f], track.y[f], track.d[f] = x, y, np.nan
            track.recovered[f] = True
            added.append(f)
    return added


def locate_hoops(mp4: Path, n: int, fps: float, cuts: list[int], *, settings: Settings, gateway,
                 provider, cached, offset: int, width: int, height: int, warnings_list: list,
                 progress=_noop):
    """SAM on one frame per camera segment (and per ``hoop_resample_seconds`` within a long
    one). Returns ``(HoopTimeline, cost, {frame: image})``.

    Replaces one SAM call on the middle frame, which on footage with camera cuts
    located a hoop that was only right for part of the clip.
    """
    if settings.hoop_frame_seconds is not None:
        spans = [(0, n)]
        probe = {(0, n): min(n - 1, int(settings.hoop_frame_seconds * fps))}
    else:
        spans = scenes.segments(n, cuts, max_len=max(1, int(settings.hoop_resample_seconds * fps)))
        probe = {sp: (sp[0] + sp[1]) // 2 for sp in spans}
    min_len = int(settings.min_hoop_segment_seconds * fps)
    eligible = [sp for sp in spans if sp[1] - sp[0] >= min_len] or [max(spans, key=lambda sp: sp[1] - sp[0])]
    if len(eligible) > settings.max_hoop_segments:
        eligible = sorted(sorted(eligible, key=lambda sp: sp[0] - sp[1])[:settings.max_hoop_segments])
        warnings_list.append(f"More than {settings.max_hoop_segments} camera segments: the hoop "
                             "was located only in the longest ones; shots elsewhere are unknown.")
    images, segments, cost = {}, [], 0.0
    for i, sp in enumerate(spans):
        if sp not in eligible:
            segments.append(hoop_mod.HoopSegment(sp[0], sp[1], None, [], "segment too short"))
            continue
        progress("hoop", i / len(spans), f"Finding the hoop (SAM 3.1), segment {i + 1} of {len(spans)}")
        frame = probe[sp]
        img = video.read_frame(mp4, frame)
        images[frame] = img
        ok, png = cv2.imencode(".png", img)
        payload, c, _ = cached(
            "hoop", {"model": gateway.sam_model, "prompt": settings.hoop_prompt, "frame": frame,
                     "contract": hoop_mod.CONTRACT},
            lambda png=png, frame=frame: provider.segment(
                png.tobytes(), model=gateway.sam_model, prompt=settings.hoop_prompt,
                ctx={"source_offset": offset + frame}))
        cost += c
        found = hoop_mod.candidates(payload, frame=frame, width=width, height=height)
        if not found:
            segments.append(hoop_mod.HoopSegment(sp[0], sp[1], None, [], "SAM found no hoop"))
            continue
        chosen, why = hoop_mod.choose(found, settings.hoop_choice)
        segments.append(hoop_mod.HoopSegment(sp[0], sp[1], chosen, found, why))
    timeline = hoop_mod.HoopTimeline(segments)
    if not timeline.hoops():
        raise RuntimeError("SAM found no hoop in the clip. Is the hoop (rim and net) in shot?")
    _hoop_warnings(timeline, cuts, settings, warnings_list)
    return timeline, cost, images


def rechoose_by_ball(timeline, track, warnings_list: list) -> None:
    """"auto" hoop choice, second pass: where several hoops were found, switch to the
    one the ball is actually seen coming down through, if that is not the chosen one."""
    for seg in timeline.segments:
        if seg.hoop is None or len(seg.candidates) < 2:
            continue
        better, switched = hoop_mod.prefer_by_ball(seg.candidates, seg.hoop, track.x, track.y,
                                                   range(seg.start, seg.end))
        if switched:
            seg.hoop, seg.why = better, "the hoop the ball comes down through"
            warnings_list.append(
                f"Switched to the hoop at ({better.cx:.0f}, {better.cy:.0f}) px "
                f"({better.width:.0f} px wide) for frames {seg.start}-{seg.end}: the ball is "
                "seen dropping through it.")


def _hoop_warnings(timeline, cuts, settings: Settings, warnings_list: list) -> None:
    multi = [sg for sg in timeline.segments if len(sg.candidates) > 1]
    if multi:
        sg = multi[0]
        h = sg.hoop
        warnings_list.append(
            f"SAM found more than one hoop ({max(len(m.candidates) for m in multi)} at most, in "
            f"{len(multi)} segment(s)). Using the one at ({h.cx:.0f}, {h.cy:.0f}) px, "
            f"{h.width:.0f} px wide ({sg.why}). If that is the wrong hoop, set \"Hoop\" in "
            "the settings.")
    # The rim is read from the top of SAM's mask; say when the mask holds more than the hoop.
    noted = [sg for sg in timeline.segments if sg.hoop is not None and sg.hoop.notes]
    if noted:
        h = noted[0].hoop
        warnings_list.append(
            "The hoop mask looks contaminated (" + "; ".join(h.notes) + "). The rim was read from "
            f"the top of the mask: center ({h.cx:.0f}, {h.cy:.0f}) px, {h.width:.0f} px wide. "
            "Check hoop_check.png: every make/miss call depends on that rim line.")
    # Within one camera segment (no cut between), the hoop should not move.
    segs = timeline.segments
    for a, b in zip(segs, segs[1:]):
        if a.hoop is None or b.hoop is None or any(a.start < c <= b.start for c in cuts):
            continue
        moved = np.hypot(a.hoop.cx - b.hoop.cx, a.hoop.cy - b.hoop.cy)
        w = max(a.hoop.width, b.hoop.width)
        if moved > settings.hoop_disagree_rims * w:
            warnings_list.append(
                f"The hoop moved between frames {a.start} and {b.end} with no camera cut: the "
                "camera may be moving, which makes make/miss calls less reliable.")
            break
        if moved > settings.hoop_rim_unstable_rims * w:
            warnings_list.append(
                f"The rim estimate moved {moved:.0f} px ({moved / w:.2f} rim widths) between frames "
                f"{a.start} and {b.end} with no camera cut: the rim may be read unreliably. Check "
                "hoop_check.png.")
            break


def shutil_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def summary_json(result: dict) -> str:
    return json.dumps(result, indent=2)

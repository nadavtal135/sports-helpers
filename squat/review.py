"""Descriptive squat feedback from measured landmarks. Every observation has video evidence."""

from dataclasses import dataclass
import math

DEPTH = {"above": "Hip above knee", "roughly level": "Hip roughly level with knee", "below": "Hip below knee"}


@dataclass(frozen=True)
class Measure:
    field: str
    phase: str
    name: str
    unit: str
    repeat_band: float     # display resolution for a within-set comparison


MEASURES = {
    "depth": Measure("depth_2d", "bottom", "visible depth", "", 0),
    "lean": Measure("torso_lean_at_bottom_deg", "bottom", "torso lean at the bottom", "°", 5),
    "descent": Measure("descent_s", "descent", "descent time", " s", 0.15),
    "ascent": Measure("ascent_s", "ascent", "ascent time", " s", 0.15),
    "knee": Measure("peak_knee_flexion_projected_deg", "peak_flexion", "peak projected knee flexion", "°", 7),
}
SPATIAL = {"depth", "lean", "knee"}


def _value(row, key):
    value = row.get(MEASURES[key].field)
    return None if value == "unknown" else value


def _display(key, value):
    if value is None:
        return "Unmeasured"
    if key == "depth":
        return DEPTH[value]
    unit = MEASURES[key].unit
    decimals = 2 if unit == " s" else 1
    return f"{value:.{decimals}f}{unit}"


def _cue(row, phase, frame, measurement):
    return {"attempt": row.get("attempt"), "rep": row.get("rep"), "phase": phase,
            "source_frame": frame, "measurement": measurement}


def _evidence(row, key):
    measure = MEASURES[key]
    value = _display(key, _value(row, key))
    label = value if key == "depth" else f"{measure.name.capitalize()} {value}"
    return _cue(row, measure.phase, row[f"source_frame_{measure.phase}"], label)


def _rep_list(rows):
    nums = sorted(r["rep"] for r in rows)
    if len(nums) == 1:
        return f"rep {nums[0]}"
    if len(nums) > 2 and nums[-1] - nums[0] == len(nums) - 1:
        return f"reps {nums[0]}–{nums[-1]}"
    if len(nums) > 4:
        return f"{len(nums)} measured reps"
    return "reps " + ", ".join(map(str, nums[:-1])) + f" and {nums[-1]}"


def _note(kind, title, detail, evidence):
    return {"kind": kind, "title": title, "detail": detail, "evidence": evidence}


def _consistency(available):
    # Prefer the observation supported by the most reps. A repeated depth
    # category describes that band only, regardless of any intended target.
    best, count = None, 1
    for key, rows in available.items():
        if key == "depth":
            groups = [[r for r in rows if _value(r, key) == band] for band in DEPTH]
            group = max(groups, key=len)
        else:
            group = rows
        if len(group) <= count:
            continue
        values = [_value(r, key) for r in group]
        lo, hi = min(values), max(values)
        if key == "depth":
            title = f"You reached the same visible depth band on {_rep_list(group)}."
            detail = f"{DEPTH[lo]} at each of those bottoms."
        elif hi - lo <= MEASURES[key].repeat_band + 1e-6:
            title = f"Your {MEASURES[key].name} was similar across {_rep_list(group)}."
            reference = " from vertical" if key == "lean" else ""
            detail = (f"Each measured {_display(key, lo)}{reference}." if lo == hi else
                      f"Measurements ranged from {_display(key, lo)} to {_display(key, hi)}{reference}.")
        else:
            continue
        count = len(group)
        best = _note("Consistency", title, detail, [_evidence(r, key) for r in group])
    return best


def _variation(available):
    rows = available["depth"]
    if len(rows) >= 2:
        order = list(DEPTH)

        def position(r):
            return order.index(_value(r, "depth")), r.get("bottom_hip_minus_knee_leg_lengths") or 0

        shallow, deep = min(rows, key=position), max(rows, key=position)
        if _value(shallow, "depth") != _value(deep, "depth"):
            # Avoid featuring small changes immediately either side of a band boundary.
            near = shallow.get("bottom_hip_minus_knee_leg_lengths")
            references = []
            for r in rows:
                if _value(r, "depth") == _value(deep, "depth"):
                    far = r.get("bottom_hip_minus_knee_leg_lengths")
                    separated = (far - near >= 0.08 if near is not None and far is not None
                                 else position(r)[0] - position(shallow)[0] == 2)
                    if separated:
                        references.append(r)
            if references:
                shallow_band, deep_band = _value(shallow, "depth"), _value(deep, "depth")
                positions = {"above": "above the knee", "roughly level": "roughly level with the knee",
                             "below": "below the knee"}
                return _note("Depth comparison", f"Rep {shallow['rep']} was shallower than {_rep_list(references)}.",
                             f"In the 2D view, your hip was {positions[shallow_band]} on rep {shallow['rep']} "
                             f"and {positions[deep_band]} on {_rep_list(references)}.",
                             [_evidence(r, "depth") for r in [shallow, *references]])
    for key in ("descent", "ascent", "lean", "knee"):
        rows = available[key]
        if len(rows) < 2:
            continue
        lo, hi = (choose(rows, key=lambda r: _value(r, key)) for choose in (min, max))
        if _value(hi, key) - _value(lo, key) >= 2 * MEASURES[key].repeat_band:
            return _note("Set comparison", f"Your {MEASURES[key].name} varied between reps {lo['rep']} and {hi['rep']}.",
                         f"Rep {lo['rep']} measured {_display(key, _value(lo, key))}. "
                         f"Rep {hi['rep']} measured {_display(key, _value(hi, key))}.",
                         [_evidence(r, key) for r in (hi, lo)])
    return None


def _checks(row, goals, tolerance, spatial_ok):
    checks = []
    for key, target in goals.items():
        actual = _value(row, key)
        phase = MEASURES[key].phase
        frame = row.get(f"source_frame_{phase}")
        status, difference = "unknown", "Measurement unavailable"
        if row["status"] != "completed":
            difference = "Full cycle unavailable"
        elif not spatial_ok and key in SPATIAL:
            difference = "Camera view makes this position uncertain"
        elif actual is not None and frame is not None:
            if key == "depth":
                met = actual == target
                difference = "Requested band" if met else f"{DEPTH[actual]} vs target {DEPTH[target].lower()}"
            else:
                delta = actual - target
                met = abs(delta) <= tolerance + 1e-6
                magnitude = "<0.01" if abs(delta) < 0.01 else f"{abs(delta):.2f}"
                difference = (f"{magnitude} s {'longer' if delta > 0 else 'shorter'}"
                              if abs(delta) > 1e-6 else "At target")
            status = "met" if met else "different"
        checks.append({"metric": key, "status": status, "actual": _display(key, actual),
                       "target": _display(key, target), "difference": difference, "source_frame": frame, "phase": phase})
    return checks


def _quality(row, fps):
    bottom = row.get("source_frame_bottom")
    gap = row.get("source_frame_gap")
    phase = "bottom"
    frame = bottom
    if row["status"] == "uncertain":
        phase, frame = ("tracking gap", gap) if gap is not None else ("interval", None)
        detail = "Depth, angles and tempo are uncertain. Keep the near hip, knee and ankle visible throughout the turn."
        if gap is not None:
            detail = "Leg tracking drops out here. " + detail
    elif row["status"] == "incomplete":
        at_start = row.get("source_frame_start") is None
        frame = round(row["review_start_s" if at_start else "review_end_s"] * fps)
        phase = "clip boundary"
        if at_start and row.get("source_frame_end") is None:
            detail = "Both standing landmarks are missing, so full-rep tempo is uncertain."
        elif at_start:
            detail = "The clip misses the standing start, so descent time is unknown."
        else:
            detail = "The clip misses the standing finish, so ascent time is unknown."
        detail += " Include a standing pause at both ends of the trim."
    else:
        missing = []
        if _value(row, "depth") is None:
            missing.append("depth")
        if _value(row, "lean") is None:
            missing.append("bottom torso lean")
        if _value(row, "knee") is None:
            missing.append("peak knee flexion")
        if not missing:
            return None
        if missing == ["peak knee flexion"] and row.get("source_frame_peak_flexion") is not None:
            frame, phase = row["source_frame_peak_flexion"], "knee visibility"
        detail = (f"Uncertain measurements include {', '.join(missing)}. Film directly from the side "
                  "with the near shoulder, hip, knee and ankle clear of equipment.")
    if frame is None:
        frame = round(row["review_start_s"] * fps)
        phase = "interval"
    return _note("Visibility", f"Check the footage for attempt {row['attempt']}.", detail,
                 [_cue(row, phase, frame, row["reason"] or "Limited joint visibility")])


def legacy_landmarks(report):
    """Older saved jobs have timed phase intervals but no sampled phase or peak frames."""
    # These warnings predate per-frame capture evidence. Open the trim start
    # for context instead of implying that the triggering frame was recorded.
    if "capture" not in report:
        for warning in report.get("warnings", []):
            if warning.startswith("The ankle shifts substantially in the image."):
                message = "Camera-relative ankle movement makes depth and angle comparisons uncertain. Keep the camera still and level."
            elif warning.startswith("No leg could be tracked."):
                message = "Depth, angles and tempo are unavailable. Film from the side with the near hip, knee and ankle in frame."
            else:
                continue
            report["capture"] = {"message": message, "source_frame": report["meta"]["trim"]["source_frame_offset"],
                                 "phase": "clip start"}
            break
    for row in report["reps"]:
        for phase, start, end in (("descent", "start", "bottom"), ("ascent", "bottom", "end")):
            a, b = row.get(f"source_frame_{start}"), row.get(f"source_frame_{end}")
            if f"source_frame_{phase}" not in row and a is not None and b is not None:
                row[f"source_frame_{phase}"] = (a + b) // 2


def for_set(report, *, depth: str | None = None, descent: float | None = None,
            ascent: float | None = None, tolerance: float = 0.2):
    """Compare only completed, measurable reps. Targets are optional personal choices."""
    if depth is not None and depth not in DEPTH:
        raise ValueError("Choose above, roughly level or below for the depth target.")
    if any(not math.isfinite(v) or v <= 0 for v in (descent, ascent) if v is not None):
        raise ValueError("Tempo targets must be positive, finite durations in seconds.")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tempo tolerance must be a finite, nonnegative duration.")
    goals = {k: v for k, v in (("depth", depth), ("descent", descent), ("ascent", ascent)) if v is not None}
    rows = report["reps"]
    completed = [r for r in rows if r["status"] == "completed"]
    capture = report.get("capture")
    fps = report["meta"]["analysis_video"]["fps"]
    available = {key: [r for r in completed if _value(r, key) is not None
                       and r.get(f"source_frame_{m.phase}") is not None and (not capture or key not in SPATIAL)]
                 for key, m in MEASURES.items()}
    checks = {str(r["attempt"]): _checks(r, goals, tolerance, not capture) for r in rows}
    issues = {str(r["attempt"]): issue for r in rows if (issue := _quality(r, fps))}
    strength = _consistency(available)
    closer = None
    for key in goals:
        measured = [(r, c) for r in available[key] for c in checks[str(r["attempt"])] if c["metric"] == key]
        matches = [r for r, c in measured if c["status"] == "met"]
        if matches and strength is None:
            strength = _note("Your target", f"You met your {key} target on {_rep_list(matches)}.",
                             f"Your target was {_display(key, goals[key])}.", [_evidence(r, key) for r in matches])
        miss = next(((r, c) for r, c in measured if c["status"] == "different"), None)
        if miss and closer is None:
            different, check = miss
            detail = f"{check['actual']}. Your target was {check['target'].lower()}."
            if key != "depth":
                detail = (f"The {key} took {check['actual']}, {check['difference']} than your target. "
                          f"Aim for {check['target']} (±{tolerance:g} s) to match your chosen tempo.")
            else:
                detail += " Use the bottom frame to check your intended hip position."
            evidence = [_evidence(different, key)] + [_evidence(r, key) for r in matches[:1]]
            closer = _note("Your target", f"Rep {different['rep']} differed from your {key} target.", detail, evidence)

    if capture:
        closer = _note("Camera view", "Check the camera view before comparing positions.", capture["message"],
                       [_cue({}, capture.get("phase", "camera view"), capture["source_frame"], "Camera view")])
    closer = closer or _variation(available) or next(iter(issues.values()), None)
    if strength is None and completed:
        r = completed[0]
        # A single rep can supply a personal reference without implying an ideal posture.
        if r in available["descent"] and r in available["ascent"]:
            strength = _note("Rep reference", f"In rep {r['rep']} you lowered for {r['descent_s']:.2f} s "
                             f"and rose in {r['ascent_s']:.2f} s.",
                             "Use these phases as a pace reference for your next rep.",
                             [_evidence(r, "descent"), _evidence(r, "ascent")])
    return {"strength": strength, "closer_look": closer, "quality": issues, "checks": checks,
            "targets": goals, "tempo_tolerance_s": tolerance, "source_fps": fps}

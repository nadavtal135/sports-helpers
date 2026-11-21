"""Choose at most two descriptive observations and attach their measured source frames."""

import math

PHASES = {
    "dip_start": "Dip start", "countermovement": "Lowest dip", "takeoff": "Takeoff estimate",
    "flight": "Flight", "contact": "First contact estimate", "landing_bottom": "Deepest landing bend",
    "step_lift": "Recovery step", "step_plant": "Foot replant", "recovery": "Settled recovery",
}
# These targets are personal ranges. There are no preset technique goals.
TARGETS = {
    "descent_s": ("descent time", "s", ("countermovement", "dip_start")),
    "landing_bend_deg": ("added landing knee bend", "°", ("contact", "landing_bottom")),
    "landing_torso_deg": ("landing torso lean", "°", ("landing_bottom",)),
}


def quantity(value, unit):
    return f"{value:.2f} s" if unit == "s" else f"{value:.1f}°"


def cue(row, phase):
    m = row["metrics"]
    measures = {
        "countermovement": (("Descent", "descent_s", "s"), ("Knee", "dip_knee_deg", "°"), ("Torso lean", "dip_torso_deg", "°")),
        "takeoff": (("Knee", "takeoff_knee_deg", "°"),),
        "flight": (("Estimated flight", "flight_time_estimate_s", "s"),),
        "contact": (("Knee at contact", "contact_knee_deg", "°"),),
        "landing_bottom": (("Knee", "landing_knee_deg", "°"), ("Added after contact", "landing_bend_deg", "°"),
                           ("Torso lean", "landing_torso_deg", "°")),
    }
    detail = " · ".join(f"{label} {quantity(m[key], unit)}" for label, key, unit in measures.get(phase, ())
                        if m[key] is not None)
    method = row["timing_evidence"].get(phase)
    if method:
        timing = ("Ankle-motion estimate · shoe edge unclear, check nearby frames" if method == "ankle motion"
                  else "Timing from visible shoe edge")
        detail = " · ".join(filter(None, (detail, timing)))
    return {"attempt": row["attempt"], "phase": PHASES[phase], "source_frame": row["frames"][phase], "measurement": detail}


def note(kind, title, detail, evidence):
    return {"kind": kind, "title": title, "detail": detail, "evidence": evidence}


def names(rows):
    nums = [str(r["attempt"]) for r in rows]
    if len(nums) == 1:
        return "attempt " + nums[0]
    if len(nums) > 2 and int(nums[-1]) - int(nums[0]) + 1 == len(nums):
        return f"attempts {nums[0]}–{nums[-1]}"
    return "attempts " + ", ".join(nums[:-1]) + " and " + nums[-1]


def observations(rows):
    strength = closer = None
    dips = [r for r in rows if r["metrics"]["descent_s"] is not None]
    bends = [r for r in rows if r["metrics"]["landing_bend_deg"] is not None]
    if len(dips) >= 2:
        times = [r["metrics"]["descent_s"] for r in dips]
        if max(times) - min(times) <= 0.12:
            detail = (f"Each descent measured {min(times):.2f} s." if round(max(times) - min(times), 2) == 0
                      else f"The descents ranged from {min(times):.2f} to {max(times):.2f} s.")
            strength = note("Timing consistency", f"Your descent timing was similar on {names(dips)}.",
                            detail + " Compare the lowest dip frames before takeoff.", [cue(r, "countermovement") for r in dips])
    if not strength and bends:
        r = bends[0]
        bend = r["metrics"]["landing_bend_deg"]
        title = (f"On attempt {r['attempt']}, your visible knee bent another {quantity(bend, '°')} after contact."
                 if bend > 0 else f"Your visible knee showed little additional bend after contact on attempt {r['attempt']}.")
        strength = note("Landing reference", title,
                        "Compare first contact with the deepest visible landing bend.",
                        [cue(r, "contact"), cue(r, "landing_bottom")])
    if not strength:
        r = next((r for r in rows if r.get("visible_landing_bend")), None)
        if r:
            strength = note("Landing reference", f"Your visible knee continued to bend after first contact on attempt {r['attempt']}.",
                            "Compare the estimated first contact with the lowest visible landing position.",
                            [cue(r, "contact"), cue(r, "landing_bottom")])
    if not strength:
        r = next((r for r in rows if r["frames"]["takeoff"] is not None), None)
        if r:
            flex = r["metrics"]["takeoff_knee_deg"]
            title = (f"Your visible knee measured {quantity(flex, '°')} of flexion at estimated takeoff on attempt {r['attempt']}."
                     if flex is not None else f"Both feet left the floor on attempt {r['attempt']}.")
            strength = note("Takeoff reference", title,
                            "Step back one frame to inspect the shoes as they leave the floor.", [cue(r, "takeoff")])
    if len(bends) >= 2:
        least = min(bends, key=lambda r: r["metrics"]["landing_bend_deg"])
        delta = least["metrics"]["landing_bend_deg"]
        others = [r for r in bends if r["metrics"]["landing_bend_deg"] - delta >= 12]
        if others:
            values = [r["metrics"]["landing_bend_deg"] for r in others]
            span = (quantity(values[0], "°") if min(values) == max(values) else
                    f"{min(values):.1f}–{max(values):.1f}°")
            group = [least, *others]
            closer = note("Landing comparison", f"You added less knee bend after contact on attempt {least['attempt']} "
                          f"than on {names(others)}.", f"The visible knee added {delta:.1f}° on attempt {least['attempt']} "
                          f"and {span} on {names(others)}. Inspect the contact and deepest landing frames.",
                          [cue(r, phase) for phase in ("landing_bottom", "contact") for r in group])
    if not closer:
        r = next((r for r in rows if r["step"] == "observed"), None)
        if r:
            closer = note("Recovery", f"You took a visible step after landing on attempt {r['attempt']}.",
                          "One foot lifted and replanted farther along the floor while the other stayed in place.",
                          [cue(r, "step_lift"), cue(r, "step_plant")])
    if not closer:
        for r, following in zip(rows, rows[1:]):
            if r.get("rebounds_into_next"):
                closer = note("Recovery", f"Your landing on attempt {r['attempt']} flowed into the next dip.",
                              "Compare first contact with the lowest position before the next push-off.",
                              [cue(r, "contact"), cue(following, "countermovement")])
                break
    if not closer:
        issues = [(r, q) for r in rows for q in r["issues"]]
        if issues:
            r, q = next(((r, q) for r, q in issues if q["phase"] == "contact"), issues[0])
            closer = note("Visibility", f"{PHASES[q['phase']]} is uncertain on attempt {r['attempt']}.",
                          q["detail"], [{"attempt": r["attempt"], "phase": "Visibility check",
                                         "source_frame": q["source_frame"], "measurement": "Review this interval"}])
    return strength, closer


def for_set(rows, metric=None, low=None, high=None):
    if metric is None:
        if low is not None or high is not None:
            raise ValueError("Choose a measurement for the target range.")
    elif (metric not in TARGETS or low is None or high is None or not all(map(math.isfinite, (low, high)))
          or low < 0 or low > high or (metric.endswith("_deg") and high > 180)):
        raise ValueError("Choose a valid target range with a minimum and maximum in the displayed units.")

    strength, closer = observations(rows)
    checks = {}
    if metric:
        label, unit, phases = TARGETS[metric]
        for row in rows:
            value = row["metrics"][metric]
            status = "unknown" if value is None else "below" if value < low else "above" if value > high else "met"
            checks[str(row["attempt"])] = {
                "status": status, "measurement": quantity(value, unit) if value is not None else "Unmeasured",
                "evidence": [cue(row, phase) for phase in phases if row["frames"][phase] is not None]}
        matched = [r for r in rows if checks[str(r["attempt"])]["status"] == "met"]
        if matched:
            strength = note("Your target", f"You met your {label} target on {names(matched)}.",
                            f"Your range was {quantity(low, unit)} to {quantity(high, unit)}. Inspect the measured frames.",
                            [c for r in matched for c in checks[str(r["attempt"])]["evidence"]])
        for row in rows:
            check = checks[str(row["attempt"])]
            if check["status"] in ("below", "above"):
                closer = note("Your target", f"Attempt {row['attempt']} was {check['status']} your {label} target.",
                              f"You measured {check['measurement']}. Your range was {quantity(low, unit)} to {quantity(high, unit)}.",
                              check["evidence"])
                break
    return {"strength": strength, "closer_look": closer, "checks": checks,
            "target": {"metric": metric, "low": low, "high": high} if metric else None}

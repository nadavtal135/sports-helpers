# Adapted from jeremyipark/vision-demos running/src/gait.py (Apache-2.0),
# commit c06cfb78ccd8e2c63629222695aac710fe4c199c. Modifications: the observed
# mask is taken before interpolation (it was computed after, so it was always
# true); gaps beyond a limit are not interpolated and are handled explicitly in
# strike detection, strides and intervals; absent limbs are reported instead of
# failing; explicit settings instead of a config module; live-panel-only series
# and the clip-specific report notes are removed; summary_text rewritten.
"""Ankle height, foot strikes and cadence.

One signal per leg: how far that **ankle** sits above its own stance level, in
units of the runner's leg length. It is near zero while the foot is planted and
rises to a peak at mid-swing, so a run reads as two interleaved waves — which is
what the panel plots and what every number here is measured off.

The ankle and not the foot, because COCO-17 has no foot: the gateway's ViTPose
returns 17 keypoints and the ankles are the lowest two, with no heel and no toes.
Two consequences run through this whole module. The zero has to be *calibrated*
per leg rather than being the floor, since an ankle joint stands several
centimetres above the ground even with the foot flat; and the angle of the foot
at touchdown is invisible, so the strike landmark carries an offset that depends
on how the runner lands. cadence-explained.md has the detail — it is one of the
reasons intervals are timed on mid-stance instead.

See cadence-explained.md for the method, and config.py for what each default is
doing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .skeleton import KPT_NAMES

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}

FEET = ("left", "right")


# ── signal helpers ───────────────────────────────────────────────────────────
def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Kill isolated spikes without rounding off the peaks."""
    if k < 3:
        return x
    k += (k + 1) % 2   # force odd
    pad = np.pad(x, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average. Centered and edge-padded, so it does not shift peak times."""
    if k < 2:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation across NaN gaps of at most *max_gap* frames.

    Replaces the reference's ``_interp_nan``, which filled every gap however
    long. A long gap is missing evidence, and interpolating across it would turn
    a guess into a smooth, confident-looking signal; it stays NaN instead, and
    the clip's ends are never extrapolated.
    """
    out = np.array(x, dtype=float, copy=True)
    idx = np.flatnonzero(np.isfinite(out))
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [out[a], out[b]])
    return out


def joint_series(frames: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` of one joint on the first tracked person, per returned frame.

    (0, 0) is the model's "not visible" sentinel, so it becomes NaN.
    """
    j = _INDEX[name]
    xs = np.full(len(frames), np.nan)
    ys = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []
        if j >= len(kpts) or tuple(kpts[j]) == (0, 0):
            continue
        xs[i], ys[i] = float(kpts[j][0]), float(kpts[j][1])
    return xs, ys


def _to_grid(values: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    """Observations placed on the per-source-frame grid. Nothing is filled in."""
    out = np.full(n, np.nan)
    for value, index in zip(values, indices):
        k = int(round(index))
        if 0 <= k < n and np.isfinite(value):
            out[k] = value
    return out


def _densify(values: np.ndarray, indices: np.ndarray, n: int, max_gap: int) -> np.ndarray:
    """A per-returned-frame series on the per-source-frame grid, short gaps filled."""
    return _fill_short_gaps(_to_grid(values, indices, n), max_gap)


def _joint_xy(frames: list[dict], name: str, indices: np.ndarray, n: int,
              max_gap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One joint's ``(x, y)`` on the per-source-frame grid, plus an *observed* mask.

    Fixed from the reference, which computed the mask *after* ``_densify`` had
    already interpolated between returned frames, so every frame between the
    first and last sighting counted as observed and the knee-landmark gate in
    ``analyze`` could never trip. The mask here is taken from the model's actual
    returns before anything is filled, and gaps longer than *max_gap* stay NaN.
    """
    xs, ys = joint_series(frames, name)
    grid_x, grid_y = _to_grid(xs, indices, n), _to_grid(ys, indices, n)
    observed = np.isfinite(grid_x) & np.isfinite(grid_y)
    return _fill_short_gaps(grid_x, max_gap), _fill_short_gaps(grid_y, max_gap), observed


def _smooth_runs(x: np.ndarray, median_k: int, mean_k: int) -> np.ndarray:
    """The reference's median-then-mean smoothing, applied within each finite run
    so a gap neither spreads NaN nor gets smoothed over."""
    out = np.full(len(x), np.nan)
    for start, end, label in _segments(np.isfinite(x).astype(int)):
        if label:
            out[start:end] = _box_filter(_median_filter(x[start:end], median_k), mean_k)
    return out


def _robust_peak(signal: np.ndarray) -> float:
    """Typical swing height, as the median of the peaks rather than the max.

    Every threshold below is a fraction of this number, so one badly placed
    ankle must not be allowed to move all of them.
    """
    finite = signal[np.isfinite(signal)]
    if not finite.size:
        return 0.0
    top = float(np.max(finite))
    if top <= 0:
        return 0.0

    peaks = [
        signal[i]
        for i in range(1, len(signal) - 1)
        if np.isfinite(signal[i - 1:i + 2]).all()
        and signal[i] >= signal[i - 1]
        and signal[i] > signal[i + 1]
        and signal[i] > 0.4 * top
    ]
    if len(peaks) >= 3:
        return float(np.median(peaks))
    return float(np.percentile(finite, 95))


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs as ``(start, end_exclusive, label)``."""
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, int(labels[start])))
            start = i
    return runs


def _cross_time(signal: np.ndarray, before: int, after: int, level: float,
                fps: float) -> float:
    """Sub-frame time at which *signal* crosses *level* between two frames.

    Foot strike is timed on the crossing rather than on the frame that follows
    it, because at 30 fps a whole frame is 33 ms against a ~350 ms step: taking
    the frame index alone quantizes every interval to that grid and the cadence
    inherits the staircase. The signal is steep at the crossing, which is
    exactly where linear interpolation is worth doing.
    """
    v0, v1 = signal[before], signal[after]
    if not (np.isfinite(v0) and np.isfinite(v1)) or v1 == v0:
        return after / fps
    frac = float(np.clip((level - v0) / (v1 - v0), 0.0, 1.0))
    return (before + frac) / fps


def _leg_length(limb: dict) -> float:
    """The leg, hip to ankle, as the two rigid segments it is made of.

    Measured as median thigh + median shank rather than as any average of the
    hip-to-ankle distance, because the knee is bent for most of a running clip
    and an average of that distance is really an average of how bent it
    happened to be. On this clip it ran 8-11% short, which is the kind of error
    that shows: the two segments then summed to 1.09-1.14 of the "leg" they are
    the two halves of.

    A segment length is only a length at all once x and y are in the same unit
    (see analyze), and once they are it does not depend on the limb's
    orientation — which is what makes a median of it meaningful rather than a
    mixture over the stride.

    One figure for the clip, so the two legs are averaged. They differ by about
    5% here and that is the camera: the near leg is closer to the lens and
    projects larger. Neither is more correct, and using each foot's own would
    make the two legs' numbers incomparable, which is the opposite of what the
    unit is for.
    """
    per_leg = []
    for joints in limb.values():
        total = 0.0
        for a_, b_ in (("hip", "knee"), ("knee", "ankle")):
            d = np.hypot(joints[b_][0] - joints[a_][0], joints[b_][1] - joints[a_][1])
            d = d[np.isfinite(d)]
            if not d.size:
                total = np.nan
                break
            total += float(np.median(d))
        if np.isfinite(total):
            per_leg.append(total)
    return float(np.mean(per_leg)) if per_leg else 0.0


def _knee_flexion(hip, knee, ankle) -> np.ndarray:
    """Knee flexion in degrees, per frame. 0 is a straight leg.

    Sampled for the panel at its own local minimum near each strike rather than
    at the strike itself; see the note in analyze().

    The interior angle at the knee, between the thigh (knee -> hip) and the
    shank (knee -> ankle), subtracted from 180. A fully extended leg puts those
    two vectors 180 apart and reads 0 flexion; a bent knee reads positive.

    This is the projected angle in the image plane, not the anatomical one. Seen
    from the side that is very nearly the sagittal knee angle, which is the one
    worth having — but a leg swinging toward or away from the camera foreshortens
    and reads flatter than it is.

    The coordinates must already be in one unit on both axes; analyze() puts
    them there. Fed the model's raw normalized x, this measures the angle in a
    space stretched by the frame's aspect ratio, which inflates every deviation
    from vertical — and flexion is exactly such a deviation, bounded below at
    zero, so it can only read high. On this clip that was worth +10 degrees at
    foot strike and +25 at the swing peak.
    """
    v1 = np.vstack([hip[0] - knee[0], hip[1] - knee[1]])
    v2 = np.vstack([ankle[0] - knee[0], ankle[1] - knee[1]])
    n1, n2 = np.hypot(*v1), np.hypot(*v2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(axis=0) / (n1 * n2)
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _heading(limb: dict, planted: dict, nose_x: np.ndarray) -> tuple[float, float]:
    """Which way the runner is *travelling*, in image x. +1 is towards +x.

    Returned as ``(heading, facing)``, both -1, +1 or 0 when undecidable.

    Measured off the planted foot, not off the runner's displacement: on a
    treadmill they do not go anywhere, so displacement is zero and would decide
    nothing. What holds either way is that a foot in contact with the ground is
    the part of the runner that is *not* moving with them — so relative to the
    hips it travels backwards through every stance, on a belt or on a road.
    Averaged over every planted frame of both feet, the sign of that is the
    direction of travel reversed.

    `facing` is the cross-check, from the nose against the mid-hip. It is
    anatomy rather than motion, so it is only the fallback and the disagreement
    flag: a runner backwards on a belt, or a clip where the model has the body
    turned around, is exactly the case where the two differ.
    """
    drift = []
    for foot, joints in limb.items():
        ankle_x, hip_x = joints["ankle"][0], joints["hip"][0]
        rel = np.asarray(ankle_x) - np.asarray(hip_x)
        stance = np.asarray(planted[foot])
        # Both ends of a step have to be in contact for the difference to be a
        # stance movement rather than the jump into or out of one.
        keep = stance[1:] & stance[:-1]
        d = np.diff(rel)[keep]
        drift += [v for v in d if np.isfinite(v)]

    heading = -float(np.sign(np.mean(drift))) if drift else 0.0

    hips = [np.asarray(joints["hip"][0]) for joints in limb.values()]
    mid_hip_x = np.nanmean(np.vstack(hips), axis=0) if hips else np.zeros(0)
    lead = np.asarray(nose_x) - mid_hip_x
    lead = lead[np.isfinite(lead)]
    facing = float(np.sign(np.mean(lead))) if lead.size else 0.0
    return heading or facing, facing


def _vertex(series: np.ndarray, centre: int, reach: int) -> tuple[float, float]:
    """Sub-frame minimum of *series* near *centre*, as ``(frame, value)``.

    A least-squares parabola over the window, taking its vertex. Every point in
    the window contributes, so the noise averages down instead of being selected
    for — which is the whole problem with taking the lowest frame.

    The two obvious alternatives are both biased, in opposite directions:

    * ``argmin`` of the raw series picks the noisiest low sample. With this
      clip's 4.8 deg of per-frame flexion jitter that is worth about -5 deg.
    * ``argmin`` of a smoothed series removes that, but box-smoothing flattens
      an extremum, so it reads about +3 deg high.

    Measured on this clip the three give 13.8 / 19.5 / 17.5 degrees for the same
    left knee. The parabola is the one with no thumb on the scale.
    """
    lo = max(0, centre - reach)
    hi = min(len(series), centre + reach + 1)
    window = series[lo:hi]
    if len(window) < 5 or not np.isfinite(window).all():
        finite = [i for i in range(lo, hi) if np.isfinite(series[i])]
        if not finite:
            return float(centre), float("nan")
        best = min(finite, key=lambda i: series[i])
        return float(best), float(series[best])

    t = np.arange(len(window)) - (len(window) - 1) / 2.0
    a2, a1, a0 = np.polyfit(t, window, 2)
    if a2 <= 0:                      # not a minimum: fall back to the lowest
        j = int(np.argmin(window))
        return float(lo + j), float(window[j])
    peak = -a1 / (2.0 * a2)
    peak = float(np.clip(peak, t[0], t[-1]))
    return (lo + (len(window) - 1) / 2.0 + peak,
            float(a2 * peak ** 2 + a1 * peak + a0))


def _segment_angle(a, b) -> np.ndarray:
    """Direction from *a* to *b* in radians, in image coordinates (y down)."""
    return np.arctan2(b[1] - a[1], b[0] - a[0])


def _circular_mean(angles: np.ndarray) -> float:
    """Mean of angles, via the unit vectors — so 359 and 1 average to 0."""
    finite = angles[np.isfinite(angles)]
    if not finite.size:
        return float("nan")
    return float(np.arctan2(np.mean(np.sin(finite)), np.mean(np.cos(finite))))


def _circular_sd(angles: np.ndarray) -> float:
    """Spread of angles in radians, from the length of the mean resultant.

    ``sqrt(-2 ln R)``, which for a tight cluster agrees with the ordinary
    standard deviation and, unlike it, cannot be thrown by the wrap at ±pi.
    """
    finite = angles[np.isfinite(angles)]
    if finite.size < 2:
        return 0.0
    r = np.hypot(np.mean(np.cos(finite)), np.mean(np.sin(finite)))
    return float(np.sqrt(-2.0 * np.log(max(r, 1e-12))))


@dataclass
class KneeShape:
    """The mean hip-knee-ankle geometry at foot strike, for one leg.

    Held as two segment *directions* plus two lengths rather than as three
    averaged points. Averaging the points directly shortens the limb whenever
    the orientation varies — the mean of two opposite positions is the midpoint
    — which would draw a leg that shrinks the more inconsistent the runner is.
    Averaging directions and taking the median lengths keeps the limb the right
    size and puts all the variation where it belongs, in the angles.
    """

    foot: str
    n: int
    thigh_angle: float      # radians, image coordinates
    shank_angle: float
    thigh_length: float     # leg lengths
    shank_length: float
    flexion: float          # degrees, 0 = straight
    flexion_sd: float       # spread of the knee angle across those strikes
    thigh_sd: float = 0.0   # spread of the whole limb's orientation, radians
    bend_sign: float = 1.0  # which way the shank turns off the thigh

    def as_dict(self) -> dict:
        return {
            "foot": self.foot, "strikes_averaged": self.n,
            "flexion_deg": _round(self.flexion, 2),
            "flexion_sd_deg": _round(self.flexion_sd, 2),
            "thigh_deg": _round(np.degrees(self.thigh_angle), 2),
            "shank_deg": _round(np.degrees(self.shank_angle), 2),
            "thigh_deg_sd": _round(np.degrees(self.thigh_sd), 2),
            "thigh_length_legs": _round(self.thigh_length, 4),
            "shank_length_legs": _round(self.shank_length, 4),
        }


# ── events ───────────────────────────────────────────────────────────────────
@dataclass
class Step:
    """One foot strike, and the swing that preceded it.

    Two landmarks, because no single one does both jobs. ``time`` is the strike
    — the instant the foot lands, which is what the overlay flashes on and what
    the panel dots. ``mid_stance`` is the centre of the same stance, which is
    what every *interval* is measured between. See cadence-explained.md; the
    short version is that a threshold crossing is where the signal is steep and
    therefore precise against noise, but it also slides when the threshold is
    slightly wrong — and it slides by different amounts on the near and the far
    leg, which a side-on camera guarantees. The centre of a stance run does not
    move when the threshold is wrong, because the run grows at both ends.
    """

    number: int
    foot: str
    time: float                     # foot strike, seconds
    mid_stance: float               # centre of the stance; intervals are timed here
    frame: int                      # nearest source frame, for the overlay flash
    peak_time: float                # mid-swing, the highest the foot got
    peak_clearance: float           # in leg lengths
    contact_seconds: float          # strike to toe-off; a proxy, see the docs
    knee_frame: int = 0                  # nearest frame, for gating and reporting
    knee_time: float = 0.0               # sub-frame instant the shape is read at
    knee_offset: float = 0.0             # seconds from the strike to that instant
    knee_flexion: float = float("nan")   # degrees there
    interval: float = float("nan")  # since the previous step, either foot
    stride: float = float("nan")    # since the previous step, same foot

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "foot": self.foot,
            "time": round(self.time, 4),
            "mid_stance": _round(self.mid_stance, 4),
            "peak_time": _round(self.peak_time, 4),
            "peak_clearance": _round(self.peak_clearance, 4),
            "contact_seconds": _round(self.contact_seconds, 4),
            "knee_flexion_deg": None if not np.isfinite(self.knee_flexion)
            else round(self.knee_flexion, 2),
            "knee_sample_offset_ms": round(self.knee_offset * 1000.0, 1),
            "step_interval": None if not np.isfinite(self.interval)
            else round(self.interval, 4),
            "stride_seconds": None if not np.isfinite(self.stride)
            else round(self.stride, 4),
        }


@dataclass
class GaitAnalysis:
    """Everything the panel, the plots and the report read from."""

    fps: float
    n_frames: int
    clearance: dict[str, np.ndarray]      # per foot, per source frame, leg lengths
    raw: dict[str, np.ndarray]            # the same before smoothing
    amplitude: dict[str, float]           # typical swing height per foot
    contact_level: float                  # the threshold a strike is timed on
    leg_length: float                     # normalized frame units
    ground: dict[str, float]              # each foot's own belt line, normalized y
    hip_height: np.ndarray                # mid-hip, leg lengths above ground
    knee_flexion: dict = field(default_factory=dict)   # per foot, per frame, degrees
    limb: dict = field(default_factory=dict)           # per foot: hip/knee/ankle x,y
    steps: list[Step] = field(default_factory=list)
    cadence_live: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _smoothing: float = 0.2      # EMA weight the live trace was built with
    warnings: dict = field(default_factory=dict)   # silent failure modes, counted
    airborne: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    # Direction of travel in image x, +1 towards +x. See _heading.
    heading: float = 0.0
    facing: float = 0.0
    # The source frame's width / height. `limb` is already scaled by it, so
    # nothing downstream needs to apply it again; it is here to be recorded.
    frame_aspect: float = 1.0
    observed: dict = field(default_factory=dict)   # per foot: whole limb returned by the model
    present: list = field(default_factory=lambda: list(FEET))   # feet tracked well enough
    min_strides_confident: int = 4

    # ── counts and headline numbers ──────────────────────────────────────────
    @property
    def count(self) -> int:
        return len(self.steps)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / (self.fps or 30.0)

    @property
    def intervals(self) -> list[float]:
        return [s.interval for s in self.steps if np.isfinite(s.interval)]

    def mean_cadence_by(self, until: float | None = None) -> float:
        """Steps per minute: ``120 / mean(same-foot stride seconds)``.

        Strides cancel any per-foot offset in where the landmark sits (see the
        reference's cadence-explained.md). Unlike the reference there is no
        fallback to first-to-last step timing when no stride exists: those
        steps may span a tracking gap, so the cadence is NaN instead.
        """
        strides = [st.stride for st in self.steps
                   if np.isfinite(st.stride) and (until is None or st.mid_stance <= until)]
        return 120.0 / float(np.mean(strides)) if strides else float("nan")

    @property
    def stride_count(self) -> int:
        return sum(1 for st in self.steps if np.isfinite(st.stride))

    @property
    def cadence_confidence(self) -> str:
        """"none" without a measurable stride, "low" under ``min_strides_confident``."""
        n = self.stride_count
        return "none" if n == 0 else "low" if n < self.min_strides_confident else "ok"

    @property
    def mean_step(self) -> float:
        return float(np.mean(self.intervals)) if self.intervals else float("nan")

    @property
    def mean_cadence(self) -> float:
        """Steps per minute over the whole clip."""
        return self.mean_cadence_by(None)

    @property
    def mean_stride(self) -> float:
        strides = [s.stride for s in self.steps if np.isfinite(s.stride)]
        return float(np.mean(strides)) if strides else float("nan")

    @property
    def flight_ratio(self) -> float:
        """Share of the run with neither foot planted — the airborne phase.

        This is what separates running from walking: a walk always has a foot
        down, so it would read 0 here.
        """
        return float(np.mean(self.airborne)) if len(self.airborne) else float("nan")

    def stride_spread(self, until: float | None = None) -> float:
        """Stride-to-stride variability, in seconds. How metronomic the run was.

        Every standard deviation in this module is a *sample* one (``ddof=1``).
        They were mixed before — this one corrected, the step spread and the
        knee spread not — which made a 22-sample figure read about 2% low
        against its neighbour for no reason a reader could see.

        Measured on the stride rather than on the step, and that is the whole
        point: a stride is one foot's landmark to that same foot's next one, so
        any per-foot offset in where the landmark sits cancels in the
        subtraction. The step interval does not have that property — the
        intervals alternate long/short, and on this clip that alternation alone
        contributes 7.4 ms of the 10.8 ms step spread, none of it the runner.

        Stride spread comes out 10.7 ms here, and the two feet agree on it
        (10.2 and 11.3 ms), which is what an uncontaminated measure looks like.
        """
        strides = [s.stride for s in self.steps
                   if np.isfinite(s.stride)
                   and (until is None or s.mid_stance <= until)]
        return float(np.std(strides, ddof=1)) if len(strides) > 1 else float("nan")

    def step_spread(self) -> float:
        """Standard deviation of the step interval — how metronomic the run was.

        Worth reporting only because the timing landmark is a stable one: on the
        strike crossing this clip reads 34 ms, on mid-stance 12 ms, and the
        difference is measurement rather than running. See cadence-explained.md.
        """
        intervals = self.intervals
        return float(np.std(intervals, ddof=1)) if len(intervals) > 1 else float("nan")

    def step_interval(self, foot: str, until: float | None = None) -> float:
        """Mean interval of the steps that *land* on this foot, in seconds.

        The step landing on the left foot is timed from the right foot's
        landmark to the left's. That is the quantity asymmetry lives in: whether
        one foot comes down halfway between the other's, or early, or late.

        Deliberately not the per-foot *stride*. A stride is one foot's landmark
        to that same foot's next one, so both feet complete the same cycles over
        the same clip and their rates are pinned together by arithmetic — 179.4
        against 179.6 here, and that would hold however unevenly someone ran.

        *until* bounds it to steps already landed, for the live panel.
        """
        values = [s.interval for s in self.steps
                  if s.foot == foot and np.isfinite(s.interval)
                  and (until is None or s.mid_stance <= until)]
        return float(np.mean(values)) if values else float("nan")

    def foot_stride_rate(self, foot: str, until: float | None = None) -> float:
        """How often this foot actually lands, as a steps-per-minute equivalent.

        This is the *only* honest per-foot rate, and it is the same for both
        feet by construction — the feet alternate, so over any window each takes
        half the steps. On this clip 179.37 against 179.56, a difference of
        0.18 spm at t = 0.20. If these two ever differ meaningfully, something is
        wrong with the detection, not with the runner.
        """
        strides = [s.stride for s in self.steps_for(foot)
                   if np.isfinite(s.stride)
                   and (until is None or s.mid_stance <= until)]
        return 120.0 / float(np.mean(strides)) if strides else float("nan")

    def phase_split(self, until: float | None = None) -> tuple[float, float]:
        """The stride split into its two halves, as percentages summing to 100.

        This is what a left/right difference in this data really is, and it is
        one number rather than two: the interval before a left landing plus the
        interval before a right landing *is* the stride, so if one half grows
        the other shrinks by exactly as much.

        Reporting the two halves as two step rates in spm — which an earlier
        version of the panel did — reads as "the right foot takes more steps per
        minute than the left". That cannot happen. 50/50 means each foot lands
        exactly halfway between the other's landings.
        """
        left = self.step_interval("left", until)
        right = self.step_interval("right", until)
        total = left + right
        if not np.isfinite(total) or total <= 0:
            return float("nan"), float("nan")
        return 100.0 * left / total, 100.0 * right / total

    def symmetry(self, until: float | None = None) -> float:
        """How even the stride's two halves are, as a percentage.

        **Not resolvable from a single side-on camera. Kept for the record, and
        deliberately absent from the panel.** Read cadence-explained.md before
        using it for anything.

        A *phase* measure, not a rate one. Both feet land at the same rate — see
        foot_stride_rate — so this asks whether each foot comes down halfway
        between the other's landings, or early, or late.

        The reason it does not survive: a real phase shift is a constant time
        offset, so it must read the same wherever in the gait cycle it is
        measured. Swept around the cycle on this clip it runs from -41 ms to
        +69 ms and changes sign — 110 ms of spread on a 15 ms result.

        And the strikes are in fact evenly spaced. A foot strike is a physical
        event, so no camera distance can move it; what moves is the *landmark*,
        which precedes touchdown by (threshold - touchdown height) / descent
        speed, and the two legs descend at different speeds. Sweeping the
        threshold, the alternation passes through zero at about 23% of swing
        amplitude, where the two feet come out even to 0.2 +/- 2.9 ms. One free
        parameter erases the whole asymmetry, so it carried no information about
        the runner. The threshold is not moved there because at that height the
        airborne fraction collapses to ~20% and then 8%, which no runner does;
        see cadence-explained.md.

        The shorter over the longer, so 100 means the feet land exactly evenly
        spaced and there is no direction to the number — which is right, since
        "the left leads by 4%" and "the right trails by 4%" describe one thing.

        **Read the floor before reading the value.** A single side-on camera
        cannot resolve this well: the two ankles descend through the strike
        threshold at different rates (0.80 leg lengths a second on the right,
        0.53 on the left), so a fixed landmark sits at a different point in each
        leg's cycle. Whether that difference is the runner or the view of them
        is not separable from one camera — but either way the *timing* reading
        is invalid, which is what the landmark sweep in cadence-explained.md
        shows.
        """
        left = self.step_interval("left", until)
        right = self.step_interval("right", until)
        if not (np.isfinite(left) and np.isfinite(right)) or min(left, right) <= 0:
            return float("nan")
        return 100.0 * min(left, right) / max(left, right)

    def knee_shape(self, foot: str, until: float | None = None,
                   window: int | None = None) -> KneeShape | None:
        """Mean hip-knee-ankle geometry at this foot's strikes, or None.

        Averaged over the strikes that have already landed, so the shape fills
        in as the run goes rather than being the whole clip's answer shown from
        frame one.

        ``window`` averages only the most recent N strikes. A mean over *every*
        strike converges after a few strides and then sits perfectly still —
        1.0 degrees of movement across this clip — which reads as a frozen
        readout rather than a steady one. A trailing window keeps moving because
        it keeps forgetting: eight strikes gives 1.7 degrees of drift and is
        still an average, just over a stated span. The panel uses a window; the
        figures in gait.json and summary.txt use every strike.

        Every length is in leg lengths, so the drawing is the same size whatever
        the camera distance or export resolution. See KneeShape for why the
        directions are averaged rather than the joint positions.
        """
        if foot not in self.limb:
            return None
        # Gated on the frame the shape is *sampled* at, not on the strike. The
        # landmark can sit up to KNEE_SEARCH_SECONDS after the strike — on this
        # clip 15 of 45 do, by as much as 67 ms — so filtering on the strike let
        # the live panel average in geometry from frames that had not played
        # yet. Everything else on the panel is strictly causal; this was not.
        # Gated on the instant the shape is *sampled* at, not on the strike: the
        # landmark can sit after its own strike, so filtering on the strike let
        # the live panel average in geometry from frames that had not played yet.
        cut = np.inf if until is None else float(until)
        landed = [s for s in self.steps_for(foot)
                  if s.knee_time <= cut and np.isfinite(s.knee_flexion)]
        if window:
            landed = landed[-int(window):]
        if not landed:
            return None

        hip, knee, ankle = (self.limb[foot][j] for j in ("hip", "knee", "ankle"))
        grid = np.arange(self.n_frames)
        at = np.array([s.knee_time * (self.fps or 30.0) for s in landed])
        # Read at the sub-frame instant the vertex found, rather than snapped to
        # the nearest frame: the knee moves 150-420 deg/s through here, so half
        # a frame is several degrees of limb geometry.
        pick = lambda j: (np.interp(at, grid, j[0]),
                          np.interp(at, grid, j[1]))   # noqa: E731
        h, k, a = pick(hip), pick(knee), pick(ankle)

        thigh = _segment_angle(h, k)
        shank = _segment_angle(k, a)
        good = np.isfinite(thigh) & np.isfinite(shank)
        if not good.any():
            return None

        flex = np.array([s.knee_flexion for s in landed])
        flex = flex[np.isfinite(flex)]

        # The shank direction is *derived* from the reported flexion rather
        # than averaged independently, so the limb drawn and the number beside
        # it cannot disagree. Averaging both separately left them 1-2 degrees
        # apart: the label is a mean of fitted vertex values, the drawing was a
        # mean of measured directions, and nothing forces two different
        # averages of the same thing to coincide. A panel whose picture and
        # caption differ is worse than either.
        #
        # The thigh direction and both lengths are still measured; only the
        # combination is reconstructed. A straight leg (0 flexion) puts the
        # shank along the thigh, which is what collinear hip-knee-ankle means.
        thigh_mean = _circular_mean(thigh[good])
        shank_mean = _circular_mean(shank[good])
        flexion = float(np.mean(flex)) if flex.size else float("nan")
        bend_sign = 1.0
        if np.isfinite(flexion) and np.isfinite(thigh_mean):
            turn = np.radians(flexion)
            options = ((thigh_mean + turn, 1.0), (thigh_mean - turn, -1.0))
            shank_mean, bend_sign = min(options, key=lambda c: abs(np.arctan2(
                np.sin(c[0] - shank_mean), np.cos(c[0] - shank_mean))))

        return KneeShape(
            foot=foot,
            n=int(good.sum()),
            thigh_angle=thigh_mean,
            shank_angle=shank_mean,
            thigh_length=float(np.median(np.hypot(k[0] - h[0], k[1] - h[1])[good]))
            / self.leg_length,
            shank_length=float(np.median(np.hypot(a[0] - k[0], a[1] - k[1])[good]))
            / self.leg_length,
            flexion=flexion,
            flexion_sd=float(np.std(flex, ddof=1)) if flex.size > 1 else float("nan"),
            thigh_sd=_circular_sd(thigh[good]),
            bend_sign=bend_sign,
        )

    def segment_ratio(self, foot: str) -> float:
        """Measured shank length over thigh length, at foot strike.

        A keypoint-quality check. Hip-to-knee and knee-to-ankle are near-equal
        in humans — both about 0.245 of standing height — so this should read
        near 1.0. On this clip it gives 1.04 left and 1.04 right: both sane, and
        equal, so neither knee is misplaced relative to the other.

        This is the number that caught the coordinate space being wrong. While
        an x distance and a y distance were still in different units (see
        analyze) it read 0.93 and 0.91 at contact and 1.6-1.9 through mid-swing,
        and the swing figure was written off as the shank foreshortening as it
        points toward or away from the camera. It was mostly not: in one unit it
        reads 1.00 and 1.13 there. A shank cannot change length, so a "ratio"
        that moves through the stride is the measurement moving, not the leg.

        Measured **at the strike frames** rather than over a low-flexion
        percentile of the whole clip. That percentile was the first attempt and
        it was wrong: running has two knee-extension minima per stride, one at
        contact and one near toe-off, so the filter mixed two different phases
        and reported 0.93 against 1.33 — a difference that was entirely which
        frames each leg happened to contribute.

        Still read it near contact by preference. The residual through mid-swing
        (1.00 left, 1.13 right) is real projection — the swinging shank does
        turn out of the image plane — and near contact both legs are close to
        side-on, which is the phase this is a check on.
        """
        if foot not in self.limb:
            return float("nan")
        L = self.limb[foot]
        thigh = np.hypot(L["knee"][0] - L["hip"][0], L["knee"][1] - L["hip"][1])
        shank = np.hypot(L["ankle"][0] - L["knee"][0], L["ankle"][1] - L["knee"][1])
        at = np.array([s.knee_time * (self.fps or 30.0)
                       for s in self.steps_for(foot)])
        if not at.size:
            return float("nan")
        grid = np.arange(self.n_frames)
        ratio = np.interp(at, grid, shank / thigh)
        ratio = ratio[np.isfinite(ratio)]
        return float(np.median(ratio)) if ratio.size else float("nan")

    def peak_flexion(self, foot: str) -> float:
        """The most the knee bends anywhere in the cycle, in degrees.

        Mid-swing, not at contact — the heel comes up toward the glute — so it
        is a different measurement from the strike figure and much larger.
        """
        series = self.knee_flexion.get(foot)
        if series is None or not len(series):
            return float("nan")
        finite = series[np.isfinite(series)]
        return float(np.percentile(finite, 98)) if finite.size else float("nan")

    def steps_for(self, foot: str) -> list[Step]:
        return [s for s in self.steps if s.foot == foot]

    def contact_mean(self, foot: str) -> float:
        values = [s.contact_seconds for s in self.steps_for(foot)
                  if np.isfinite(s.contact_seconds)]
        return float(np.mean(values)) if values else float("nan")

    def balance(self) -> tuple[float, float]:
        """Left / right share of total contact time, as percentages.

        Even at a real asymmetry the two numbers sit within a point or two of
        50, which is why it is reported to one decimal: the interesting range is
        narrow, and rounding to whole percent throws the finding away.
        """
        left, right = self.contact_mean("left"), self.contact_mean("right")
        if not (np.isfinite(left) and np.isfinite(right)) or left + right <= 0:
            return float("nan"), float("nan")
        total = left + right
        return 100.0 * left / total, 100.0 * right / total

    def vertical_oscillation(self, until: float | None = None) -> float:
        """Median hip rise and fall within a stride, in leg lengths.

        Per stride rather than over the clip, because a runner drifting a little
        forward or back on the belt would otherwise be measured as bounce.

        *until* bounds it to strides already completed, which is what the live
        panel needs: the figure has to be the median of what has happened, not
        of the whole clip.
        """
        strides = [(a.mid_stance, b.mid_stance)
                   for foot in FEET
                   for a, b in zip(self.steps_for(foot), self.steps_for(foot)[1:])
                   if np.isfinite(a.mid_stance) and np.isfinite(b.stride)
                   and (until is None or b.mid_stance <= until)]
        if not strides:
            return float("nan")
        fps = self.fps or 30.0
        spans = []
        for t0, t1 in strides:
            i0, i1 = int(round(t0 * fps)), int(round(t1 * fps))
            window = self.hip_height[max(0, i0):min(self.n_frames, i1 + 1)]
            window = window[np.isfinite(window)]
            if window.size > 2:
                spans.append(float(window.max() - window.min()))
        return float(np.median(spans)) if spans else float("nan")

    def cadence_at(self, frame: int) -> float:
        if not len(self.cadence_live) or frame >= len(self.cadence_live):
            return float("nan")
        return float(self.cadence_live[frame])

    def steps_by(self, frame: int) -> int:
        """Strikes landed at or before *frame*, for the live counter."""
        t = frame / (self.fps or 30.0)
        return sum(1 for s in self.steps if s.time <= t)

    # ── serialization ────────────────────────────────────────────────────────
    def summary(self) -> dict:
        left_pct, right_pct = self.balance()
        intervals = self.intervals
        return {
            "steps": self.count,
            "cadence_spm": _round(self.mean_cadence, 1),
            "cadence_method": "120 / mean same-foot stride seconds (mid-stance to mid-stance)",
            "cadence_confidence": self.cadence_confidence,
            "strides_measured": self.stride_count,
            "mean_step_seconds": _round(float(np.mean(intervals)) if intervals
                                        else float("nan"), 4),
            # Step spread is inflated by the long/short alternation; stride
            # spread is the clean one. See stride_spread.
            "step_seconds_sd": _round(self.step_spread(), 4),
            "stride_seconds_sd": _round(self.stride_spread(), 4),
            "mean_stride_seconds": _round(self.mean_stride, 4),
            "step_seconds_per_foot": {foot: _round(self.step_interval(foot), 4)
                                      for foot in FEET},
            # The real per-foot rate, which is equal for both by construction.
            "stride_rate_spm_per_foot": {
                foot: _round(self.foot_stride_rate(foot), 2) for foot in FEET},
            "phase_split_percent": dict(zip(
                FEET, [_round(v, 1) for v in self.phase_split()])),
            "contact_seconds": {foot: _round(self.contact_mean(foot), 4)
                                for foot in FEET},
            "contact_balance_percent": {"left": _round(left_pct, 1),
                                        "right": _round(right_pct, 1)},
            "flight_ratio": _round(self.flight_ratio, 4),
            # Image-space direction of travel, and the anatomical cross-check.
            # See _heading; the trails stream against `heading`.
            "heading_x": self.heading,
            "facing_x": self.facing,
            "peak_clearance_legs": {
                foot: _round(float(np.mean([s.peak_clearance
                                            for s in self.steps_for(foot)]))
                             if self.steps_for(foot) else float("nan"), 4)
                for foot in FEET
            },
            "vertical_oscillation_legs": _round(self.vertical_oscillation(), 4),
            "knee_flexion_at_strike_deg": {
                foot: {"mean": _round(shape.flexion, 2),
                       "sd": _round(shape.flexion_sd, 2),
                       "strikes": shape.n}
                for foot in FEET
                if (shape := self.knee_shape(foot)) is not None
            },
            "knee_peak_flexion_deg": {foot: _round(self.peak_flexion(foot), 2)
                                      for foot in FEET},
            # Keypoint-quality check: should be near 1.0. See segment_ratio.
            "shank_thigh_ratio_at_contact": {
                foot: _round(self.segment_ratio(foot), 3) for foot in FEET},
            "knee_shape_at_strike": {
                foot: shape.as_dict() for foot in FEET
                if (shape := self.knee_shape(foot)) is not None
            },
            "steps_per_foot": {foot: len(self.steps_for(foot)) for foot in FEET},
            "quality": self.warnings,
            "steps_detail": [s.as_dict() for s in self.steps],
        }

    def series(self) -> dict:
        """The full per-frame signals, for gait.json."""
        return {
            "time": [round(float(v), 4) for v in self.times],
            "clearance": {foot: [_round(float(v), 5) for v in self.clearance[foot]]
                          for foot in FEET},
            "hip_height": [_round(float(v), 5) for v in self.hip_height],
            "knee_flexion_deg": {
                foot: [_round(float(v), 3) for v in self.knee_flexion[foot]]
                for foot in FEET if foot in self.knee_flexion
            },
            "cadence_spm": [_round(float(v), 2) for v in self.cadence_live],
            "airborne": [bool(v) for v in self.airborne],
            "observed": {foot: [bool(v) for v in self.observed[foot]]
                         for foot in FEET if foot in self.observed},
        }


def _round(value: float, places: int):
    """JSON has no NaN. None is what a missing measurement actually is."""
    return None if value is None or not np.isfinite(value) else round(float(value), places)


def signal_config(settings) -> dict:
    """The settings that shaped the signal, snapshotted into the JSON export."""
    keys = ("foot_reference", "per_foot_ground", "ground_percentile", "smooth_median_frames",
            "smooth_mean_frames", "swing_peak_fraction", "contact_fraction",
            "min_contact_seconds", "cadence_smoothing", "knee_search_seconds",
            "max_interp_gap_seconds", "min_limb_coverage", "stride_outlier_ratio")
    return {k: getattr(settings, k) for k in keys}


# ── the analysis ─────────────────────────────────────────────────────────────
def analyze(frames: list[dict], fps: float, n_source_frames: int, *, settings,
            aspect: float) -> GaitAnalysis:
    """Build both feet's clearance signals and find every foot strike.

    *aspect* is the analyzed frame's width / height. The model normalizes x by
    the width and y by the height, so x is multiplied by the aspect ratio to put
    both axes in units of the frame height before any length or angle is taken.

    Changes from the reference: gaps longer than ``max_interp_gap_seconds`` are
    not interpolated; the observed mask comes from actual model returns; a foot
    seen on too few frames is reported absent and left out (instead of crashing
    or producing strikes from interpolation); strides and step intervals that
    span a tracking gap, or a missed opposite-foot strike, are not measured.
    """
    s = settings
    indices = np.array([f["index"] for f in frames], dtype=float)
    n = max(n_source_frames, 1)
    stride = int(np.median(np.diff(indices))) if len(indices) > 1 else 1
    max_gap = max(int(round(s.max_interp_gap_seconds * fps)), stride)

    raw_limb = {foot: {joint: _joint_xy(frames, f"{foot}_{joint}", indices, n, max_gap)
                       for joint in ("hip", "knee", "ankle")}
                for foot in FEET}
    scale_x = float(aspect)
    limb = {foot: {j: (v[0] * scale_x, v[1]) for j, v in joints.items()}
            for foot, joints in raw_limb.items()}
    observed = {foot: np.logical_and.reduce([raw_limb[foot][j][2]
                                             for j in ("hip", "knee", "ankle")])
                for foot in FEET}
    ankle_observed = {foot: raw_limb[foot]["ankle"][2] for foot in FEET}

    coverage = {foot: float(ankle_observed[foot].mean()) for foot in FEET}
    present = [foot for foot in FEET if coverage[foot] >= s.min_limb_coverage]
    absent = [foot for foot in FEET if foot not in present]
    if not present:
        raise RuntimeError(
            "Neither ankle was tracked on enough frames "
            f"(left {coverage['left']:.0%}, right {coverage['right']:.0%}). "
            "Film from the side with the runner's feet in frame.")

    ankle_y = {foot: raw_limb[foot]["ankle"][1] for foot in FEET}
    hip_y = {foot: raw_limb[foot]["hip"][1] for foot in FEET}

    leg_length = _leg_length({foot: limb[foot] for foot in present})
    if leg_length <= 1e-6:
        raise RuntimeError("Could not measure a leg length: hips, knees or ankles were never "
                           "detected together. Check that the runner is fully in frame.")

    def ground_of(values):
        finite = values[np.isfinite(values)]
        return float(np.percentile(finite, s.ground_percentile)) if finite.size else float("nan")

    if s.per_foot_ground:
        ground = {foot: ground_of(ankle_y[foot]) for foot in FEET}
    else:
        shared = ground_of(np.concatenate([ankle_y[foot] for foot in present]))
        ground = {foot: shared for foot in FEET}

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mid_hip = np.nanmean(np.vstack([hip_y[foot] for foot in FEET]), axis=0)

    raw, clearance, amplitude = {}, {}, {}
    for foot in FEET:
        series = (ground[foot] - ankle_y[foot]) / leg_length
        if s.foot_reference == "hip":
            series = (mid_hip - ankle_y[foot]) / leg_length
            if np.isfinite(series).any():
                series = series - np.nanpercentile(series, 100 - s.ground_percentile)
        if foot in absent:
            series = np.full(n, np.nan)
        raw[foot] = series
        clearance[foot] = _smooth_runs(series, s.smooth_median_frames, s.smooth_mean_frames)
        amplitude[foot] = _robust_peak(clearance[foot]) if foot in present else 0.0

    swing = min((v for v in amplitude.values() if v > 0), default=0.0)
    contact_level = s.contact_fraction * swing
    high = s.swing_peak_fraction * swing

    strikes: list[Step] = []
    hidden = 0
    for foot in present:
        found, lost = _strikes(clearance[foot], foot, fps, high=high, low=contact_level,
                               min_frames=max(1, int(round(s.min_contact_seconds * fps))),
                               fraction=s.swing_peak_fraction)
        strikes += found
        hidden += lost
    strikes.sort(key=lambda st: st.time)

    gap = {foot: ~np.isfinite(clearance[foot]) for foot in FEET}

    def gap_between(foot, t0, t1) -> bool:
        a, b = int(np.floor(t0 * fps)), int(np.ceil(t1 * fps)) + 1
        return bool(gap[foot][max(0, a):min(n, b)].any())

    last_by_foot: dict[str, Step] = {}
    spanning_gaps = same_foot = 0
    for i, step in enumerate(strikes):
        step.number = i + 1
        step.frame = int(min(n - 1, round(step.time * fps)))
        if not np.isfinite(step.mid_stance):
            continue
        prev = strikes[i - 1] if i else None
        if prev is not None and np.isfinite(prev.mid_stance):
            if prev.foot == step.foot:
                same_foot += 1   # the other foot's strike was missed: this is no step
            elif any(gap_between(f, prev.mid_stance, step.mid_stance) for f in present):
                spanning_gaps += 1
            else:
                step.interval = step.mid_stance - prev.mid_stance
        last = last_by_foot.get(step.foot)
        if last is not None:
            if gap_between(step.foot, last.mid_stance, step.mid_stance):
                spanning_gaps += 1   # a strike could be hiding in the gap
            else:
                step.stride = step.mid_stance - last.mid_stance
        last_by_foot[step.foot] = step

    # A "stride" far off the runner's own median is not a stride of continuous
    # running: a pause (walking, standing) or a missed strike sits inside it.
    # Measured on real footage with a mid-clip pause, keeping them took the
    # cadence from 266 to 168 spm. Dropped and counted, not averaged in.
    irregular = _drop_irregular(strikes, "stride", s.stride_outlier_ratio)
    irregular += _drop_irregular(strikes, "interval", s.stride_outlier_ratio)

    planted = {foot: np.isfinite(clearance[foot]) & (clearance[foot] < contact_level)
               for foot in FEET}
    airborne = np.ones(n, dtype=bool)
    for foot in present:
        airborne &= ~planted[foot]
    airborne &= np.logical_and.reduce([~gap[f] for f in present])

    # Scaled like the limb. The reference compared the unscaled nose x against
    # aspect-scaled hip x, so "facing" was wrong on most landscape clips.
    nose_x = _densify(joint_series(frames, "nose")[0], indices, n, max_gap) * scale_x
    heading, facing = _heading({f: limb[f] for f in present}, planted, nose_x)
    heading_conflict = bool(heading and facing and heading != facing)

    knee_flexion = {foot: _knee_flexion(limb[foot]["hip"], limb[foot]["knee"],
                                        limb[foot]["ankle"])
                    for foot in FEET}
    reach = max(1, int(round(s.knee_search_seconds * fps)))
    edge_hits = knee_skipped = 0
    for step in strikes:
        series = knee_flexion[step.foot]
        seen = observed[step.foot]
        step.knee_frame = step.frame
        step.knee_time = step.frame / fps
        step.knee_offset = 0.0
        lo, hi = max(0, step.frame - reach), min(len(series), step.frame + reach + 1)
        # Only frames the model actually returned may shape a landmark.
        if not seen[lo:hi].all():
            knee_skipped += 1
            continue
        at, value = _vertex(series, step.frame, reach)
        if abs(at - step.frame) >= reach - 1e-6:
            edge_hits += 1
        step.knee_time = at / fps
        step.knee_frame = int(np.clip(round(at), 0, n - 1))
        step.knee_offset = (at - step.frame) / fps
        step.knee_flexion = value

    quality = {
        "absent_feet": absent,
        "ankle_coverage": {foot: round(coverage[foot], 3) for foot in FEET},
        "frames_not_observed": {foot: int(n - int(observed[foot].sum())) for foot in FEET},
        "frames_in_long_gaps": {foot: int(gap[foot].sum()) if foot in present else n
                                for foot in FEET},
        "untimed_strikes": sum(1 for st in strikes if not np.isfinite(st.mid_stance)),
        "stances_hidden_by_gaps": hidden,
        "consecutive_same_foot_strikes": same_foot,
        "intervals_dropped_for_gaps": spanning_gaps,
        "irregular_intervals_dropped": irregular,
        "knee_landmarks_skipped_unobserved": knee_skipped,
        "knee_landmark_on_window_edge": edge_hits,
        "implausible_stance_flexion_frames": _implausible_stance(
            knee_flexion, strikes, fps, s.knee_stance_limit_deg),
        "heading_disagrees_with_facing": int(heading_conflict),
        "max_interp_gap_frames": max_gap,
    }

    return GaitAnalysis(
        fps=fps, n_frames=n,
        clearance=clearance, raw=raw, amplitude=amplitude,
        contact_level=contact_level, leg_length=leg_length, ground=ground,
        hip_height=(float(np.nanmean([ground[f] for f in present])) - mid_hip) / leg_length,
        knee_flexion=knee_flexion,
        limb=limb,
        steps=strikes,
        cadence_live=_densify_samples(_stride_ema_samples(strikes, s.cadence_smoothing), fps, n),
        _smoothing=s.cadence_smoothing,
        warnings=quality,
        airborne=airborne,
        heading=heading,
        facing=facing,
        frame_aspect=scale_x,
        observed=observed,
        present=present,
        min_strides_confident=s.min_strides_confident,
    )


def _swing_gates(segs, signal: np.ndarray, fps: float, *, high: float, low: float,
                 fraction: float | None, window_s: float = 2.0) -> dict[int, float]:
    """The height each swing segment must reach to count as a swing, by segment index.

    The reference used one clip-wide gate, ``high`` = a fraction of the typical
    swing. A runner who speeds up (or slows down) within the clip then loses
    their smaller swings, which merge two stances into one and halve the
    measured cadence; that happened on real footage. Here each swing is judged
    against the swings within ±*window_s*: ``fraction`` × their median height.
    The gate is never stricter than the reference's, and never below ``2 × low``,
    so ankle wobble at the bottom of a stance still does not count as a swing.
    """
    swings = [(k, (a + b) / 2 / fps, float(np.nanmax(signal[a:b])))
              for k, (a, b, lab) in enumerate(segs) if lab == 0 and np.isfinite(signal[a:b]).any()]
    gates = {}
    for k, centre, _ in swings:
        if fraction is None:
            gates[k] = high
            continue
        near = [h for _, c, h in swings if abs(c - centre) <= window_s and h >= 2 * low]
        local = fraction * float(np.median(near)) if near else high
        gates[k] = max(2 * low, min(high, local))
    return gates


def _drop_irregular(steps: list[Step], attr: str, ratio: float) -> int:
    """Set *attr* (``stride`` or ``interval``) to NaN where it is more than *ratio*
    times, or less than 1/*ratio* of, the median over the clip. Needs at least
    three values to have a median worth trusting; returns how many were dropped."""
    values = np.array([getattr(st, attr) for st in steps], float)
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return 0
    med = float(np.median(finite))
    dropped = 0
    for st in steps:
        v = getattr(st, attr)
        if np.isfinite(v) and not (med / ratio <= v <= med * ratio):
            setattr(st, attr, float("nan"))
            dropped += 1
    return dropped


def _strikes(signal: np.ndarray, foot: str, fps: float, *, high: float, low: float,
             min_frames: int, fraction: float | None = None) -> tuple[list[Step], int]:
    """Every timed foot strike on one foot's clearance signal, and how many
    stances could not be timed because a gap hides their start.

    Reference logic (stance/swing runs either side of one threshold; a swing
    that never reaches *high* is absorbed into stance; a stance shorter than
    *min_frames* into swing) with gaps handled explicitly. The reference labelled
    a NaN frame as swing and then, since that "swing" had no peak, absorbed the
    gap into a stance: a tracking dropout became ground contact. Here missing
    frames are their own label (-1) and never merge with either phase:

    * a stance that starts after a gap (or the clip start) has no observed
      strike, so it is not reported as one;
    * a stance that ends at a gap (or the clip end) has no toe-off: the strike
      counts, but contact time and mid-stance are NaN (untimed);
    * a swing next to a gap is never absorbed, since its peak may be in the gap.

    Two further changes, both found on real footage (see ``_swing_gates``):
    the swing-height gate is local rather than clip-wide when *fraction* is
    given; and a stance shorter than *min_frames* is kept when the ankle really
    reaches the ground inside it (below half the contact threshold), since only
    a shallow dip is the threshold-clipping the minimum guards against. Short,
    deep contacts are real at high cadence or in sped-up footage.
    """
    if high <= 0 or len(signal) < 3:
        return [], 0
    finite = np.isfinite(signal)
    labels = np.where(finite, (signal < low).astype(int), -1)

    for _ in range(2):
        changed = False
        segs = _segments(labels)
        gates = _swing_gates(segs, signal, fps, high=high, low=low, fraction=fraction)
        for k, (start, end, label) in enumerate(segs):
            touches_gap = ((k > 0 and segs[k - 1][2] == -1) or
                           (k + 1 < len(segs) and segs[k + 1][2] == -1))
            if label == 0 and not touches_gap and float(np.max(signal[start:end])) < gates[k]:
                labels[start:end] = 1
                changed = True
            elif label == 1 and end - start < min_frames and not touches_gap \
                    and 0 < start and end < len(labels) \
                    and float(np.min(signal[start:end])) > 0.5 * low:
                labels[start:end] = 0
                changed = True
        if not changed:
            break

    steps: list[Step] = []
    hidden = 0
    segs = _segments(labels)
    for k, (start, end, label) in enumerate(segs):
        if label != 1:
            continue
        if start == 0 or labels[start - 1] != 0:
            if start != 0:
                hidden += 1
            continue
        strike = _cross_time(signal, start - 1, start, low, fps)
        timed_exit = end < len(signal) and labels[end] == 0
        toe_off = _cross_time(signal, end - 1, end, low, fps) if timed_exit else float("nan")

        # The swing that led into this strike, back to the previous stance or gap.
        prev_end = max((b for a, b, lab in segs[:k] if lab != 0), default=0)
        window = signal[prev_end:start]
        ok = np.isfinite(window)
        peak = float(window[ok].max()) if ok.any() else float("nan")
        peak_i = prev_end + int(np.nanargmax(window)) if ok.any() else start
        mid = (strike + toe_off) / 2 if np.isfinite(toe_off) else float("nan")
        steps.append(Step(
            number=0, foot=foot, time=strike, frame=0, mid_stance=mid,
            peak_time=peak_i / fps, peak_clearance=peak,
            contact_seconds=toe_off - strike if np.isfinite(toe_off) else float("nan"),
        ))
    return steps, hidden


def _implausible_stance(knee_flexion: dict, steps: list, fps: float,
                        limit: float) -> int:
    """Frames inside a detected ground contact where the knee is impossibly bent.

    Run on the **unsmoothed** flexion. A planted knee does not exceed about 70
    degrees; anything near the swing peak (100-120) while the foot is supposed
    to be on the ground is the pose, not the runner. Smoothing would defeat the
    check -- it pulled this clip's 113-degree spike down to 76. On this clip it catches the stretch where ViTPose
    has swapped the left and right leg keypoints as the legs scissor past each
    other in the image.

    This replaced a frame-to-frame swap detector, which was the textbook test
    and did not work here: it compares each frame's assignment against the
    previous frame's, so a swap *sustained* over several frames looks perfectly
    continuous from the inside, and at the transition the two legs are close
    enough together that both assignments cost about the same. It returned zero
    on a clip with a known swap, which is worse than not testing -- a detector
    that cannot fail is indistinguishable from a clean result.

    Counted, not repaired: rewriting a model's output on a heuristic risks
    replacing a visible problem with an invisible one.
    """
    bad = 0
    for step in steps:
        if not np.isfinite(step.contact_seconds):
            continue
        series = knee_flexion.get(step.foot)
        if series is None:
            continue
        lo = int(round(step.time * fps))
        hi = int(round((step.time + step.contact_seconds) * fps))
        window = series[max(0, lo):min(len(series), hi + 1)]
        bad += int(np.sum(np.isfinite(window) & (window > limit)))
    return bad


def _stride_ema_samples(steps: list[Step], alpha: float) -> list[tuple[float, float]]:
    """The live cadence reading, once per measured stride: ``(time, spm)``.

    An exponential average of same-foot strides (the reference averaged
    ``marks[i] - marks[i-2]``, which assumes the feet alternate and no step was
    lost; that breaks across a gap). *alpha* weights the newest stride.
    """
    out: list[tuple[float, float]] = []
    average = None
    for st in steps:
        if not np.isfinite(st.stride):
            continue
        average = st.stride if average is None else alpha * st.stride + (1.0 - alpha) * average
        out.append((st.mid_stance, 120.0 / average))
    return out


def _densify_samples(samples, fps: float, n: int) -> np.ndarray:
    """Hold each reading from its own moment until the next one.

    A step function, because nothing is known about the cadence between two
    steps. The panel smooths this for *display* by curving through the readings;
    the array itself stays honest, which is what gait.json records.
    """
    out = np.full(n, np.nan)
    for t, value in samples:
        start = int(min(n, np.ceil(t * (fps or 30.0))))
        out[start:] = value
    return out


def summary_text(analysis: GaitAnalysis, *, video_seconds: float, source: str = "") -> str:
    """A plain-text report. The reference's report quoted numbers measured on its
    author's own clip in its explanatory notes; those are replaced by notes that
    hold for any clip."""
    def f(value, fmt, unit=""):
        return "—" if value is None or not np.isfinite(value) else f"{value:{fmt}}{unit}"

    q = analysis.warnings
    lines = ["Running gait summary", "=" * 60]
    if source:
        lines.append(f"{'source':<26}{source}")
    lines += [
        f"{'clip length':<26}{video_seconds:.2f} s",
        f"{'foot strikes':<26}{analysis.count} ({len(analysis.steps_for('left'))} left, "
        f"{len(analysis.steps_for('right'))} right)",
        f"{'same-foot strides':<26}{analysis.stride_count}",
        f"{'cadence':<26}{f(analysis.mean_cadence, '.1f', ' steps/min')}"
        f"  [{analysis.cadence_confidence}]",
        f"{'step time':<26}{f(analysis.mean_step, '.3f', ' s')}",
        f"{'stride time':<26}{f(analysis.mean_stride, '.3f', ' s')}",
        f"{'stride variability':<26}{f(analysis.stride_spread() * 1000, '.1f', ' ms')}",
        f"{'contact time (proxy)':<26}{f(analysis.contact_mean('left'), '.3f', ' s')} left, "
        f"{f(analysis.contact_mean('right'), '.3f', ' s')} right",
        f"{'airborne':<26}{f(analysis.flight_ratio * 100, '.0f', '% of the clip')}",
        "",
        "Knee (projected 2D angle, 0° = straight leg)",
        "-" * 60,
    ]
    for foot in FEET:
        shape = analysis.knee_shape(foot)
        lines.append(f"{'flexion near contact':<26}{foot}: " + (
            f"{shape.flexion:.1f}° ± {f(shape.flexion_sd, '.1f')}° over {shape.n} strikes"
            if shape else "not measured (too few fully observed strikes)"))
    lines += [
        "",
        "Quality",
        "-" * 60,
        f"{'absent feet':<26}{', '.join(q['absent_feet']) or 'none'}",
        f"{'ankle coverage':<26}left {q['ankle_coverage']['left']:.0%}, right "
        f"{q['ankle_coverage']['right']:.0%}",
        f"{'untimed strikes':<26}{q['untimed_strikes']}  (stance cut by a gap or the clip end)",
        f"{'stances hidden by gaps':<26}{q['stances_hidden_by_gaps']}",
        f"{'intervals dropped':<26}{q['intervals_dropped_for_gaps']}  (span a tracking gap)",
        f"{'missed opposite strikes':<26}{q['consecutive_same_foot_strikes']}",
        "",
        "Per step",
        "-" * 60,
        f"{'#':>3} {'foot':<6} {'strike s':>9} {'step s':>8} {'stride s':>9} {'contact s':>10}"
        f" {'knee °':>7}",
    ]
    for st in analysis.steps:
        lines.append(f"{st.number:>3} {st.foot:<6} {st.time:>9.3f} {f(st.interval, '8.3f'):>8} "
                     f"{f(st.stride, '9.3f'):>9} {f(st.contact_seconds, '10.3f'):>10} "
                     f"{f(st.knee_flexion, '7.1f'):>7}")
    lines += [
        "",
        "Notes",
        "-" * 60,
        "Cadence = 120 / mean same-foot stride time, timed between mid-stance landmarks.",
        "Strikes are timed from the ankle keypoint crossing a clearance threshold; COCO-17",
        "has no heel or toe, so this does not classify heel/forefoot strike, pronation, or",
        "measure belt speed. Contact time is time under the threshold: a proxy.",
        "Knee angles are projected into the image plane, most meaningful from a true side",
        "view; they are not anatomical joint angles.",
        "Gaps longer than the interpolation limit are not filled; strides spanning them",
        "are not measured.",
    ]
    return "\n".join(lines) + "\n"

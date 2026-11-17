# running

Cadence and foot-strike timing for one runner filmed side-on. Adapted from
[vision-demos/running](https://github.com/jeremyipark/vision-demos/tree/c06cfb78ccd8e2c63629222695aac710fe4c199c/running)
(Apache-2.0); see its `cadence-explained.md` for the derivation and `../docs/UPSTREAM.md` for the changes.

## Method

1. **Pose** (ViTPose, every frame). Hip, knee and ankle per leg. A mask records the frames the model actually returned. Gaps up to 0.1 s are interpolated; longer ones stay gaps.
2. **Units**: x is multiplied by the frame's width/height, so both axes are in frame heights before any length or angle is taken. Leg length is the median thigh plus the median shank.
3. **Clearance**: each ankle's height above its *own* stance level (90th-percentile ankle y), in leg lengths, median- and mean-smoothed within observed runs.
4. **Strikes**: stance and swing runs either side of a shared threshold (18% of the smaller swing amplitude). A swing must reach 55% of the amplitude, and a stance must last ≥ 0.10 s. Crossing times are interpolated to sub-frame precision. A stance after a gap has no observed strike; a stance cut by a gap or the clip end is counted but untimed.
5. **Cadence** = 120 ÷ mean same-foot stride, timed between mid-stance landmarks. Strides that span a tracking gap are not measured. With fewer than 4 strides the cadence is flagged low-confidence; with none it is not reported.
6. **Knee**: projected flexion (0° = straight) at its parabola-fitted minimum near each strike. It is measured only when every frame in the window was actually observed.

Not measured, because COCO-17 has no heel or toe: foot-strike type, pronation, belt speed.

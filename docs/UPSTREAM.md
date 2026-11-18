# Upstream: jeremyipark/vision-demos

Reference revision: **`c06cfb78ccd8e2c63629222695aac710fe4c199c`** (2026-11-05,
"basketball: add rim, trajectory, form, highlights, compilation and social tools"),
Apache-2.0. The upstream repository was only read, never modified. Only the
basketball and running demos were used. The other demos, the Gradio app, and the
basketball helper scripts (`history.py`, `highlights.py`, `social.py`, …) were not copied.

## Adapted files

| here | upstream | what changed |
|---|---|---|
| `basketball/src/shots.py` | `basketball/src/shots.py` | `judge()` evidence rules (below); explicit unknown; `find_release` reports its method; `streaks()` treats unknown as neither a make nor a miss; `tally()` added |
| `basketball/src/ball.py` | `basketball/src/ball.py` | request and cache-key code moved out; selection unchanged |
| `basketball/src/hoop.py` | `basketball/src/hoop.py` | request and cache-key code moved out; rim parsing unchanged |
| `basketball/src/court.py` | `basketball/src/court.py` | unchanged |
| `basketball/src/render.py` | `basketball/src/render.py` | skeleton, rim and ball trail kept. The court panel, fire effects, credit and elbow strip are replaced by a scoreboard bar that shows unknown separately; the shot chart is drawn with OpenCV |
| `basketball/src/export.py` | `basketball/src/export.py` | unknown kept separate; FG% denominator stated; times in original-clip seconds with source frame numbers |
| `basketball/pipeline.py` | `basketball/main.py` | a function with explicit settings, provider and cache instead of a script over a mutated `config` |
| `basketball/settings.py` | `basketball/config.py` | a frozen dataclass; new `evidence_gap_seconds`, `min_flight_seconds` |
| `running/src/gait.py` | `running/src/gait.py` | observed mask, gap handling, absent limbs, heading fix, summary text (below) |
| `running/src/render.py` | `running/src/render.py` | skeleton and strike flash kept; panel and comet trails replaced by a HUD; clearance chart redrawn |
| `running/pipeline.py` | `running/main.py` | a function, as for basketball |
| `running/settings.py` | `running/config.py` | a frozen dataclass; new `max_interp_gap_seconds`, `min_strides_confident`, `min_limb_coverage` |
| `*/src/pose.py` | `*/src/pose.py` | reply parsing only; the request moved to `src/providers/gateway.py` |
| `*/src/providers/gateway.py` | the `request_*` functions | same request contracts, collected in one module |
| `*/src/video.py` | `basketball/src/video.py` | trim start offset, constant frame rate, preview rendition, FFmpeg check; HDR tone-mapping dropped |
| `*/src/skeleton.py` | `basketball/src/skeleton.py` | unchanged |

## Differences from the reference

**Basketball**
1. **Tracking loss is not a make.** Upstream, a ball that crossed inside the rim and was never seen again was called a make ("into the net"). Here that is **unknown**.
2. **Bounded evidence gaps.** Upstream bridged a gap of any length between the rim crossing and the next sighting, so a ball seen next in someone's hands could pass the "slowed by the net" check. Here every gap after the crossing must be ≤ 0.3 s, and the first sighting below the net must be at the net exit (within a rim width). Otherwise the shot is unknown.
3. A crossing hidden at the rim by a long gap is **unknown**, not "never came down through the rim". A ball lost while still above the rim is unknown. "Never above the rim" is a miss only with ≥ 0.8 s of gap-free tracking after release.
4. **FG% = makes ÷ (makes + misses).** Upstream divided by every attempt. Unknown shots are shown separately everywhere: tiles, timeline, video scoreboard, chart (hollow amber), CSV and JSON. The denominator is stated in each place.
5. Streaks: an unknown neither extends nor resets a run. The fire effects are not ported; on a streak the rim turns orange and the scoreboard shows "HOT xN".
6. The output video is the clip with a scoreboard bar. Upstream stacked an animated court panel under the clip. The court chart is a separate PNG.
7. Release time records whether it came from the ball or from the pose fallback.
8. With several people tracked, the subject is the person with both wrists overhead on the most frames. Upstream kept the largest person, which followed a bystander on real game footage (docs/EVALUATION.md).

**Running**
1. **Observed mask before interpolation.** Upstream `_joint_xy()` computed `observed` after `_densify()` had already interpolated, so every frame between the first and last sighting counted as observed and the knee-landmark gate never tripped. Here the mask comes from the model's actual returns.
2. **Gaps stay gaps.** Upstream interpolated every gap, however long. Here only gaps ≤ 0.1 s are filled. Longer ones stay NaN through smoothing and strike detection. Upstream also labelled missing frames as swing, then folded that peakless "swing" into stance, which turned a dropout into ground contact. Missing frames are now their own label.
3. Strides and step intervals that span a tracking gap, or a missed opposite-foot strike, are not measured. The live cadence uses measured same-foot strides only.
4. There is no cadence fallback to first-to-last step timing: with no stride, cadence is "not measurable". With fewer than 4 strides it is flagged low-confidence.
5. A leg seen on fewer than 50% of frames is reported absent and left out. Upstream it crashed or produced strikes from interpolation. If neither leg is usable, the user gets a clear error.
6. Heading fix: upstream compared the unscaled nose x with aspect-scaled hip x, so "facing" was wrong on most landscape clips and could raise a false "disagrees" warning.
7. The plain-text report is rewritten. Upstream's report quoted measurements from its own sample clip in its explanatory notes (for example, ankle descent speeds and a 23% threshold sweep), which would be wrong for any other clip.
8. Found on real footage (docs/EVALUATION.md): the swing-height gate is local (±2 s) instead of clip-wide, so a runner who speeds up keeps their early strikes. A stance shorter than the minimum contact is kept when the ankle reaches the ground inside it. Strides beyond 1.4× the median (a pause or a missed strike) are dropped. Together these took real-clip cadence from 198 to 254 spm, against a ~261 spm reference.
9. Step symmetry is not exported, consistent with upstream's own note that it cannot be resolved from one side view. The phase split and contact balance remain in `gait.json`.

**Shared**
- Proper packages (`basketball.*`, `running.*`) with explicit settings dataclasses, instead of global `config`/`main`/`src` modules mutated per run. Each job still runs in its own process (`webapp/worker.py`) so it can be cancelled with a hard kill and a crash stays contained.
- Model-provider calls live only in `src/providers/`. The fixture provider answers in the same reply shapes for the bundled synthetic clip only.
- Cache keys hash video **content**, the model, every request parameter, the provider and a contract version. Upstream used the file name, size and mtime.
- Trimming carries a source-frame offset. All reported times and frame numbers refer to the original upload.
- The default inference height is 720 px (upstream 1080) to cut upload size. It is configurable in `settings.py`.
- Not ported: HDR tone-mapping, the elbow strip figure, credit lines, Gradio UI and helper scripts.

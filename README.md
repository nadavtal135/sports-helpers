# Sports Helpers

Review basketball shots, running strides, squats and standing jumps in the browser with a Python backend.

- **Basketball** (`basketball/`): finds every shot from the shooter's pose and calls each one a **make, miss or unknown** by following the ball through the rim. It plots shots on an estimated court chart, reports FG% over judged shots, measures projected elbow angles on side-on shots, and exports per-shot CSV/JSON.
- **Running** (`running/`): times every foot strike from the ankles and reports **cadence** (120 ÷ mean same-foot stride), step, stride and contact timing, and projected knee flexion near strike. It produces an annotated video, an ankle-clearance chart, CSV/JSON exports and a text report, and warns when tracking is poor.
- **Squat Analysis** (`squat/`) leads with technique observations for one side-view lifter. Compare the supporting video frames, toggle the pose overlay and check reps against your own depth and tempo targets.
- **Standing Jump Analysis** opens with frame-linked observations about takeoff, landing and recovery. Compare attempts, step through the shoes meeting the floor, or check a personal technique target. [Standing jump folder and guide](standing_jump/)
- **Web app** (`webapp/`): choose a sport → upload → preview and trim → guidance and settings → analyze (progress, cancel) → review (video, metrics, charts, clickable timeline) → download.

Pose comes from `vitpose-plus-large` (17 COCO keypoints). Basketball also uses SAM 3.1 for the hoop and ball. Both are called through the [VLM Run Gateway](https://www.vlm.run/gateway) (OpenAI-compatible). The analysis methods are adapted from [jeremyipark/vision-demos](https://github.com/jeremyipark/vision-demos) (Apache-2.0); see [NOTICE](NOTICE) and [docs/UPSTREAM.md](docs/UPSTREAM.md) for what was reused and changed.

## Setup

You need Python 3.11+ and **FFmpeg**. Install FFmpeg with `winget install Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS, or `sudo apt install ffmpeg` on Debian/Ubuntu.

**Quick start:**
- **Windows:** double-click `start.bat`.
- **macOS/Linux:** run `./start.sh`.

The first run creates `.venv` and installs the packages. The launcher (`run.py`) then checks Python, FFmpeg and the packages, picks a free port starting at 8000, and opens your browser. If something is missing, it says exactly what to install.

**Manual setup:**

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python run.py            # or: python -m webapp.server
```

### Fixture mode (default, no credentials)

With no `VLMRUN_API_KEY`, the app runs in **fixture mode**. Choose a sport, click **Use the synthetic sample clip**, and run the full flow. Basketball's sample has 3 makes, 1 rim-out and 1 tracker dropout. Running's sample has a cadence of 172 spm and a 0.4 s tracking gap. The squat sample has three completed reps (deep, paused and shallow) followed by a fourth attempt with a 0.7 s tracking gap around the bottom. The standing jump sample has four attempts, including a recovery step, a smaller landing knee bend and an obscured contact. Model replies are simulated in the gateway's reply format, so the real parsing and analysis code runs. Results carry a "simulated" banner. Fixture mode **refuses any other video** (checked by content hash), so simulated results can never appear to describe your footage. Your own clips can still be uploaded and previewed.

### Live mode

```bash
cp .env.example .env    # then set VLMRUN_API_KEY=...
python -m webapp.server
```

With a key the badge reads **Live mode**. Live analysis **uploads the trimmed clip to the model provider** (VLM Run). The UI says so and asks for consent before each run. Calls are billed per frame, so clips are capped at 180 s after trimming (`SPORTS_HELPERS_MAX_SECONDS`). The key stays on the server: it is passed only to the analysis worker's environment, never sent to the browser, and redacted from worker logs. `.env` is git-ignored. Replies are cached in `data/cache/`, so re-running a clip with the same settings makes no new calls. The synthetic sample always uses fixture replies, even in live mode.

## Filming guidance

| | Basketball | Running | Squat Analysis | Standing Jump Analysis |
|---|---|---|---|---|
| Subjects | one shooter, one ball | one runner | one lifter, bodyweight or barbell | one person doing two-foot standing vertical jumps |
| Camera | still (tripod), landscape | still, level, about hip height | still, level, side-on at hip height | fixed, level and directly side-on |
| In frame | whole hoop (rim and net) and the shooter's full body | both ankles, side-on view | full body with the near shoulder, hip, knee and ankle visible | whole body and arm swing, both shoes, floor and room above the jump |
| Timing | each shot needs the ball visible to the rim | ≥ 5–10 s of steady running | include a brief standing pause before the first rep and after the last | start from standing and keep recording through the final recovery |

### Reviewing a squat set

Choose **Squat Analysis** and upload the iPhone MOV or MP4, or use the synthetic sample. Keep the standing pauses inside the trim and select **Analyze**. Results open with one or two observations about what you repeated and what to look at more closely. Select an observation to pause on its evidence. For the full synthetic sample, compare the bottom frames of rep 3 (7.40 s) and rep 1 (1.80 s). **Show pose overlay** switches both views at the same frame.

Select **My depth and tempo targets** to compare each completed rep with a bottom band or phase duration of your choice. The timeline lets you pause during the descent, at the bottom or during the ascent. Open **Rep measurements** for timing and projected angles. [Squat technique review specification](squat/README.md)

| Metric | How to read it |
|---|---|
| Completed reps | The detector selects one side for the clip using leg coverage, joint confidence and shoulder visibility. It smooths hip height above the ankle and projected knee flexion, then looks for a departure from standing, a bottom reversal and a sustained return to standing. A bottom pause or small bounce stays within the same attempt. A shallow rep can complete the full cycle. |
| Incomplete and uncertain | A missing standing start or finish at the clip boundary gives an incomplete attempt. Interior tracking gaps up to 0.10 s can be interpolated for motion detection. A longer gap within an attempt gives one uncertain review interval between standing observations. Its rep count is unknown. Only completed reps increment the video counter. |
| Start, bottom and end | Start is the last frame inside the standing band before descent. End is the first frame of a confirmed return to that band. The bottom time is near the lowest smoothed hip height, at the middle of the deepest plateau during a hold. Descent runs from start to bottom and ascent from bottom to end, so a bottom pause contributes to the elapsed times. |
| Peak knee flexion | Maximum smoothed hip–knee–ankle flexion during the completed rep, projected in the image plane. A straight leg is 0°. |
| Torso lean | Angle between the hip-to-shoulder line and upward image vertical at the bottom. An upright torso is 0°. |
| Depth (2D estimate) | Height of the visible hip keypoint relative to the visible knee at the bottom. The labels are **above**, **roughly level** (within 4% of projected leg length), **below** and **unknown**. Unreliable or missing bottom keypoints give unknown depth. |

### Reviewing a standing jump

Choose **Standing Jump Analysis**, upload a side-view MOV or MP4, or select **Use the synthetic sample clip**. Keep the pre-jump stance and the landing inside the trim. Choose **Analyze**. Results open with up to two observations and the frames that support them.

In the sample, select the landing comparison to see the deepest landing frames for attempts 3 and 1 at **7.93 s** and **2.23 s**. Use **Inspect a frame** to see first contact, switch the pose overlay off to see the shoes, or step by one frame. The timeline also replays the step after attempt 2 and opens the interval where the feet disappear on attempt 4. **My jump technique target** compares attempts with your own range for descent time, added landing knee bend or torso lean. [Phase definitions and real-footage checks](standing_jump/README.md)

## Outputs

**Basketball**: `basketball_annotated.mp4` (pose, ball trail, rim lit green/red/amber per verdict, and a running scoreboard: makes, misses, unknown, FG% with its denominator) · `shot_chart.png` (estimated spots) · `ball_paths.png` (each ball path against the rim line) · `shots.csv` (one row per shot, including the evidence behind each call) · `frames.csv` · `analysis.json`.

**Running**: `running_annotated.mp4` (pose, strike flashes, live cadence, tracking-gap notice) · `clearance.png` · `steps.csv` · `frames.csv` (includes per-frame `observed` flags) · `gait.json` (summary, quality counters, per-frame series) · `summary.txt`.

**Squat Analysis** produces frame-aligned `squat_clean.mp4` and `squat_annotated.mp4` renditions. The annotated video highlights the measured side and shows phase and completed count. `reps.csv` and `reps.json` contain one row per detected attempt or uncertain interval, its status, review bounds, available landmarks and metrics. The JSON includes the default technique observations and their evidence frames. The results screen also offers the current personal target review as a JSON download. Unavailable numeric measurements are blank in CSV and null in JSON.

**Standing Jump Analysis** exports `standing_jump_clean.mp4`, `standing_jump_annotated.mp4`, `jumps.csv` and `jumps.json`. The JSON includes each available phase, its original frame and timestamp, the source of the takeoff and landing estimates, capture warnings and the observations with their supporting frames. CSV has one row per attempt. Missing measurements are null in JSON and blank in CSV.

All times are seconds in the **original uploaded clip**, with source frame numbers, even when trimmed.
Frame indices use the constant-rate analysis grid. For variable-frame-rate phone recordings, use the timestamps to locate landmarks in the original.

## Command-line use

The pipelines are plain functions:

```python
from pathlib import Path
from running import pipeline, settings
from running.src.cache import Cache
from running.src.providers.gateway import GatewayProvider

gw = settings.Gateway()
result = pipeline.run(Path("clip.mov"), Path("out/"), settings=settings.Settings(), gateway=gw,
                      provider=GatewayProvider(KEY, base_url=gw.base_url, timeout=gw.request_timeout),
                      cache=Cache(Path("data/cache/running")))
```

## Architecture

```
basketball/              running/                 webapp/
  pipeline.py  (orchestration)                      server.py   FastAPI API + static UI
  settings.py  (frozen dataclass, user-settable fields, guidance)   jobs.py     uploads, validation, job processes
  sample.py    (synthetic fixture clip)            worker.py   one process per job
  src/                                              static/     HTML/CSS/JS, no build step
    providers/gateway.py   VLM Run calls (the only network code)
    providers/fixture.py   simulated replies for the synthetic clip
    pose.py, video.py, cache.py, skeleton.py
    shots/ball/hoop/court  |  gait               analysis (no I/O)
    render.py, export.py                         video, charts, files
```

- Provider calls, analysis and rendering are separate modules. Analysis functions take arrays and return results, which is what the tests exercise.
- Squats and standing jumps have separate top-level packages, `squat/` and `standing_jump/`, each with its own detector, review logic, fixture and exports. Both reuse Running’s trimmed video preparation, ViTPose client, pose cache and side-view rendering. Their web reviews share frame-aligned players and comparison controls.
- Each job runs in its own process with its own folder (`data/jobs/<id>/`). Cancel is a hard kill, and partial outputs are deleted. Up to `SPORTS_HELPERS_MAX_JOBS` jobs run at once, and the rest queue.
- The cache is content-addressed: video bytes + model + request parameters + provider + contract version. Writes are atomic.
- Each job's state is saved to `status.json`, so finished results survive a server restart. A job that was running when the server stopped is reported as failed. Uploads and jobs older than 7 days are swept at startup.
- The results URL (`#job=<id>`) restores a job after a page reload. **Download all** zips every output.

## Tests

```bash
python -m pytest -q
```

The suite covers:

- **Basketball `judge()`**: synthetic tracks for a make, a make with net occlusion, an outside miss, a rim-out, falling past the net, in front of the rim, no observations, a short track, an airball, the ball lost after the crossing, long gaps, a hidden crossing, a non-exit sighting, and `until` bounds. Also `tally()`/FG% and streaks with unknowns.
- **Basketball pipeline**: the end-to-end synthetic clip matches the scripted outcomes, stats stay consistent, trim offsets carry through, and recorded gateway replies parse in `tests/fixtures/`.
- **Running**: cadence from a known periodic signal (with and without noise), aspect-ratio correction, observed vs interpolated masks, interpolated frames never becoming knee landmarks, long-gap handling, gaps never becoming contact, a missing limb, both limbs missing, a truncated final stance, insufficient strides, and JSON safety.
- **Squats** cover complete, paused and shallow reps, both clip boundaries, bottom bounces and tracking gaps, side selection, missing measurements, trimmed timestamps, per-attempt exports, footage-backed comparisons and optional targets. The browser test exercises frame seeking, overlay switching and target editing in fixture mode.
- **Standing jumps** cover clean and repeated attempts, a grounded dip, a recovery step, obscured shoes, cut-off landings, oblique views and shoe-edge timing. The browser test uploads the fixture, trims it, opens the technique observations and checks the decoded frames, phase controls, pose toggle and personal targets.
- **Web** covers sample upload → job → results → downloads for all four sports, fixture-mode refusal of real uploads, bad inputs, cancel, and concurrent isolated jobs.
- **Live contract**: request shapes against a fake client (no network), the missing-key error, the consent requirement and the length cap.

## Limitations

- **Real-footage evaluation is minimal.** See [docs/EVALUATION.md](docs/EVALUATION.md). On one live running clip, cadence lands within 1–3% of a pixel-based reference, but the fixes were found on that same clip. Basketball make/miss accuracy is unevaluated. Labeled clips are needed: tripod basketball sessions with a make/miss list, and side-on runs with cadence from a footpod, watch or metronome.
- Basketball: one shooter, one ball, a still camera. Shots are found from the pose, so a shot out of frame is missed. Court spots are estimates scaled by the shooter-height and field-of-view settings. Elbow angles are 2D projections, shown for side-on shots only, and are not anatomical joint angles. The stricter evidence rules will mark more shots unknown than upstream did, especially when the net hides the ball for long.
- Running: COCO-17 has no heel or toe keypoints, so there is **no heel/forefoot strike classification, no pronation, and no belt speed**. Contact time is time under a clearance threshold (a proxy). Knee angles are projected. Left/right timing differences from one side view mix runner and camera geometry.
- **Squat validation is synthetic.** Camera roll and oblique views change the projected angles and depth. Real side-view iPhone sets with manually marked reps and landmarks are needed to check counts, timing, depth bands and tracking through barbell occlusion across lifters and camera positions.
- **Standing jump validation** includes a synthetic browser review and a frame-by-frame comparison on two real 60 fps recordings using a local COCO-17 pose model. The live VLM Run path has an offline contract test. See the [manual timing comparisons and remaining validation](standing_jump/README.md#real-footage-check).
- HDR (iPhone HLG/Dolby Vision) clips are converted without tone-mapping and may look washed out in the annotated video. Analysis is unaffected beyond what the model sees.

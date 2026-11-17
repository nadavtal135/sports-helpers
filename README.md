# Sports Helpers

Basketball and running video analysis in the browser, with a Python backend.

- **Basketball** (`basketball/`): finds every shot from the shooter's pose and calls each one a **make, miss or unknown** by following the ball through the rim. It plots shots on an estimated court chart, reports FG% over judged shots, measures projected elbow angles on side-on shots, and exports per-shot CSV/JSON.
- **Running** (`running/`): times every foot strike from the ankles and reports **cadence** (120 ÷ mean same-foot stride), step, stride and contact timing, and projected knee flexion near strike. It produces an annotated video, an ankle-clearance chart, CSV/JSON exports and a text report, and warns when tracking is poor.
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

With no `VLMRUN_API_KEY`, the app runs in **fixture mode**. Choose a sport, click **Use the synthetic sample clip**, and run the full flow. The sample is a generated scene with known outcomes (basketball: 3 makes, 1 rim-out, 1 simulated tracker dropout that must come out *unknown*; running: 172 spm with a 0.4 s tracking gap). Model replies are simulated in the gateway's exact reply format, so the real parsing and analysis code runs. Results carry a "simulated" banner. Fixture mode **refuses any other video** (checked by content hash), so simulated results can never appear to describe your footage. Your own clips can still be uploaded and previewed.

### Live mode

```bash
cp .env.example .env    # then set VLMRUN_API_KEY=...
python -m webapp.server
```

With a key the badge reads **Live mode**. Live analysis **uploads the trimmed clip to the model provider** (VLM Run). The UI says so and asks for consent before each run. Calls are billed per frame, so clips are capped at 180 s after trimming (`SPORTS_HELPERS_MAX_SECONDS`). The key stays on the server: it is passed only to the analysis worker's environment, never sent to the browser, and redacted from worker logs. `.env` is git-ignored. Replies are cached in `data/cache/`, so re-running a clip with the same settings makes no new calls. The synthetic sample always uses fixture replies, even in live mode.

## Filming guidance

| | Basketball | Running |
|---|---|---|
| Subjects | one shooter, one ball | one runner |
| Camera | still (tripod), landscape | still, level, about hip height |
| In frame | whole hoop (rim and net) and the shooter's full body | both ankles, side-on view |
| Length | any; each shot needs the ball visible to the rim | ≥ 5–10 s of steady running |

## Outputs

**Basketball**: `basketball_annotated.mp4` (pose, ball trail, rim lit green/red/amber per verdict, and a running scoreboard: makes, misses, unknown, FG% with its denominator) · `shot_chart.png` (estimated spots) · `ball_paths.png` (each ball path against the rim line) · `shots.csv` (one row per shot, including the evidence behind each call) · `frames.csv` · `analysis.json`.

**Running**: `running_annotated.mp4` (pose, strike flashes, live cadence, tracking-gap notice) · `clearance.png` · `steps.csv` · `frames.csv` (includes per-frame `observed` flags) · `gait.json` (summary, quality counters, per-frame series) · `summary.txt`.

All times are seconds in the **original uploaded clip**, with source frame numbers, even when trimmed.

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
- **Web**: sample upload → job → results → downloads for both sports, fixture-mode refusal of real uploads, bad inputs, cancel, and concurrent isolated jobs.
- **Live contract**: request shapes against a fake client (no network), the missing-key error, the consent requirement and the length cap.

## Limitations

- **Accuracy on real footage has not been evaluated** here. The tests use synthetic signals and fixture replies. Labeled clips (shots with known make/miss; runs with known cadence, for example from a footpod or a metronome) are needed to measure it.
- Basketball: one shooter, one ball, a still camera. Shots are found from the pose, so a shot out of frame is missed. Court spots are estimates scaled by the shooter-height and field-of-view settings. Elbow angles are 2D projections, shown for side-on shots only, and are not anatomical joint angles. The stricter evidence rules will mark more shots unknown than upstream did, especially when the net hides the ball for long.
- Running: COCO-17 has no heel or toe keypoints, so there is **no heel/forefoot strike classification, no pronation, and no belt speed**. Contact time is time under a clearance threshold (a proxy). Knee angles are projected. Left/right timing differences from one side view mix runner and camera geometry.
- HDR (iPhone HLG/Dolby Vision) clips are converted without tone-mapping and may look washed out in the annotated video. Analysis is unaffected beyond what the model sees.

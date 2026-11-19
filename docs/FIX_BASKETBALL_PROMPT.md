You are working in the `sports-helpers` repo (Python + FastAPI app; read README.md first).
Fix the basketball analysis so it works on real footage.

## Hard rules
- **Do not run any git command** (no status, diff, add, commit, branch, stash, push, checkout).
  Leave every change as plain edits in the working tree; I will review and push them myself later.
- Do not delete or rewrite tests to make them pass. `python -m pytest -q` must pass at the end.
- Keep the existing code style: analysis functions stay pure (arrays in, results out),
  provider/network code stays in `basketball/src/providers/`.

## What went wrong (from a live-mode run on a real clip)
Clip: an 88 s, 576×1024 portrait TikTok shooting workout. The camera sits under the hoop
facing the shooter, the big rim is near the top of the frame, a second (far-court) hoop is
visible small on the back wall, and the video cuts between camera angles.
The app reported 3 shots, 0 makes, 2 misses, 1 unknown. In reality the clip has at least
15 makes, and the two "misses" went cleanly through the net.

1. **Wrong hoop chosen.** `basketball/src/hoop.py` `locate()` (around line 64) keeps the
   SAM item with the highest `score`. Here that was the small far-court hoop, so every ball
   that dropped through the real rim failed the "came down through the rim" check.
2. **Hoop located once, from one frame** (`basketball/pipeline.py`, section "3. hoop",
   the middle frame by default). Clips with camera cuts or a moving camera need more than one.
3. **Shooter posed on only 461 of 2639 frames** (warning from `pipeline.py`). Likely cause to
   check first: `pose.keep_track(frames, shooter)` keeps a single tracker `track_id`, and
   the id changes across camera cuts / re-detections, so most of the clip is thrown away.
   Only 3 shot candidates were found as a result.

## What to do
1. **Hoop choice.** Choose among all SAM hoop items by something better than score alone:
   prefer the largest rim width, and when ball tracks exist, prefer the hoop the ball
   trajectories actually approach (re-judging against the alternative is fine). Keep the
   current behaviour for single-hoop frames. Add a warning to the result when more than one
   hoop was found and say which was used.
2. **Manual override.** Add an optional user setting (in `basketball/settings.py`
   `USER_SETTINGS`, so it shows in the web UI) to pick the hoop: e.g. "auto / largest /
   highest score", or a hoop position. Keep it simple and validated like the other settings.
3. **Multiple hoop frames / camera cuts.** Detect scene cuts (e.g. a frame-difference or
   ffmpeg `scene` threshold) and locate the hoop per segment, using each shot's own segment
   when judging it. If that is too large a change, at least locate the hoop on a few frames
   and use the one nearest each shot, and warn when hoop positions disagree a lot.
4. **Pose coverage.** Make shooter tracking survive track-id changes (e.g. re-link by bbox
   IoU / position across gaps, or re-pick the shooter per camera segment), so coverage on a
   clip like this is far above 17%. Don't regress the multi-person case the current code
   handles (the shooter is the one raising both hands).
5. **Tests.** Add unit tests with synthetic inputs (no network): two hoops in one SAM
   payload (a large near rim and a small far one with a higher score) → the large one is
   chosen; the override setting is respected; tracks with a changing `track_id` keep the
   shooter's coverage; per-segment hoops are used when a cut is present. Follow the patterns
   in `tests/test_basketball_judge.py` and `tests/test_basketball_pipeline.py`, and the reply
   formats in `tests/fixtures/`. The existing synthetic-clip test must still give its scripted
   outcomes (3 makes, 1 rim-out, 1 unknown).
6. If you change what the cache key depends on (hoop prompt/frames), bump the relevant
   `contract` string in `pipeline.py` so stale cached replies are not reused.

## Checking
- `python -m pytest -q` must pass.
- If `VLMRUN_API_KEY` is set and I give you the clip, run it through the app in live mode
  (`python -m webapp.server`, or call `basketball.pipeline.run` directly) and report shots,
  makes/misses/unknown, pose coverage and which hoop was used. Live calls are billed per
  frame, so run it once, not repeatedly. If there is no key or clip, say so and stop at tests.

## When done
Give me a short summary: which files changed and why, what the tests cover, anything you
could not finish, and remaining limits. Remember: no git commands.

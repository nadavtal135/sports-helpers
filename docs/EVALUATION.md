# Evaluation on real footage (live gateway)

First live run against the VLM Run Gateway, 2026-11-18. Total model spend for
everything below: about **$0.04**.

This is **one running clip and one basketball clip**. It is a smoke test of the
live path and a source of failure modes, not an accuracy study. The running
fixes below were found on this clip and then scored on this same clip, so the
"after" numbers are optimistic. They need confirming on held-out footage.

## Footage

| clip | source | license | notes |
|---|---|---|---|
| Running | [SpeedyTime -3 – Treadmill 2](https://commons.wikimedia.org/wiki/File:SpeedyTime_-3_%E2%80%93_Treadmill_2.webm) (NASA) | public domain | Astronaut on the ISS treadmill, side-on, 1280×720 at 59.94 fps. Runs 32.3–37.7 s, stands still, runs again around 41 s. Probably sped up ("SpeedyTime"), so cadences are in video time. |
| Basketball | [20240301 Jackson McAndrew free throw video 01](https://commons.wikimedia.org/wiki/File:20240301_Jackson_McAndrew_free_throw_video_01.webm) | CC BY 4.0, User:TonyTheTiger | 2.3 s game free throw, 540×720 portrait, ten players, hand-held, hoop at the top edge. The clip ends before the ball reaches the rim. |

The clips are not committed to the repository.

## Running: cadence

**Reference.** Pose-free and independent of the app: the lag at which the leg-region pixels best repeat (frame self-similarity, parabolic sub-frame peak) is the stride period. The runner accelerates, so the bout is split in two:

| window (s) | reference stride | reference cadence |
|---|---|---|
| 32.3–34.6 | 0.501 s | 239.4 spm |
| 34.6–37.5 | 0.433 s | 277.0 spm |
| 32.3–37.5 | stride-weighted | ≈ 261 spm |

**App** (`cadence_spm`, live ViTPose poses, same windows):

| window (s) | before | after | error after |
|---|---|---|---|
| 32.3–34.6 | 237.4 | 237.4 | −0.8% |
| 34.6–37.5 | **216.3** | 281.2 | +1.5% |
| 32.3–37.5 | **198.2** | 254.3 (253.9 via the web app) | −2.6% |
| 28–42 (two runs with a standing pause between) | **149.7** | 241.4 | −7.5% vs the running bout |

**What went wrong, and the fixes** (`running/src/gait.py`):
1. *The swing-height gate was clip-wide* (55% of the typical swing). As the runner sped up, swings grew from ~0.22 to ~0.6 leg lengths. The early swings failed the gate and two stances merged, halving the cadence. **Fix:** each swing is judged against swings within ±2 s. The gate is never stricter than before, and never below 2× the contact threshold, so stance wobble still doesn't count.
2. *The minimum contact was a fixed 0.10 s.* Real 0.07 s contacts were discarded. **Fix:** a short stance is kept when the ankle actually reaches the ground inside it (below half the contact threshold). Shallow mid-swing dips are still absorbed.
3. *Pauses became giant "strides".* **Fix:** strides and step intervals more than 1.4× (or less than 1/1.4×) the clip median are dropped and reported in the warnings.

ViTPose keypoints were clean (each ankle's own signal repeats at the true stride). There were 3–7 frames of brief left/right swaps, which the existing implausible-knee check flags.

**Not evaluated:** contact time, knee angles, and real-world (not sped-up) running.

## Basketball

No suitable labeled footage was found (one shooter, still camera, outcomes visible), so **make/miss accuracy is unevaluated.**

On the free-throw clip:
- **Before:** the pose step kept the largest person, a bystander, so no shot was found.
- **Fix:** `pick_shooter` follows the tracked person with both wrists overhead on the most frames, falling back to the old rule.
- **After:** one shot is found (release 1.77 s, matching the video) and judged **unknown, "ball lost while still above the rim"**. That is correct: the outcome is not in the clip.
- The hoop (SAM 3.1) was found at the top edge (score 0.82). The ball was found on 26 of 38 tracked frames.

## Needed for a real evaluation

- **Basketball:** 3–5 sessions from a tripod, one shooter, the whole hoop in frame, about 20+ shots each, with a make/miss list.
- **Running:** several 20–30 s side-on clips at different speeds, with cadence from a footpod, watch or metronome, plus at least one outdoor (non-treadmill) clip.

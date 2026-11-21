# Squat technique review

The purpose of this feature is to help a lifter watch a rep, see what they repeated or changed, and choose what to try on the next one. It supports one person doing bodyweight or barbell squats in a still, side-on iPhone video. Rep counts help locate the movements in the recording.

## The results screen

| Question | What the lifter sees |
|---|---|
| What did I do well? | A repeatable feature of the measured reps, a personal target they met, or a fully visible rep they can use as a reference. |
| What should I look at more closely? | One specific difference from a chosen target, a difference between measured reps, or an observation made uncertain by the recording. |
| Where in the video can I see it? | Selecting an observation pauses at the supporting source frame. Comparisons show two labeled frames side by side, with the actual measurement and original-clip timestamp for each. A selector gives access to additional cited reps. |

The review leads with at most two observations. The timeline below gives each attempt its own descent, bottom and ascent controls. Each available control pauses on a frame in that phase. Playing a rep stops at its finish. Start and end landmarks, projected angles and elapsed times are available for closer inspection. The pose overlay can be switched off and on at the same frame in both views. The annotated download shows the phase prominently and the completed count underneath.

## What the comparisons mean

- **Visible depth** compares the height of the tracked hip and knee keypoints on the selected side at the bottom. A difference within 4% of projected leg length is roughly level. The other bands are above and below. The observation says when reps reached the same band. A shallower/deeper comparison also requires a separation of at least 8% of leg length when both offsets are available.
- **Torso lean** uses the hip-to-shoulder line relative to image vertical at the bottom. **Peak knee flexion** uses the largest smoothed projected hip–knee–ankle flexion during the rep, with 0° for a straight knee. Its evidence is the frame where the peak was checked. It can differ from the bottom frame.
- **Tempo** uses the time from departure from the standing band to the bottom landmark and from that landmark back to standing. A bottom hold is included in these elapsed times. The clickable phase frames are selected from visible movement within the corresponding phases.
- **Consistency** compares completed, measurable reps within this set. For example, the review can recognize a repeated bottom torso lean even when every rep differs from a chosen depth target. The display uses a spread of up to 5° for bottom lean, 7° for peak flexion and 0.15 s for an individual phase time when highlighting similar measurements. These thresholds select observations for review. The lifter's targets express their intended technique.

There are no preset technique goals. A lifter can choose a bottom band, descent duration, ascent duration, or a combination after opening the results. Depth matches the selected band exactly. Tempo matches within the displayed tolerance, initially ±0.20 s and editable. Each selected target shows **met**, **different** or **unknown** for each attempt, with the measured value and its evidence. Changing a target reuses the saved measurements. The browser remembers targets for that job and can download the current target review as JSON.

## Visibility and scope

Counting requires a visible standing-to-standing cycle. A shallow completed rep can meet an above-knee target. A clipped cycle or a substantial tracking gap remains available to inspect. Target and consistency comparisons use completed reps with the required measurements. Missing phases have unavailable controls.

When a landmark is obscured, its observation identifies the affected measurement and offers a recording adjustment. A lost knee around the turn, for example, makes depth and phase timing uncertain. The timeline can open the tracking gap. Substantial camera-relative ankle movement or a view with both shoulders and hips widely separated flags spatial comparisons as uncertain. Film level beside the lifter, keep the near shoulder, hip, knee and ankle visible, and include standing pauses in the trim.

Bottom measurements tolerate ordinary keypoint jitter. When the lifter pauses at the bottom for 0.3 s or longer, depth and torso lean are the median over the pause. A touch-and-go bottom is read from a short curve fitted through the turn, not from one frame. On synthetic reps with 2–3 px of jitter this keeps the depth band right on about 99% of reps (90% from a single frame) and the median torso-lean error near 0.5°. Torso lean is measured against image vertical, so a tilted camera adds its tilt to every rep's lean. Comparisons between reps in one clip are unaffected.

The review describes projected movement in the recorded view. Clinical assessments, injury-risk ratings, rehab clearance and competition judging are outside its scope.

## Acceptance check

Run the synthetic squat sample in fixture mode. **What did I do well?** should identify similar bottom torso lean on reps 1–3. **What should I look at more closely?** should compare the shallower third rep with the first. Selecting that comparison should pause the two bottom frames at 7.40 s and 1.80 s in the original clip. Switching off the overlay should reveal the same positions. Setting an above-knee target should mark rep 3 as met and the obscured fourth attempt as unknown.

The automated browser test repeats this flow after trimming 0.5 s from the start. It checks both video clocks, the displayed pixels with the overlay on and off, phase jumps, target differences, reload and the gap link. Run it with Playwright and Chromium installed.

```bash
python -m pip install playwright
python -m playwright install chromium
python -m pytest -q tests/test_squat_review_browser.py
```

Feedback validation currently uses synthetic side-view footage. Real sets with manually marked landmarks are needed to validate observation selection, frame alignment, estimated angles and target comparisons across lifters, camera positions and barbell occlusion.

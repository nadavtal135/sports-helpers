# basketball

Makes, misses and unknowns from a still camera, with one shooter and one ball. Adapted from
[vision-demos/basketball](https://github.com/jeremyipark/vision-demos/tree/c06cfb78ccd8e2c63629222695aac710fe4c199c/basketball)
(Apache-2.0); see `../docs/UPSTREAM.md` for the changes.

## Method

1. **Pose** (ViTPose, every frame). With several people, the shooter is the one with both wrists overhead most often, chosen per camera segment and followed through tracker id changes (re-linked by box overlap). A candidate shot is both wrists above the head, re-armed only after both drop below the shoulders, with candidates at least 0.75 s apart (so back-to-back shots ~2 s apart stay separate).
2. **Hoop** (SAM 3.1, prompt "basketball hoop"). The rim is the mask's widest rows; the net bottom is the mask's lowest row.
   - **Camera cuts:** frame-difference spikes split the clip into camera segments, and SAM runs once per segment (and every 20 s within a long one). Each shot is judged against its own segment's hoop, and the ball evidence stops at the segment's end.
   - **Several hoops in view:** the largest rim is used, not the highest-scoring detection, since a far-court hoop is small. In "auto" mode, if the ball is seen dropping through another found hoop, that one is used instead. The "Hoop" setting can force largest, highest score, leftmost or rightmost. The run warns whenever more than one hoop was found.
3. **Ball** (SAM 3.1 tracking, prompt "basketball"), only from 0.6 s before to 2.4 s after each shot. The window is extended by 1.5 s (at most twice) while the ball is still at the rim. A quick second shot keeps its own 0.6 s lead-in even if that overlaps the previous window, and overlapping detections are merged. Boxes that are not ball-sized or round are dropped, except that in a frame with no accepted ball, a box stretched by motion blur is accepted when its short side is ball-sized, it is stretched along the ball's motion, and it lies on the path predicted from the sightings just before.
4. **Release**: the first frame the rising ball is 1.5 diameters above the shooting wrist, else the wrist's peak (recorded as `release_method = pose`).
5. **Verdict**. A **make** comes down from above the rim, crosses the rim line in its middle 80%, is ball-sized for the rim's distance (not in front of it), is *seen* leaving under the net within the rim's width, and never rises back over the rim. Net timing scales with the clip: gravity is converted to pixels per frame from the rim's real width and the frame rate, so a fast clean swish (about 3 frames rim to net at 30 fps) is still a make. Only a ball moving through the net faster than free fall from its own entry speed, which can't be inside the net, is called "fell past the net". A ball leaving at the net's edge needs visible net drag, or it stays unknown.
   A **miss** needs positive evidence: outside the rim, rimmed out or off the rim, in front, fell past the net, wide of the net, or never above the rim with at least 0.8 s of tracking. Everything else is **unknown**: lost after the crossing, a gap longer than 0.3 s before the net exit, the crossing hidden at the rim, or the ball lost above the rim.
   **Possible non-shots** (opt-in setting, off by default): when the release time came from the pose alone and the ball is seen moving away from the hoop in its first moves without getting above the rim, the shot is kept but shown as UNKNOWN, "possible non-shot", unless it was judged a make. It is removed (listed as a note) only if a stricter test also holds: twice the recession, and no later sighting back toward the hoop. A real shot is never removed silently.
6. **Elbow**: the projected 2D angle at the set point and at release, side-on shots only (shoulder width under 0.55 of torso length).
7. **Court spot**: a pinhole camera model. The rim's pixel width gives its distance, and the shooter's shoulder-to-ankle height gives theirs. It is an estimate scaled by the height and FOV settings.

FG% = makes ÷ (makes + misses). Unknown shots are excluded and listed separately. Thresholds are in `settings.py`.

## Checking a call yourself

- **Mark the true outcome.** Each shot in the results has a "Your call" menu: Make, Miss or Not a shot. Your mark is saved with the job and used for the results, FG% and `shots_reviewed.csv`, which lists the automatic call next to yours. "Not a shot" leaves attempts and FG%. Choosing "Auto" clears the mark.
- **See what the ball detector saw.** Every job exports `ball_detections.csv`: every box SAM returned, at any score, with its score band (`main` ≥ 0.5, `rescue` 0.3–0.5, `below`) and whether the track kept it. To draw them on the original clip:

  ```bash
  python -m basketball.diagnose path/to/job/output path/to/clip.mp4 --start 20.0 --end 23.0 --scale 2 --out sheet.png --frames-dir frames/
  ```

  This reads only the job's exported files, so it costs nothing. Green boxes could be picked, orange ones only by the blur rescue, and red ones are never used. A cyan ring marks the tracked ball, an orange line the rim, and a grey line the net bottom. "No ball" or "not tracked" in a frame's corner tells you whether SAM looked at that frame.
- A ball that leaves the top of the frame on a high arc is judged when it comes back down through the rim. If it's never seen again, the shot is unknown, never a miss.

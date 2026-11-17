# basketball

Makes, misses and unknowns from a still camera, with one shooter and one ball. Adapted from
[vision-demos/basketball](https://github.com/jeremyipark/vision-demos/tree/c06cfb78ccd8e2c63629222695aac710fe4c199c/basketball)
(Apache-2.0); see `../docs/UPSTREAM.md` for the changes.

## Method

1. **Pose** (ViTPose, every frame). A candidate shot is both wrists above the head, re-armed only after both drop below the shoulders, with candidates at least 1.5 s apart.
2. **Hoop** (SAM 3.1, one frame, prompt "basketball hoop"). The rim is the mask's widest rows; the net bottom is the mask's lowest row.
3. **Ball** (SAM 3.1 tracking, prompt "basketball"), only from 0.6 s before to 2.4 s after each shot. The window is extended by 1.5 s (at most twice) while the ball is still at the rim. Boxes that are not ball-sized or round are dropped.
4. **Release**: the first frame the rising ball is 1.5 diameters above the shooting wrist, else the wrist's peak (recorded as `release_method = pose`).
5. **Verdict**. A **make** comes down from above the rim, crosses the rim line in its middle 80%, is ball-sized for the rim's distance (not in front of it), is *seen* leaving the net no faster than 0.15 s rim-to-net, and never rises back over the rim. A **miss** needs positive evidence: outside the rim, rimmed out or off the rim, in front, fell past the net, wide of the net, or never above the rim with at least 0.8 s of tracking. Everything else is **unknown**: lost after the crossing, a gap longer than 0.3 s before the net exit, the crossing hidden at the rim, or the ball lost above the rim.
6. **Elbow**: the projected 2D angle at the set point and at release, side-on shots only (shoulder width under 0.55 of torso length).
7. **Court spot**: a pinhole camera model. The rim's pixel width gives its distance, and the shooter's shoulder-to-ankle height gives theirs. It is an estimate scaled by the height and FOV settings.

FG% = makes ÷ (makes + misses). Unknown shots are excluded and listed separately. Thresholds are in `settings.py`.

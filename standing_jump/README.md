# Standing jump technique review

Use a fixed, level camera directly beside one person doing two-foot standing vertical jumps. A countermovement and natural arm swing are supported. Leave room for the entire body at the apex, both shoes and a strip of floor beneath them. Keep the shoes visible through contact. Start with a short standing pause and film until the jumper has finished recovering.

## Open a review

Run the web app from the repository root with `python run.py`. Choose **Standing Jump Analysis**, upload a MOV or MP4, keep the full movement inside the trim and select **Analyze**. Personal footage uses the server’s `VLMRUN_API_KEY` and the usual live-analysis consent. **Use the synthetic sample clip** runs the same detector, video review and exports without a key. Fixture mode accepts only the registered sample for analysis.

Results open with up to two observations. Select one to pause at the cited frame and see its measurement. A comparison opens two labeled frames side by side. **Inspect a frame** selects other cited attempts or phases. **Previous frame** and **Next frame** step the main view. **Show pose overlay** switches both views while preserving their positions. Each timeline entry can replay the attempt or recovery, open a phase, or show where visibility was lost.

For example, the sample repeats the descent timing across four attempts. On attempt 3 the visible knee added **17.6°** after contact, compared with **59.8°** and **57.2°** on attempts 1 and 2. The comparison opens on frames **238** and **67**. Select attempt 3’s contact frame **231** to inspect how the movement started. Attempt 2 contains a recovery step. On attempt 4 both shoes disappear around landing, so takeoff and the later standing stance can be inspected and first contact is unavailable.

## How to read the phases

| Phase or measurement | What the cited frame shows |
|---|---|
| Countermovement | The lowest visible hip position in the dip before push-off. Descent time runs from departure from a visible standing pause to this position. Consecutive jumps can share a landing and dip without a separate measurable standing start. |
| Takeoff | The estimated first frame both feet are airborne. The detector checks rising hips, sustained ankle clearance and evidence of a grounded stance. A grounded dip or arm swing alone does not count. |
| Flight | A frame near the greatest visible two-foot clearance. Estimated flight time is the interval between estimated takeoff and first contact when tracking supports both endpoints and the intervening frames. |
| First landing contact | The estimated first returning shoe to meet the floor. COCO-17 tracks ankles. Where possible the detector refines the ankle estimate by comparing shoe pixels with the exposed floor during flight. The frame caption identifies the evidence and flags an unclear shoe edge. |
| Deepest landing bend | The greatest observed projected knee flexion in the initial landing bend. With a visibly oblique view the landmark uses the lowest visible hip and the angle measurements are withheld. |
| Added landing knee bend | The change in the measured side’s projected hip–knee–ankle flexion from first contact to deepest landing. A straight projected leg measures 0°. Both frames are available for inspection. |
| Torso lean | The angle between the visible hip-to-shoulder line and image vertical at the lowest dip or landing. Upright in the image is 0°. |
| Recovery | A near-standing stance with still feet and hip. A visible step requires one foot to lift, travel along the floor and replant while the other remains supported. The replay starts at contact. A landing that flows straight into the next dip can be reviewed without a separate settled frame. |

The analysis selects one well-observed side for the clip. Descriptions and comparisons apply to the visible movement. A hidden leg cannot establish bilateral symmetry. Low confidence at one phase leaves independently observed phases available. An oblique view limits angle comparisons. Dark shoes against a dark floor, occlusion and motion blur can make first contact hard to locate. A level side view, contrasting shoes and good lighting make the supporting frames easier to check.

### Imperfect tracking

Takeoff and first contact need both ankles. In an exact side view the near leg can cover the far shoe for most of the clip. When the far ankle is hidden on at least half of the frames where the near one is visible, the results explain that two-foot takeoff and contact cannot be confirmed. Lowering the camera toward shoe height or turning it about 15° toward the jumper's front usually keeps both shoes visible.

A pose dropout of one or two frames (up to 0.07 s) is bridged by interpolating each joint, so flight time survives it. Contact or takeoff inside a bridged gap is labelled “across a one- or two-frame pose dropout (±1 frame)”. Longer gaps stay gaps. When the track shows keypoint jitter, the knee and torso angles at each landmark come from a short local fit (one-sided at takeoff and contact), and the settled-stance check uses lightly smoothed ankles and hip. A clean track is read frame by frame as before.

**My jump technique target** accepts a personal minimum and maximum for descent time, added landing knee bend or landing torso lean. Each attempt shows within, above, below or unknown, with its evidence when available. Default consistency observations describe repetition. There is no prescribed target, form score or jump-height output.

## Files and frame numbering

The download bundle contains a frame-aligned clean video, an annotated video, `jumps.csv` and `jumps.json`. Frames are zero-based on the constant-rate analysis grid. Timestamps are seconds from the beginning of the original upload, including after a trim. Use timestamps to locate events in a variable-frame-rate phone recording. JSON includes each attempt’s phase landmarks, available measurements, visibility issues, contact timing evidence and the default technique observations. CSV leaves missing measurements blank.

## Real-footage check

Two public 60 fps recordings show stationary-camera repeated jumps from somewhat oblique side views. The camera model is unknown. Each takeoff and landing was labeled by stepping through the unannotated shoe frames before running pose inference on that clip. Labels use the first frame both shoes clear the floor and the first frame either shoe touches down. Toe visibility and shadows leave roughly one frame of manual uncertainty. Both runs used a local COCO-17 pose model and the app’s ordinary conversion, detector, feedback and export path. [First recording](https://media.musclewiki.com/media/uploads/videos/branded/female-Bodyweight-jump-squats-side.mp4) · [Second recording](https://media.musclewiki.com/media/uploads/videos/branded/male-Bodyweight-jump-squats-side.mp4) · [Evaluation pose model](https://huggingface.co/Xenova/yolov8n-pose)

| Recording | Attempt | Manual takeoff / contact | App takeoff / contact | Timing evidence |
|---|---:|---:|---:|---|
| Development | 1 | 95 / 110 | 95 / 110 | Shoe edge |
| Development | 2 | 170 / 188 | 170 / 188 | Shoe edge |
| Development | 3 | 246 / 265 | 246 / 264 | Shoe edge |
| Separate check | 1 | 66 / 84 | 64 / 86 | Ankle motion |
| Separate check | 2 | 136 / 153 | 132 / 154 | Ankle motion |
| Separate check | 3 | 206 / 223 | 203 / 224 | Ankle motion |
| Separate check | 4 | 272 / 294 | 269 / 296 | Ankle motion |

The first clip informed the shoe-edge check. The detector was then run without retuning on the second clip, where dark shoes and floor made the sole boundary unclear. Its takeoff estimates were 2–4 frames early and contact estimates 1–2 frames late. At 60 fps the largest difference was about 67 ms. In both recordings the cited first landing showed the visible knee continuing to bend after contact, successive landings flowed directly into the next dip, and the jumper settled after the final attempt. The oblique-view warning withheld numerical angle rankings.

The synthetic tests also exercise a grounded dip, a single recovery step, missing contact, a clip cut off in flight, personal targets, source-time alignment and the browser interactions. The live VLM Run client has an offline request/response and cache test. Directly side-on iPhone recordings through the live provider are still needed to validate angle magnitudes and recovery-step feedback across people, lighting and camera positions.

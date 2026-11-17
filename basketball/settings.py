"""Explicit per-run settings for the basketball pipeline.

The reference (jeremyipark/vision-demos ``basketball/config.py``, Apache-2.0,
commit c06cfb78ccd8e2c63629222695aac710fe4c199c) keeps these as module
constants mutated per run. Here they are a frozen dataclass passed into the
pipeline, so nothing global changes between runs. Defaults are the reference's
unless a comment says otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Bump when the analysis can produce different output for the same inputs.
# Part of the analysis cache key and recorded in every result.
PIPELINE_VERSION = "basketball-1.0.0"


@dataclass(frozen=True)
class Gateway:
    base_url: str = "https://gateway.vlm.run/v1/openai"
    request_timeout: float = 1800.0
    pose_model: str = "usyd-community/vitpose-plus-large"
    sam_model: str = "facebook/sam3.1"
    precision: int = 4


@dataclass(frozen=True)
class Settings:
    # Video
    inference_height: int = 720     # reference: 1080; coordinates are normalized either way
    export_height: int = 720
    convert_crf: int = 23
    output_crf: int = 20
    # Pose
    pose_fps: float = 15.0          # detector cadence; pose still runs on every frame
    max_persons: int = 1
    person_iou: float = 0.6
    # Hoop
    hoop_prompt: str = "basketball hoop"
    hoop_frame_seconds: float | None = None   # None = the middle of the clip
    rim_diameter_m: float = 0.49
    # Ball
    ball_prompt: str = "basketball"
    ball_before_seconds: float = 0.6
    ball_after_seconds: float = 2.4
    ball_stride: int = 1
    ball_min_score: float = 0.5
    ball_size_tolerance: float = 0.35
    ball_draw_size_tolerance: float = 0.6
    ball_extend_seconds: float = 1.5
    ball_max_extensions: int = 2
    # Shots
    shot_armed_below: str = "shoulder"
    shot_min_gap_seconds: float = 1.5
    shooting_hand: str = "auto"
    held_reach: float = 1.5
    # Make / miss
    rim_inner_fraction: float = 0.80
    net_min_seconds: float = 0.15
    net_side_tolerance: float = 0.35
    rebound_margin_rims: float = 0.15
    rim_gap_seconds: float = 0.5
    # New: the longest gap in ball observations a verdict may bridge after the
    # rim crossing (through the net). Longer, or no observation at all = unknown.
    evidence_gap_seconds: float = 0.3
    # New: a "never above the rim" miss needs the ball tracked this long after release.
    min_flight_seconds: float = 0.8
    ball_diameter_m: float = 0.239
    front_of_rim_size: float = 1.2
    # Elbow (projected 2D angle, side-on shots only)
    side_on_max_shoulder_ratio: float = 0.55
    elbow_min_projected: float = 0.5
    set_window_seconds: float = 0.3
    # Court (an estimate from one camera)
    camera_hfov_degrees: float = 70.0
    player_height_m: float = 1.75
    court_yaw_degrees: float = 0.0
    three_point_radius_m: float = 6.75
    free_throw_distance_m: float = 4.19
    # Video overlay
    verdict_seconds: float = 1.2
    ball_trail_seconds: float = 0.5
    ball_fill_seconds: float = 0.33
    hot_streak: int = 3

    def as_dict(self) -> dict:
        return asdict(self)


# What the browser may change, with bounds. Everything else keeps its default.
USER_SETTINGS = {
    "shooting_hand": {"type": "choice", "choices": ["auto", "left", "right"],
                      "label": "Shooting hand"},
    "player_height_m": {"type": "float", "min": 1.2, "max": 2.3, "default": 1.75,
                        "label": "Shooter height (m)", "help": "Scales estimated court distances."},
    "camera_hfov_degrees": {"type": "float", "min": 40, "max": 120, "default": 70,
                            "label": "Camera horizontal field of view (°)",
                            "help": "About 70° for a phone's main lens in landscape."},
    "court_yaw_degrees": {"type": "float", "min": -45, "max": 45, "default": 0,
                          "label": "Camera yaw (°)",
                          "help": "Turns the court if the camera is not square to the backboard."},
    "three_point_radius_m": {"type": "choice", "choices": [6.75, 7.24, 6.02],
                             "labels": ["FIBA (6.75 m)", "NBA (7.24 m)", "US high school (6.02 m)"],
                             "label": "Three-point line"},
}

GUIDANCE = [
    "One shooter and one ball; keep other players out of frame.",
    "Phone on a tripod or set down: the camera must not move.",
    "Landscape, with the whole hoop (rim and net) and the shooter's full body in shot.",
    "Good light; avoid shooting straight into the sun or a bright window.",
    "Side-on views give the most useful (projected) elbow angles.",
]


def from_user(overrides: dict | None) -> Settings:
    clean = {}
    for key, value in (overrides or {}).items():
        spec = USER_SETTINGS.get(key)
        if spec is None:
            raise ValueError(f"unknown basketball setting {key!r}")
        if spec["type"] == "choice":
            match = [c for c in spec["choices"] if str(c) == str(value)]
            if not match:
                raise ValueError(f"{spec['label']} must be one of {spec['choices']}")
            value = match[0]
        else:
            value = float(value)
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{spec['label']} must be between {spec['min']} and {spec['max']}")
        clean[key] = value
    return Settings(**clean)

"""Capture requirements and pose settings for side-view standing jumps."""

from dataclasses import dataclass

from running.settings import Gateway, PoseVideoSettings

PIPELINE_VERSION = "standing-jump-1.0.0"
USER_SETTINGS = {}
GUIDANCE = [
    "Film one person performing two-foot standing vertical jumps with a natural arm swing.",
    "Set the iPhone still and level, directly beside the jumper.",
    "Leave room above the head and below both shoes. Keep the floor and both feet visible through landing.",
    "Include a brief standing pause before the dip and keep filming until recovery after each landing.",
]


@dataclass(frozen=True)
class Settings(PoseVideoSettings):
    joint_score: float = 0.55
    ground_band: float = 0.035      # fractions of projected leg length
    flight_clearance: float = 0.09
    recovery_window_s: float = 2.2
    # New: pose dropouts this short are bridged (a frame or two at 30 fps); longer stay gaps.
    max_bridge_s: float = 0.07
    # New: above this keypoint jitter (fraction of leg length, per frame) the stillness
    # checks and the angle readings use lightly smoothed signals. Clean tracks are unaffected.
    jitter_smoothing: float = 0.004
    # New: warn when the far ankle is hidden on more than this share of standing frames.
    far_ankle_hidden_warning: float = 0.5


def from_user(overrides: dict | None) -> Settings:
    if overrides:
        raise ValueError(f"Unknown standing jump setting {next(iter(overrides))!r}.")
    return Settings()

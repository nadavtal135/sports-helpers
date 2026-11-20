"""Squat capture guidance and pose/rep thresholds."""

from dataclasses import dataclass

from running.settings import Gateway, PoseVideoSettings

PIPELINE_VERSION = "squat-1.1.0"


@dataclass(frozen=True)
class Settings(PoseVideoSettings):
    min_joint_score: float = 0.35
    metric_joint_score: float = 0.55
    max_interp_gap_s: float = 0.10
    smoothing_s: float = 0.09
    standing_hold_s: float = 0.10
    standing_flexion_limit_deg: float = 30.0
    standing_flexion_band_deg: float = 12.0
    standing_height_band: float = 0.025   # fractions of the projected leg length
    min_hip_drop: float = 0.045
    min_flexion_change_deg: float = 20.0
    level_tolerance: float = 0.04


USER_SETTINGS = {}
GUIDANCE = [
    "Film one person doing bodyweight or barbell squats.",
    "Keep the camera still, level and side-on, about hip height.",
    "Keep the full body in frame with the near shoulder, hip, knee and ankle visible.",
    "Include a brief standing pause before the first rep and after the last, including when trimming.",
]


def from_user(overrides: dict | None) -> Settings:
    if overrides:
        raise ValueError(f"Unknown squat setting {next(iter(overrides))!r}.")
    return Settings()

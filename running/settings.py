"""Explicit per-run settings for the running pipeline.

The reference (jeremyipark/vision-demos ``running/config.py``, Apache-2.0,
commit c06cfb78ccd8e2c63629222695aac710fe4c199c) keeps these as module
constants mutated per run. Here they are a frozen dataclass passed into the
pipeline. Defaults are the reference's unless a comment says otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

PIPELINE_VERSION = "running-1.0.0"


@dataclass(frozen=True)
class Gateway:
    base_url: str = "https://gateway.vlm.run/v1/openai"
    request_timeout: float = 1800.0
    pose_model: str = "usyd-community/vitpose-plus-large"
    precision: int = 4


@dataclass(frozen=True)
class PoseVideoSettings:
    inference_height: int = 720     # reference: 1080
    export_height: int = 720
    convert_crf: int = 23
    output_crf: int = 20
    max_persons: int = 1
    person_iou: float = 0.6

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Settings(PoseVideoSettings):
    foot_reference: str = "ground"  # "ground" (still camera) or "hip" (moving camera)
    ground_percentile: float = 90
    per_foot_ground: bool = True
    smooth_median_frames: int = 3
    smooth_mean_frames: int = 3
    swing_peak_fraction: float = 0.55
    contact_fraction: float = 0.18
    min_contact_seconds: float = 0.10
    cadence_smoothing: float = 0.20
    knee_search_seconds: float = 0.10
    knee_stance_limit_deg: float = 85.0
    # New: observation gaps longer than this stay gaps instead of being
    # interpolated into the signal (the reference interpolated any gap).
    max_interp_gap_seconds: float = 0.10
    # New: fewer same-foot strides than this flags the cadence as low-confidence.
    min_strides_confident: int = 4
    # New: strides/step intervals beyond this ratio of the clip's median (either
    # way) span a pause or a missed strike, and are not measured.
    stride_outlier_ratio: float = 1.4
    # New: a foot observed on less than this share of frames is reported as absent.
    min_limb_coverage: float = 0.5
    flash_seconds: float = 0.25


USER_SETTINGS = {
    "foot_reference": {"type": "choice", "choices": ["ground", "hip"],
                       "labels": ["Camera still (ground reference)", "Camera moves (hip reference)"],
                       "label": "Camera"},
}

GUIDANCE = [
    "One runner in frame, filmed from the side.",
    "Both ankles visible the whole time; treadmill or a short straight stretch.",
    "Camera still (tripod) and level, about hip height.",
    "At least 5–10 seconds of steady running gives enough strides for cadence.",
    "Cadence and step timing are the reliable outputs; knee angles are projected 2D estimates.",
]


def from_user(overrides: dict | None) -> Settings:
    clean = {}
    for key, value in (overrides or {}).items():
        spec = USER_SETTINGS.get(key)
        if spec is None:
            raise ValueError(f"unknown running setting {key!r}")
        match = [c for c in spec["choices"] if str(c) == str(value)]
        if not match:
            raise ValueError(f"{spec['label']} must be one of {spec['choices']}")
        clean[key] = match[0]
    return Settings(**clean)

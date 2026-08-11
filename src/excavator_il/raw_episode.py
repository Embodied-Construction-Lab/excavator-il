from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


STEP_FIELDS = (
    "episode_id",
    "frame_index",
    "state_seq",
    "state_stamp_ms",
    "state_receive_monotonic_ns",
    "action_stamp_monotonic_ns",
    "boom_pos_m",
    "stick_pos_m",
    "bucket_pos_m",
    "boom_vel_mps",
    "stick_vel_mps",
    "bucket_vel_mps",
    "boom_angle_rad",
    "arm_angle_rad",
    "bucket_angle_rad",
    "swing_angle_rad",
    "swing_vel_radps",
    "action_boom",
    "action_stick",
    "action_bucket",
    "action_swing",
    "pump_percent",
    "sensor_valid",
    "control_mode",
)

CAMERA_FIELDS = (
    "camera_frame_index",
    "camera_stamp_monotonic_ns",
    "image_path",
)

ACTION_FIELDS = (
    "action_boom",
    "action_stick",
    "action_bucket",
    "action_swing",
)


class EpisodeValidationError(ValueError):
    """Raised when a raw demonstration episode violates its data contract."""


@dataclass(frozen=True)
class EpisodeValidationReport:
    episode_id: str
    step_count: int
    camera_frame_count: int
    image_shape: tuple[int, int, int]
    max_camera_age_ms: float
    max_action_age_ms: float


def _read_csv(path: Path, required_fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise EpisodeValidationError(f"missing required file: {path.name}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in required_fields if field not in (reader.fieldnames or ())]
        if missing:
            raise EpisodeValidationError(f"{path.name} missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise EpisodeValidationError(f"{path.name} contains no data rows")
    return rows


def _load_metadata(path: Path) -> dict:
    metadata_path = path / "episode.json"
    if not metadata_path.is_file():
        raise EpisodeValidationError("missing required file: episode.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeValidationError(f"invalid episode.json: {exc}") from exc
    episode_id = metadata.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise EpisodeValidationError("episode.json episode_id must be non-empty text")
    if metadata.get("schema_version") != "excavator_demo_raw.v1":
        raise EpisodeValidationError("episode.json schema_version must be excavator_demo_raw.v1")
    if metadata.get("task") != "ExecuteDig":
        raise EpisodeValidationError("episode.json task must be ExecuteDig")
    status = metadata.get("status")
    if status == "pending_review":
        raise EpisodeValidationError(
            "episode.json status pending_review must be classified before validation"
        )
    if status not in {"complete", "failed", "aborted"}:
        raise EpisodeValidationError(
            "episode.json status must be complete, failed, or aborted"
        )
    camera = metadata.get("camera_front")
    if not isinstance(camera, dict):
        raise EpisodeValidationError("episode.json camera_front must be an object")
    required_camera_fields = (
        "device_id",
        "width",
        "height",
        "nominal_fps",
        "pixel_format",
        "timestamp_clock",
    )
    missing = [field for field in required_camera_fields if field not in camera]
    if missing:
        raise EpisodeValidationError(
            f"episode.json camera_front missing fields: {', '.join(missing)}"
        )
    if camera["timestamp_clock"] != "CLOCK_MONOTONIC":
        raise EpisodeValidationError("camera_front timestamp_clock must be CLOCK_MONOTONIC")
    return metadata


def _validate_quality_health(path: Path) -> None:
    report_path = path / "quality_report.json"
    if not report_path.is_file():
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeValidationError(f"invalid quality_report.json: {exc}") from exc
    if not isinstance(report, dict):
        raise EpisodeValidationError("quality_report.json must be an object")
    timeout_count = report.get("joystick_timeout_count", 0)
    if (
        isinstance(timeout_count, bool)
        or not isinstance(timeout_count, int)
        or timeout_count < 0
    ):
        raise EpisodeValidationError(
            "quality_report joystick_timeout_count must be a non-negative integer"
        )
    if timeout_count:
        raise EpisodeValidationError(
            f"quality report contains {timeout_count} joystick timeout event(s)"
        )


def _as_int(value: str, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise EpisodeValidationError(f"{field} must be an integer, got {value!r}") from exc


def _as_float(value: str, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EpisodeValidationError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise EpisodeValidationError(f"{field} must be finite, got {value!r}")
    return number


def validate_episode(
    path: str | Path,
    *,
    max_camera_age_ms: float = 120.0,
    max_action_age_ms: float = 100.0,
) -> EpisodeValidationReport:
    """Validate one timestamped RGB demonstration episode."""
    episode_path = Path(path)
    metadata = _load_metadata(episode_path)
    _validate_quality_health(episode_path)
    episode_id = metadata["episode_id"]
    steps = _read_csv(episode_path / "steps.csv", STEP_FIELDS)
    camera_rows = _read_csv(episode_path / "camera_front_timestamps.csv", CAMERA_FIELDS)

    frame_indices = [_as_int(row["frame_index"], "frame_index") for row in steps]
    if frame_indices != list(range(len(steps))):
        raise EpisodeValidationError("steps.csv frame_index must be contiguous from zero")
    if any(row["episode_id"] != episode_id for row in steps):
        raise EpisodeValidationError("steps.csv episode_id does not match episode.json")
    for row_index, row in enumerate(steps):
        for field in ACTION_FIELDS:
            action = _as_float(row[field], field)
            if not -1.0 <= action <= 1.0:
                raise EpisodeValidationError(
                    f"{field} at row {row_index} must be in [-1, 1], got {action}"
                )
        if row["sensor_valid"] != "1":
            raise EpisodeValidationError(
                f"sensor_valid at row {row_index} must be 1 for training episodes"
            )
        if row["control_mode"] != "manual_joystick":
            raise EpisodeValidationError(
                f"control_mode at row {row_index} must be manual_joystick"
            )

    state_timestamps = [_as_int(row["state_receive_monotonic_ns"], "state_receive_monotonic_ns") for row in steps]
    if any(current <= previous for previous, current in zip(state_timestamps, state_timestamps[1:])):
        raise EpisodeValidationError("state_receive_monotonic_ns must strictly increase")

    action_timestamps = [
        _as_int(row["action_stamp_monotonic_ns"], "action_stamp_monotonic_ns")
        for row in steps
    ]
    action_ages_ms: list[float] = []
    for state_timestamp, action_timestamp in zip(state_timestamps, action_timestamps):
        if action_timestamp > state_timestamp:
            raise EpisodeValidationError("future expert action is not causal")
        action_ages_ms.append((state_timestamp - action_timestamp) / 1_000_000.0)
    oldest_action_age_ms = max(action_ages_ms)
    if oldest_action_age_ms > max_action_age_ms:
        raise EpisodeValidationError(
            f"expert action age {oldest_action_age_ms:.3f} ms exceeds "
            f"{max_action_age_ms:.3f} ms"
        )

    camera_timestamps = [
        _as_int(row["camera_stamp_monotonic_ns"], "camera_stamp_monotonic_ns") for row in camera_rows
    ]
    if any(current <= previous for previous, current in zip(camera_timestamps, camera_timestamps[1:])):
        raise EpisodeValidationError("camera_stamp_monotonic_ns must strictly increase")

    image_shape: tuple[int, int, int] | None = None
    for row in camera_rows:
        image_path = episode_path / row["image_path"]
        if not image_path.is_file():
            raise EpisodeValidationError(f"camera image does not exist: {row['image_path']}")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            current_shape = (rgb.height, rgb.width, 3)
        if image_shape is None:
            image_shape = current_shape
        elif current_shape != image_shape:
            raise EpisodeValidationError("all camera images must have the same shape")

    camera_index = 0
    camera_ages_ms: list[float] = []
    for state_timestamp in state_timestamps:
        while camera_index + 1 < len(camera_timestamps) and camera_timestamps[camera_index + 1] <= state_timestamp:
            camera_index += 1
        camera_timestamp = camera_timestamps[camera_index]
        if camera_timestamp > state_timestamp:
            raise EpisodeValidationError("no causal camera frame exists for the first state")
        camera_ages_ms.append((state_timestamp - camera_timestamp) / 1_000_000.0)

    oldest_camera_age_ms = max(camera_ages_ms)
    if oldest_camera_age_ms > max_camera_age_ms:
        raise EpisodeValidationError(
            f"camera frame age {oldest_camera_age_ms:.3f} ms exceeds {max_camera_age_ms:.3f} ms"
        )

    assert image_shape is not None
    camera_metadata = metadata["camera_front"]
    expected_shape = (
        _as_int(str(camera_metadata["height"]), "camera_front.height"),
        _as_int(str(camera_metadata["width"]), "camera_front.width"),
        3,
    )
    if image_shape != expected_shape:
        raise EpisodeValidationError(
            f"camera image shape {image_shape} does not match metadata {expected_shape}"
        )
    return EpisodeValidationReport(
        episode_id=episode_id,
        step_count=len(steps),
        camera_frame_count=len(camera_rows),
        image_shape=image_shape,
        max_camera_age_ms=oldest_camera_age_ms,
        max_action_age_ms=oldest_action_age_ms,
    )

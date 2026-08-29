from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image

from .collector.config import (
    validate_collection_labels,
    validate_collection_protocol,
    validate_recording_purpose,
)
from .training_segments import TRAINING_SEGMENTS_SCHEMA_VERSION


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
EPISODE_SCHEMA_VERSION = "excavator_demo_raw.v2"
LEGACY_EPISODE_SCHEMA_VERSION = "excavator_demo_raw.v1"
_CAMERA_RATE_TOLERANCE_RATIO = 1.0 / 6.0


class EpisodeValidationError(ValueError):
    """Raised when a raw demonstration episode violates its data contract."""


@dataclass(frozen=True)
class TrainingSegment:
    segment_id: str
    start_frame_index: int
    end_frame_index_exclusive: int


@dataclass(frozen=True)
class CameraValidationReport:
    camera_id: str
    frame_count: int
    estimated_rate_hz: float
    image_shape: tuple[int, int, int]
    max_age_ms: float


@dataclass(frozen=True)
class EpisodeValidationReport:
    episode_id: str
    step_count: int
    camera_frame_count: int
    image_shape: tuple[int, int, int]
    max_camera_age_ms: float
    max_action_age_ms: float
    training_segment_count: int
    training_segments: tuple[TrainingSegment, ...]
    cameras: Mapping[str, CameraValidationReport]
    max_intercamera_skew_ms: float


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
    if metadata.get("schema_version") not in {
        EPISODE_SCHEMA_VERSION,
        LEGACY_EPISODE_SCHEMA_VERSION,
    }:
        raise EpisodeValidationError(
            "episode.json schema_version must be "
            f"{EPISODE_SCHEMA_VERSION} or {LEGACY_EPISODE_SCHEMA_VERSION}"
        )
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
    schema_version = metadata["schema_version"]
    try:
        recording_purpose = validate_recording_purpose(
            metadata.get("recording_purpose", "demonstration")
        )
    except ValueError as exc:
        raise EpisodeValidationError(str(exc)) from exc
    if schema_version == LEGACY_EPISODE_SCHEMA_VERSION:
        camera_front = metadata.get("camera_front")
        if not isinstance(camera_front, dict):
            raise EpisodeValidationError("episode.json camera_front must be an object")
        cameras = {"front": camera_front}
    else:
        cameras = metadata.get("cameras")
        if not isinstance(cameras, dict):
            raise EpisodeValidationError("episode.json cameras must be an object")
        if "front" not in cameras:
            raise EpisodeValidationError("episode.json cameras.front must be an object")
        unsupported = set(cameras) - {"front", "dump"}
        if unsupported:
            raise EpisodeValidationError(
                "episode.json cameras contains unsupported roles: "
                + ", ".join(sorted(unsupported))
            )
        protocol = metadata.get("collection_protocol")
        if (
            recording_purpose == "demonstration"
            and protocol is None
            and "dump" in cameras
        ):
            raise EpisodeValidationError(
                "dual-camera episode.json requires collection_protocol"
            )
        if protocol is not None:
            if not isinstance(protocol, dict):
                raise EpisodeValidationError(
                    "episode.json collection_protocol must be an object"
                )
            try:
                validate_collection_protocol(
                    task_variant=protocol.get("task_variant"),
                    soil_reset_block_id=protocol.get("soil_reset_block_id"),
                    dig_point_id=protocol.get("dig_point_id"),
                )
            except ValueError as exc:
                raise EpisodeValidationError(
                    f"invalid collection_protocol: {exc}"
                ) from exc
        collection_labels = metadata.get("collection_labels")
        if collection_labels is not None:
            if not isinstance(collection_labels, dict) or set(collection_labels) != {
                "collection_zone_id",
                "dig_repeat_index",
                "operator_note",
            }:
                raise EpisodeValidationError(
                    "episode.json collection_labels must contain exactly "
                    "collection_zone_id, dig_repeat_index and operator_note"
                )
            try:
                validate_collection_labels(
                    collection_zone_id=collection_labels.get("collection_zone_id"),
                    dig_repeat_index=collection_labels.get("dig_repeat_index"),
                    operator_note=collection_labels.get("operator_note"),
                )
            except ValueError as exc:
                raise EpisodeValidationError(
                    f"invalid collection_labels: {exc}"
                ) from exc
    required_camera_fields = (
        "device_id",
        "width",
        "height",
        "nominal_fps",
        "pixel_format",
        "timestamp_clock",
    )
    for camera_id, camera in cameras.items():
        if not isinstance(camera, dict):
            raise EpisodeValidationError(
                f"episode.json cameras.{camera_id} must be an object"
            )
        missing = [name for name in required_camera_fields if name not in camera]
        if missing:
            raise EpisodeValidationError(
                f"episode.json cameras.{camera_id} missing fields: "
                + ", ".join(missing)
            )
        if camera["timestamp_clock"] != "CLOCK_MONOTONIC":
            raise EpisodeValidationError(
                f"cameras.{camera_id} timestamp_clock must be CLOCK_MONOTONIC"
            )
    metadata["_cameras_by_role"] = cameras
    return metadata


def _validate_camera_stream(
    *,
    episode_path: Path,
    camera_id: str,
    camera_metadata: Mapping[str, object],
    state_timestamps: list[int],
    max_camera_age_ms: float,
) -> tuple[
    CameraValidationReport,
    tuple[int, ...],
]:
    stream_name = f"camera_{camera_id}"
    rows = _read_csv(
        episode_path / f"{stream_name}_timestamps.csv", CAMERA_FIELDS
    )
    frame_indices = [
        _as_int(row["camera_frame_index"], "camera_frame_index") for row in rows
    ]
    if frame_indices != list(range(len(rows))):
        raise EpisodeValidationError(
            f"{stream_name}_timestamps.csv frame indices must be contiguous from zero"
        )
    camera_timestamps = [
        _as_int(row["camera_stamp_monotonic_ns"], "camera_stamp_monotonic_ns")
        for row in rows
    ]
    if any(
        current <= previous
        for previous, current in zip(camera_timestamps, camera_timestamps[1:])
    ):
        raise EpisodeValidationError(
            f"{stream_name}_timestamps.csv camera_stamp_monotonic_ns must strictly increase"
        )

    image_shape: tuple[int, int, int] | None = None
    for row in rows:
        relative_path = Path(row["image_path"])
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
            or relative_path.parts[0] != stream_name
        ):
            raise EpisodeValidationError(
                f"{camera_id} camera image path must be inside {stream_name}"
            )
        image_path = episode_path / relative_path
        if not image_path.is_file():
            raise EpisodeValidationError(f"camera image does not exist: {row['image_path']}")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            current_shape = (rgb.height, rgb.width, 3)
        if image_shape is None:
            image_shape = current_shape
        elif current_shape != image_shape:
            raise EpisodeValidationError(
                f"all {camera_id} camera images must have the same shape"
            )

    camera_index = 0
    camera_ages_ms: list[float] = []
    causal_timestamps: list[int] = []
    for state_timestamp in state_timestamps:
        while (
            camera_index + 1 < len(camera_timestamps)
            and camera_timestamps[camera_index + 1] <= state_timestamp
        ):
            camera_index += 1
        camera_timestamp = camera_timestamps[camera_index]
        if camera_timestamp > state_timestamp:
            raise EpisodeValidationError(
                f"no causal {camera_id} camera frame exists for the first state"
            )
        causal_timestamps.append(camera_timestamp)
        camera_ages_ms.append((state_timestamp - camera_timestamp) / 1_000_000.0)

    oldest_camera_age_ms = max(camera_ages_ms)
    if oldest_camera_age_ms > max_camera_age_ms:
        raise EpisodeValidationError(
            f"{camera_id} camera frame age {oldest_camera_age_ms:.3f} ms exceeds "
            f"{max_camera_age_ms:.3f} ms"
        )

    assert image_shape is not None
    expected_shape = (
        _as_int(str(camera_metadata["height"]), f"cameras.{camera_id}.height"),
        _as_int(str(camera_metadata["width"]), f"cameras.{camera_id}.width"),
        3,
    )
    if image_shape != expected_shape:
        raise EpisodeValidationError(
            f"{camera_id} camera image shape {image_shape} does not match metadata "
            f"{expected_shape}"
        )
    estimated_rate_hz = 0.0
    if len(camera_timestamps) > 1:
        duration_s = (camera_timestamps[-1] - camera_timestamps[0]) / 1e9
        if duration_s > 0.0:
            estimated_rate_hz = (len(camera_timestamps) - 1) / duration_s
    nominal_rate_hz = _as_float(
        str(camera_metadata["nominal_fps"]),
        f"cameras.{camera_id}.nominal_fps",
    )
    if nominal_rate_hz <= 0.0:
        raise EpisodeValidationError(
            f"cameras.{camera_id}.nominal_fps must be positive"
        )
    lower_rate_hz = nominal_rate_hz * (1.0 - _CAMERA_RATE_TOLERANCE_RATIO)
    upper_rate_hz = nominal_rate_hz * (1.0 + _CAMERA_RATE_TOLERANCE_RATIO)
    if not lower_rate_hz <= estimated_rate_hz <= upper_rate_hz:
        raise EpisodeValidationError(
            f"{camera_id} camera rate {estimated_rate_hz:.3f} Hz is outside "
            f"[{lower_rate_hz:.3f}, {upper_rate_hz:.3f}] Hz for nominal "
            f"{nominal_rate_hz:.3f} Hz"
        )
    return (
        CameraValidationReport(
            camera_id=camera_id,
            frame_count=len(rows),
            estimated_rate_hz=estimated_rate_hz,
            image_shape=image_shape,
            max_age_ms=oldest_camera_age_ms,
        ),
        tuple(causal_timestamps),
    )


def _load_quality_report(path: Path) -> dict:
    report_path = path / "quality_report.json"
    if not report_path.is_file():
        return {}
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
    return report


def _validate_training_segments(
    path: Path,
    episode_id: str,
    steps: list[dict[str, str]],
    quality: dict,
) -> tuple[TrainingSegment, ...]:
    timeout_count = int(quality.get("joystick_timeout_count", 0))
    manifest_path = path / "training_segments.json"
    if not manifest_path.is_file():
        if timeout_count:
            raise EpisodeValidationError(
                f"quality report contains {timeout_count} joystick timeout event(s) "
                "without a recovered training segment manifest"
            )
        sequences = [_as_int(row["state_seq"], "state_seq") for row in steps]
        if any(
            right != left + 1
            for left, right in zip(sequences, sequences[1:])
        ):
            raise EpisodeValidationError(
                "legacy steps.csv has a state sequence gap; rerun build-steps "
                "to create training_segments.json"
            )
        return (TrainingSegment(f"{episode_id}_segment_0000", 0, len(steps)),)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeValidationError(f"invalid training_segments.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise EpisodeValidationError("training_segments.json must be an object")
    if manifest.get("schema_version") != TRAINING_SEGMENTS_SCHEMA_VERSION:
        raise EpisodeValidationError(
            f"training_segments.json schema_version must be "
            f"{TRAINING_SEGMENTS_SCHEMA_VERSION}"
        )
    if manifest.get("parent_episode_id") != episode_id:
        raise EpisodeValidationError(
            "training_segments.json parent_episode_id does not match episode.json"
        )
    if manifest.get("strategy") != "lerobot_episode_boundaries":
        raise EpisodeValidationError(
            "training_segments.json strategy must be lerobot_episode_boundaries"
        )
    recovery_sample_count = manifest.get("recovery_joystick_sample_count")
    if (
        isinstance(recovery_sample_count, bool)
        or not isinstance(recovery_sample_count, int)
        or recovery_sample_count <= 0
    ):
        raise EpisodeValidationError(
            "training segment recovery joystick sample count must be positive"
        )
    if manifest.get("unresolved_safety_event_count") != 0:
        raise EpisodeValidationError("training segment manifest has unresolved safety events")
    fault_events = manifest.get("fault_events")
    if not isinstance(fault_events, list) or len(fault_events) != timeout_count:
        raise EpisodeValidationError(
            "training segment fault event count does not match joystick timeout count"
        )
    quarantine_intervals: list[tuple[int, int]] = []
    for event in fault_events:
        if not isinstance(event, dict) or event.get("recovered") is not True:
            raise EpisodeValidationError("training segment safety event is not recovered")
        if event.get("event_type") != "joystick_timeout":
            raise EpisodeValidationError(
                "training segment safety event type must be joystick_timeout"
            )
        fault = event.get("event_stamp_monotonic_ns")
        recovery = event.get("recovery_stamp_monotonic_ns")
        if (
            isinstance(fault, bool)
            or not isinstance(fault, int)
            or isinstance(recovery, bool)
            or not isinstance(recovery, int)
            or recovery <= fault
        ):
            raise EpisodeValidationError("training segment safety event timestamps are invalid")
        quarantine_intervals.append((fault, recovery))

    raw_segments = manifest.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise EpisodeValidationError("training_segments.json contains no training segments")
    segments: list[TrainingSegment] = []
    segment_ids: set[str] = set()
    expected_start = 0
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            raise EpisodeValidationError("training segment must be an object")
        segment_id = raw_segment.get("segment_id")
        start = raw_segment.get("start_frame_index")
        end = raw_segment.get("end_frame_index_exclusive")
        if not isinstance(segment_id, str) or not segment_id:
            raise EpisodeValidationError("training segment id must be non-empty text")
        if segment_id in segment_ids:
            raise EpisodeValidationError("training segment ids must be unique")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != expected_start
            or end <= start
            or end > len(steps)
            or raw_segment.get("step_count") != end - start
        ):
            raise EpisodeValidationError("training segment frame range is invalid")
        segment_rows = steps[start:end]
        sequences = [_as_int(row["state_seq"], "state_seq") for row in segment_rows]
        if any(right != left + 1 for left, right in zip(sequences, sequences[1:])):
            raise EpisodeValidationError("training segment crosses a state sequence gap")
        stamps = [int(row["state_receive_monotonic_ns"]) for row in segment_rows]
        if (
            raw_segment.get("start_state_receive_monotonic_ns") != stamps[0]
            or raw_segment.get("end_state_receive_monotonic_ns") != stamps[-1]
        ):
            raise EpisodeValidationError(
                "training segment timestamp provenance does not match steps.csv"
            )
        if any(
            stamps[0] < fault <= stamps[-1]
            for fault, _recovery in quarantine_intervals
        ):
            raise EpisodeValidationError("training segment crosses a safety event")
        if any(
            fault <= stamp <= recovery
            for stamp in stamps
            for fault, recovery in quarantine_intervals
        ):
            raise EpisodeValidationError("training segment contains a safety quarantine row")
        segments.append(TrainingSegment(segment_id, start, end))
        segment_ids.add(segment_id)
        expected_start = end
    if expected_start != len(steps):
        raise EpisodeValidationError("training segments do not cover all training rows")
    declared_count = quality.get("training_segment_count", len(segments))
    if declared_count != len(segments):
        raise EpisodeValidationError(
            "quality report training segment count does not match manifest"
        )
    return tuple(segments)


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
    max_intercamera_skew_ms: float = 50.0,
) -> EpisodeValidationReport:
    """Validate one timestamped RGB demonstration episode."""
    episode_path = Path(path)
    metadata = _load_metadata(episode_path)
    quality = _load_quality_report(episode_path)
    episode_id = metadata["episode_id"]
    steps = _read_csv(episode_path / "steps.csv", STEP_FIELDS)
    training_segments = _validate_training_segments(
        episode_path, episode_id, steps, quality
    )
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

    camera_reports: dict[str, CameraValidationReport] = {}
    causal_camera_timestamps: dict[str, tuple[int, ...]] = {}
    for camera_id, camera_metadata in metadata["_cameras_by_role"].items():
        report, causal_timestamps = _validate_camera_stream(
            episode_path=episode_path,
            camera_id=camera_id,
            camera_metadata=camera_metadata,
            state_timestamps=state_timestamps,
            max_camera_age_ms=max_camera_age_ms,
        )
        camera_reports[camera_id] = report
        causal_camera_timestamps[camera_id] = causal_timestamps

    max_intercamera_skew = 0.0
    if len(causal_camera_timestamps) > 1:
        for timestamps in zip(*causal_camera_timestamps.values(), strict=True):
            max_intercamera_skew = max(
                max_intercamera_skew,
                (max(timestamps) - min(timestamps)) / 1_000_000.0,
            )
        if max_intercamera_skew > max_intercamera_skew_ms:
            raise EpisodeValidationError(
                f"inter-camera skew {max_intercamera_skew:.3f} ms exceeds "
                f"{max_intercamera_skew_ms:.3f} ms"
            )

    front = camera_reports["front"]
    return EpisodeValidationReport(
        episode_id=episode_id,
        step_count=len(steps),
        camera_frame_count=front.frame_count,
        image_shape=front.image_shape,
        max_camera_age_ms=front.max_age_ms,
        max_action_age_ms=oldest_action_age_ms,
        training_segment_count=len(training_segments),
        training_segments=training_segments,
        cameras=dict(camera_reports),
        max_intercamera_skew_ms=max_intercamera_skew,
    )

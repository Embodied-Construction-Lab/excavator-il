"""Materialize causal 10 Hz ACT steps from timestamped raw episode streams."""

from __future__ import annotations

import bisect
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .raw_episode import STEP_FIELDS


@dataclass(frozen=True)
class StepBuildReport:
    episode_id: str
    raw_stm32_record_count: int
    new_state_count: int
    training_step_count: int
    rejected_state_count: int
    rejection_reasons: Mapping[str, int]
    max_action_age_ms: float
    max_camera_age_ms: float
    stream_timing: Mapping[str, Mapping[str, float | int]]
    sequence_gaps: Mapping[str, int]
    duplicate_or_out_of_order_count: int
    serial_parse_failure_count: int
    command_write_failure_count: int
    sensor_invalid_count: int
    action_age_ms: Mapping[str, float]
    camera_age_ms: Mapping[str, float]
    camera_queue_drop_count: int
    disk_queue_drop_count: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing required raw stream: {path.name}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number}: record must be an object")
            records.append(value)
    if not records:
        raise ValueError(f"{path.name} contains no records")
    return records


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return [] if not path.is_file() or path.stat().st_size == 0 else _read_jsonl(path)


def _read_camera_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError("missing required camera_front_timestamps.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"camera_frame_index", "camera_stamp_monotonic_ns", "image_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("camera_front_timestamps.csv is empty or missing required columns")
    return rows


def _latest_causal(
    records: list[dict[str, Any]],
    stamps: list[int],
    state_stamp_ns: int,
) -> dict[str, Any] | None:
    index = bisect.bisect_right(stamps, state_stamp_ns) - 1
    return None if index < 0 else records[index]


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _write_steps(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _age_statistics(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _timing_statistics(stamps_ns: list[int]) -> dict[str, float | int]:
    if len(stamps_ns) < 2:
        return {
            "count": len(stamps_ns),
            "estimated_rate_hz": 0.0,
            "mean_period_ms": 0.0,
            "p95_period_ms": 0.0,
            "max_period_ms": 0.0,
        }
    periods_ms = [
        (right - left) / 1_000_000.0
        for left, right in zip(stamps_ns, stamps_ns[1:])
        if right > left
    ]
    if not periods_ms:
        return _timing_statistics(stamps_ns[:1])
    mean_period = sum(periods_ms) / len(periods_ms)
    return {
        "count": len(stamps_ns),
        "estimated_rate_hz": 1000.0 / mean_period,
        "mean_period_ms": mean_period,
        "p95_period_ms": _percentile(periods_ms, 0.95),
        "max_period_ms": max(periods_ms),
    }


def _sequence_health(values: list[int]) -> tuple[int, int]:
    gaps = 0
    duplicate_or_out_of_order = 0
    previous: int | None = None
    for value in values:
        if previous is not None:
            if value <= previous:
                duplicate_or_out_of_order += 1
            elif value > previous + 1:
                gaps += value - previous - 1
        previous = value
    return gaps, duplicate_or_out_of_order


def build_steps(
    episode_path: str | Path,
    *,
    max_action_age_ms: float = 100.0,
    max_camera_age_ms: float = 120.0,
) -> StepBuildReport:
    """Build causal training rows and a quality report from raw streams."""
    if max_action_age_ms <= 0.0 or max_camera_age_ms <= 0.0:
        raise ValueError("maximum action and camera ages must be positive")
    episode = Path(episode_path)
    stm32_records = _read_jsonl(episode / "stm32_raw.jsonl")
    actions = [
        record
        for record in _read_jsonl(episode / "expert_action.jsonl")
        if bool(record.get("action_valid"))
    ]
    if not actions:
        raise ValueError("expert_action.jsonl contains no valid actions")
    actions = sorted(actions, key=lambda item: int(item["action_stamp_monotonic_ns"]))
    action_stamps = [int(item["action_stamp_monotonic_ns"]) for item in actions]

    camera_rows = _read_camera_rows(episode / "camera_front_timestamps.csv")
    cameras: list[dict[str, Any]] = [dict(row) for row in camera_rows]
    cameras = sorted(cameras, key=lambda item: int(item["camera_stamp_monotonic_ns"]))
    camera_stamps = [int(item["camera_stamp_monotonic_ns"]) for item in cameras]
    joystick_records = _read_optional_jsonl(episode / "joystick_raw.jsonl")
    command_records = _read_optional_jsonl(episode / "command_tx.jsonl")

    episode_id = str(stm32_records[0].get("episode_id", episode.name))
    output_rows: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    action_ages_ms: list[float] = []
    camera_ages_ms: list[float] = []
    seen_sensor_sequences: set[int] = set()
    new_state_count = 0

    for raw_record in stm32_records:
        telemetry = raw_record.get("telemetry")
        if not raw_record.get("parse_ok") or not isinstance(telemetry, dict):
            continue
        if int(telemetry.get("sensor_is_new", 0)) != 1:
            continue
        new_state_count += 1
        sensor_seq = int(telemetry["sensor_seq"])
        if sensor_seq in seen_sensor_sequences:
            rejection_reasons["duplicate_sensor_seq"] += 1
            continue
        seen_sensor_sequences.add(sensor_seq)

        state_receive_ns = int(raw_record["orin_receive_monotonic_ns"])
        if not all(int(telemetry.get(field, 0)) == 1 for field in ("rs485_ok", "dwj_ok", "imu_ok")):
            rejection_reasons["sensor_invalid"] += 1
            continue
        if int(telemetry.get("control_mode", 0)) != 1:
            rejection_reasons["not_manual_joystick"] += 1
            continue

        action = _latest_causal(actions, action_stamps, state_receive_ns)
        camera = _latest_causal(cameras, camera_stamps, state_receive_ns)
        if action is None:
            rejection_reasons["no_causal_action"] += 1
            continue
        if camera is None:
            rejection_reasons["no_causal_camera"] += 1
            continue
        action_age_ms = (
            state_receive_ns - int(action["action_stamp_monotonic_ns"])
        ) / 1_000_000.0
        camera_age_ms = (
            state_receive_ns - int(camera["camera_stamp_monotonic_ns"])
        ) / 1_000_000.0
        if action_age_ms > max_action_age_ms:
            rejection_reasons["action_stale"] += 1
            continue
        if camera_age_ms > max_camera_age_ms:
            rejection_reasons["camera_stale"] += 1
            continue
        if not (episode / str(camera["image_path"])).is_file():
            rejection_reasons["camera_file_missing"] += 1
            continue

        action_ages_ms.append(action_age_ms)
        camera_ages_ms.append(camera_age_ms)
        output_rows.append(
            {
                "episode_id": episode_id,
                "frame_index": len(output_rows),
                "state_seq": sensor_seq,
                "state_stamp_ms": int(telemetry["sensor_stamp_ms"]),
                "state_receive_monotonic_ns": state_receive_ns,
                "action_stamp_monotonic_ns": int(action["action_stamp_monotonic_ns"]),
                "boom_pos_m": _finite_float(telemetry["boom_pos_mm"], "boom_pos_mm") / 1000.0,
                "stick_pos_m": _finite_float(telemetry["stick_pos_mm"], "stick_pos_mm") / 1000.0,
                "bucket_pos_m": _finite_float(telemetry["bucket_pos_mm"], "bucket_pos_mm") / 1000.0,
                "boom_vel_mps": _finite_float(telemetry["boom_vel_mmps"], "boom_vel_mmps") / 1000.0,
                "stick_vel_mps": _finite_float(telemetry["stick_vel_mmps"], "stick_vel_mmps") / 1000.0,
                "bucket_vel_mps": _finite_float(telemetry["bucket_vel_mmps"], "bucket_vel_mmps") / 1000.0,
                "boom_angle_rad": math.radians(_finite_float(telemetry["boom_angle_deg"], "boom_angle_deg")),
                "arm_angle_rad": math.radians(_finite_float(telemetry["arm_angle_deg"], "arm_angle_deg")),
                "bucket_angle_rad": math.radians(_finite_float(telemetry["bucket_angle_deg"], "bucket_angle_deg")),
                "swing_angle_rad": math.radians(_finite_float(telemetry["swing_angle_deg"], "swing_angle_deg")),
                "swing_vel_radps": math.radians(_finite_float(telemetry["swing_vel_degps"], "swing_vel_degps")),
                "action_boom": _finite_float(action["action_boom"], "action_boom"),
                "action_stick": _finite_float(action["action_stick"], "action_stick"),
                "action_bucket": _finite_float(action["action_bucket"], "action_bucket"),
                "action_swing": _finite_float(action["action_swing"], "action_swing"),
                "pump_percent": _finite_float(telemetry["pump_percent"], "pump_percent"),
                "sensor_valid": 1,
                "control_mode": "manual_joystick",
            }
        )

    if not output_rows:
        details = ", ".join(f"{key}={value}" for key, value in rejection_reasons.items())
        raise ValueError(f"episode contains no eligible training steps ({details})")
    _write_steps(episode / "steps.csv", output_rows)
    report = StepBuildReport(
        episode_id=episode_id,
        raw_stm32_record_count=len(stm32_records),
        new_state_count=new_state_count,
        training_step_count=len(output_rows),
        rejected_state_count=sum(rejection_reasons.values()),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        max_action_age_ms=max(action_ages_ms),
        max_camera_age_ms=max(camera_ages_ms),
        stream_timing={
            "stm32_telemetry": _timing_statistics(
                [
                    int(record["orin_receive_monotonic_ns"])
                    for record in stm32_records
                    if record.get("parse_ok") and isinstance(record.get("telemetry"), dict)
                ]
            ),
            "new_sensor_state": _timing_statistics(
                [
                    int(record["orin_receive_monotonic_ns"])
                    for record in stm32_records
                    if record.get("parse_ok")
                    and isinstance(record.get("telemetry"), dict)
                    and int(record["telemetry"].get("sensor_is_new", 0)) == 1
                ]
            ),
            "expert_action": _timing_statistics(action_stamps),
            "camera_front": _timing_statistics(camera_stamps),
        },
        sequence_gaps={
            "stm32_control": _sequence_health(
                [
                    int(record["telemetry"]["control_seq"])
                    for record in stm32_records
                    if record.get("parse_ok") and isinstance(record.get("telemetry"), dict)
                ]
            )[0],
            "joystick": _sequence_health(
                [
                    int(record["joystick_sample_seq"])
                    for record in joystick_records
                    if record.get("parse_ok") and record.get("joystick_sample_seq") is not None
                ]
            )[0],
            "camera": _sequence_health(
                [int(record["camera_frame_index"]) for record in cameras]
            )[0],
        },
        duplicate_or_out_of_order_count=_sequence_health(
            [
                int(record["joystick_sample_seq"])
                for record in joystick_records
                if record.get("parse_ok") and record.get("joystick_sample_seq") is not None
            ]
        )[1],
        serial_parse_failure_count=sum(
            1 for record in stm32_records if not record.get("parse_ok")
        ),
        command_write_failure_count=sum(
            1 for record in command_records if record.get("write_ok") is not True
        ),
        sensor_invalid_count=int(rejection_reasons.get("sensor_invalid", 0)),
        action_age_ms=_age_statistics(action_ages_ms),
        camera_age_ms=_age_statistics(camera_ages_ms),
        camera_queue_drop_count=0,
        disk_queue_drop_count=0,
    )
    (episode / "quality_report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report

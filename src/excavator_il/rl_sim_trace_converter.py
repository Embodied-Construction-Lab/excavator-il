"""Fail-closed conversion of Unity RL control audits into paired traces.

The converter accepts only simulator records that carry an explicit frozen
trajectory-suite binding and explicit policy-decision identity.  It never
derives either identity from CSV row order, timestamps, or action changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile

from .experiment_run import ExperimentRunValidationError
from .rl_sim_experiment_run import (
    RL_CONTROL_TRACE_SCHEMA_VERSION,
    RL_CONTROL_TRACE_SEMANTICS,
    load_trajectory_suite_snapshot,
)


RL_SIM_CONTROL_AUDIT_SCHEMA_VERSION = "rl_excavator_sim_control_audit.v1"
_ACTION_ORDER = ("boom", "stick", "bucket", "swing")


@dataclass(frozen=True)
class RlSimControlTraceExport:
    output_path: Path
    trace_run_id: str
    sample_count: int
    first_sample_id: int
    last_sample_id: int
    first_policy_action_seq: int
    last_policy_action_seq: int
    trajectory_suite_sha256: str
    terminal_result: str
    trace_semantics: str = RL_CONTROL_TRACE_SEMANTICS


@dataclass(frozen=True)
class _TraceConversion:
    records: tuple[dict[str, object], ...]
    policy_action_sequences: tuple[int, ...]
    terminal_result: str


def export_rl_sim_control_trace(
    audit_path: str | Path,
    output_path: str | Path,
    *,
    trajectory_suite_path: str | Path,
    trace_run_id: str,
) -> RlSimControlTraceExport:
    """Validate one simulator audit segment and publish a canonical trace."""

    source = Path(audit_path).expanduser()
    target = Path(output_path).expanduser()
    suite_path = Path(trajectory_suite_path).expanduser()
    selected_trace_run_id = _required_text(trace_run_id, "trace_run_id")
    _validate_output_path(source, target)
    suite, suite_sha256 = load_trajectory_suite_snapshot(suite_path)
    records = _read_jsonl(source)
    selected = _select_contiguous_segment(records, selected_trace_run_id)
    if not selected:
        raise ExperimentRunValidationError(
            f"simulator control audit contains no trace_run_id={selected_trace_run_id}"
        )

    conversion = _convert_selected_records(
        selected,
        suite_sample_ids=frozenset(suite["sample_ids"]),
        trace_run_id=selected_trace_run_id,
        trajectory_suite_sha256=suite_sha256,
    )
    _atomic_write_jsonl(target, conversion.records)
    return RlSimControlTraceExport(
        output_path=target,
        trace_run_id=selected_trace_run_id,
        sample_count=len(conversion.policy_action_sequences),
        first_sample_id=int(conversion.records[0]["sample_id"]),
        last_sample_id=int(conversion.records[-2]["sample_id"]),
        first_policy_action_seq=conversion.policy_action_sequences[0],
        last_policy_action_seq=conversion.policy_action_sequences[-1],
        trajectory_suite_sha256=suite_sha256,
        terminal_result=conversion.terminal_result,
    )


def _convert_selected_records(
    selected: list[tuple[int, dict[str, object]]],
    *,
    suite_sample_ids: frozenset[int],
    trace_run_id: str,
    trajectory_suite_sha256: str,
) -> _TraceConversion:
    trace_records: list[dict[str, object]] = []
    policy_action_sequences: list[int] = []
    previous_sample_id = -1
    previous_policy_action_seq = -1
    previous_stamp_s = -math.inf
    if any(record.get("record_type") == "audit_error" for _, record in selected):
        raise ExperimentRunValidationError(
            "selected simulator audit segment contains audit_error"
        )
    terminal_records = [
        (line_number, record)
        for line_number, record in selected
        if record.get("record_type") == "terminal"
    ]
    if len(terminal_records) != 1 or terminal_records[0] != selected[-1]:
        raise ExperimentRunValidationError(
            "simulator control audit must contain exactly one final terminal record"
        )
    policy_records = selected[:-1]
    if not policy_records:
        raise ExperimentRunValidationError(
            "simulator control audit must contain at least one policy_sample record"
        )
    if any(record.get("record_type") != "policy_sample" for _, record in policy_records):
        raise ExperimentRunValidationError(
            "simulator control audit records before terminal must be policy_sample"
        )
    for line_number, record in policy_records:
        (
            sample_id,
            policy_action_seq,
            stamp_s,
            action,
            bucket_tip_ros_m,
            reference_waypoint_ros_m,
            waypoint_index,
            waypoint_distance_m,
            episode_progress,
        ) = _validate_record(
            record,
            line_number=line_number,
            trace_run_id=trace_run_id,
            trajectory_suite_sha256=trajectory_suite_sha256,
        )
        if sample_id not in suite_sample_ids:
            raise ExperimentRunValidationError(
                "simulator control audit contains sample_id values outside the trajectory suite"
            )
        if sample_id != len(trace_records):
            raise ExperimentRunValidationError(
                "simulator control audit sample_id values must form a contiguous "
                "prefix starting at 0"
            )
        if sample_id <= previous_sample_id:
            raise ExperimentRunValidationError(
                "simulator control audit sample_id must be unique and strictly increasing"
            )
        if policy_action_seq <= previous_policy_action_seq:
            raise ExperimentRunValidationError(
                "simulator control audit policy_action_seq must be unique and strictly increasing"
            )
        if stamp_s <= previous_stamp_s:
            raise ExperimentRunValidationError(
                "simulator control audit runtime_monotonic_s must be strictly increasing"
            )
        trace_records.append(
            {
                "schema_version": RL_CONTROL_TRACE_SCHEMA_VERSION,
                "record_type": "policy_sample",
                "sample_id": sample_id,
                "stamp_s": stamp_s,
                "action_order": list(_ACTION_ORDER),
                "action": list(action),
                "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
                "trajectory_suite_sha256": trajectory_suite_sha256,
                "bucket_tip_ros_m": list(bucket_tip_ros_m),
                "reference_waypoint_ros_m": list(reference_waypoint_ros_m),
                "waypoint_index": waypoint_index,
                "waypoint_distance_m": waypoint_distance_m,
                "episode_progress": episode_progress,
                "result": "ACTIVE",
            }
        )
        policy_action_sequences.append(policy_action_seq)
        previous_sample_id = sample_id
        previous_policy_action_seq = policy_action_seq
        previous_stamp_s = stamp_s
    terminal = _validate_terminal_record(
        terminal_records[0][1],
        line_number=terminal_records[0][0],
        trace_run_id=trace_run_id,
        trajectory_suite_sha256=trajectory_suite_sha256,
        previous_stamp_s=previous_stamp_s,
        expected_consumed_sample_count=len(policy_records),
        expected_suite_sample_count=len(suite_sample_ids),
    )
    trace_records.append(
        {
            "schema_version": RL_CONTROL_TRACE_SCHEMA_VERSION,
            "record_type": "terminal",
            "stamp_s": terminal[0],
            "elapsed_s": terminal[1],
            "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
            "trajectory_suite_sha256": trajectory_suite_sha256,
            "result": terminal[2],
        }
    )
    return _TraceConversion(
        records=tuple(trace_records),
        policy_action_sequences=tuple(policy_action_sequences),
        terminal_result=terminal[2],
    )


def _select_contiguous_segment(
    records: list[dict[str, object]],
    trace_run_id: str,
) -> list[tuple[int, dict[str, object]]]:
    matching_indexes = [
        index
        for index, record in enumerate(records)
        if record.get("trace_run_id") == trace_run_id
    ]
    if not matching_indexes:
        return []
    first_index = matching_indexes[0]
    last_index = matching_indexes[-1]
    if any(
        records[index].get("trace_run_id") != trace_run_id
        for index in range(first_index, last_index + 1)
    ):
        raise ExperimentRunValidationError(
            "trace_run_id must identify one contiguous simulator audit segment"
        )
    return [
        (index + 1, records[index])
        for index in range(first_index, last_index + 1)
    ]


def _validate_output_path(source: Path, target: Path) -> None:
    if target.is_symlink():
        raise ExperimentRunValidationError(
            f"simulator trajectory trace output must be a regular file: {target}"
        )
    if source.resolve() == target.resolve() or (
        target.exists() and os.path.samefile(source, target)
    ):
        raise ExperimentRunValidationError(
            "simulator trajectory trace output must not overwrite the source audit"
        )
    if target.exists():
        raise ExperimentRunValidationError(
            f"simulator trajectory trace output already exists: {target}"
        )


def _atomic_write_jsonl(
    path: Path,
    records: tuple[dict[str, object], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
        + "\n"
        for record in records
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise ExperimentRunValidationError(
                f"simulator trajectory trace output already exists: {path}"
            ) from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunValidationError(
            f"simulator control audit must be a regular file: {path}"
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ExperimentRunValidationError(
            f"cannot read simulator control audit {path}: {exc}"
        ) from exc
    if any(
        line.lstrip("\ufeff").strip() == "# rl_excavator_open_loop_velocity_export"
        for line in lines[:8]
    ):
        raise ExperimentRunValidationError(
            "legacy OpenLoopVelocityRecorder CSV cannot be converted: explicit "
            "trace_run_id, policy_action_seq and frozen trajectory-suite binding "
            "are missing"
        )
    if not lines or any(not line.strip() for line in lines):
        raise ExperimentRunValidationError(
            "simulator control audit must contain non-empty JSON records"
        )
    result: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentRunValidationError(
                f"simulator control audit line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ExperimentRunValidationError(
                f"simulator control audit line {line_number} must be a JSON object"
            )
        _validate_audit_envelope(value, line_number=line_number)
        result.append(value)
    return result


def _validate_audit_envelope(
    value: dict[str, object], *, line_number: int
) -> None:
    if value.get("schema_version") != RL_SIM_CONTROL_AUDIT_SCHEMA_VERSION:
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} schema_version must be "
            f"{RL_SIM_CONTROL_AUDIT_SCHEMA_VERSION}"
        )
    if value.get("record_type") not in {"policy_sample", "terminal", "audit_error"}:
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} record_type is not recognized"
        )
    trace_run_id = value.get("trace_run_id")
    if not isinstance(trace_run_id, str) or not trace_run_id.strip():
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} trace_run_id must be non-empty text"
        )


def _validate_record(
    value: dict[str, object],
    *,
    line_number: int,
    trace_run_id: str,
    trajectory_suite_sha256: str,
) -> tuple[
    int,
    int,
    float,
    tuple[float, float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    int,
    float,
    float,
]:
    expected = {
        "schema_version": RL_SIM_CONTROL_AUDIT_SCHEMA_VERSION,
        "mode": "control",
        "status": "active",
        "trajectory_controller_backend": "onnx_rl",
        "trajectory_suite_sha256": trajectory_suite_sha256,
        "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
        "trace_run_id": trace_run_id,
        "record_type": "policy_sample",
        "result": "active",
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ExperimentRunValidationError(
                f"simulator control audit line {line_number} {field} must be {expected_value}"
            )
    if value.get("action_order") != list(_ACTION_ORDER):
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} action_order must be "
            "[boom, stick, bucket, swing]"
        )
    sample_id = _required_int(value, "sample_id", line_number)
    policy_action_seq = _required_int(value, "policy_action_seq", line_number)
    stamp_s = _required_float(value, "runtime_monotonic_s", line_number)
    action_value = value.get("commanded_normalized_action")
    if not isinstance(action_value, list) or len(action_value) != 4:
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} commanded_normalized_action "
            "must contain four values"
        )
    action = tuple(
        _finite_float(item, "commanded_normalized_action", line_number)
        for item in action_value
    )
    if any(item < -1.0 or item > 1.0 for item in action):
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} commanded_normalized_action "
            "must be within [-1, 1]"
        )
    bucket_tip_ros_m = _point3(value, "bucket_tip_ros_m", line_number)
    reference_waypoint_ros_m = _point3(
        value, "reference_waypoint_ros_m", line_number
    )
    waypoint_index = _required_int(value, "waypoint_index", line_number)
    waypoint_distance_m = _required_float(
        value, "waypoint_distance_m", line_number
    )
    episode_progress = _required_float(value, "episode_progress", line_number)
    if episode_progress > 1.0:
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} episode_progress must be within [0, 1]"
        )
    return (  # type: ignore[return-value]
        sample_id,
        policy_action_seq,
        stamp_s,
        action,
        bucket_tip_ros_m,
        reference_waypoint_ros_m,
        waypoint_index,
        waypoint_distance_m,
        episode_progress,
    )


def _validate_terminal_record(
    value: dict[str, object],
    *,
    line_number: int,
    trace_run_id: str,
    trajectory_suite_sha256: str,
    previous_stamp_s: float,
    expected_consumed_sample_count: int,
    expected_suite_sample_count: int,
) -> tuple[float, float, str]:
    expected = {
        "schema_version": RL_SIM_CONTROL_AUDIT_SCHEMA_VERSION,
        "record_type": "terminal",
        "mode": "control",
        "status": "terminal",
        "trajectory_controller_backend": "onnx_rl",
        "trajectory_suite_sha256": trajectory_suite_sha256,
        "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
        "trace_run_id": trace_run_id,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ExperimentRunValidationError(
                f"simulator control audit line {line_number} {field} must be {expected_value}"
            )
    stamp_s = _required_float(value, "runtime_monotonic_s", line_number)
    if stamp_s <= previous_stamp_s:
        raise ExperimentRunValidationError(
            "simulator control audit terminal runtime_monotonic_s must follow policy samples"
        )
    elapsed_s = _required_float(value, "elapsed_s", line_number)
    consumed_sample_count = _required_positive_int(
        value,
        "consumed_sample_count",
        line_number,
    )
    if consumed_sample_count != expected_consumed_sample_count:
        raise ExperimentRunValidationError(
            "simulator control audit terminal consumed_sample_count must equal "
            f"{expected_consumed_sample_count}"
        )
    suite_sample_count = _required_positive_int(
        value,
        "suite_sample_count",
        line_number,
    )
    if suite_sample_count != expected_suite_sample_count:
        raise ExperimentRunValidationError(
            "simulator control audit terminal suite_sample_count must equal "
            f"{expected_suite_sample_count}"
        )
    result = value.get("result")
    result_map = {
        "completed": "COMPLETED",
        "timeout": "TIMEOUT",
        "rejected": "REJECTED",
        "interrupted": "INTERRUPTED",
    }
    if result not in result_map:
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} terminal result must be "
            "completed, timeout, rejected or interrupted"
        )
    return stamp_s, elapsed_s, result_map[result]


def _point3(
    value: dict[str, object], field: str, line_number: int
) -> tuple[float, float, float]:
    item = value.get(field)
    if not isinstance(item, list) or len(item) != 3:
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} {field} must contain three values"
        )
    point = tuple(_finite_float(axis, field, line_number) for axis in item)
    return point  # type: ignore[return-value]


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentRunValidationError(f"{label} must be non-empty text")
    return value


def _required_int(value: dict[str, object], field: str, line_number: int) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} {field} must be a non-negative int"
        )
    return item


def _required_positive_int(
    value: dict[str, object], field: str, line_number: int
) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} {field} must be a positive int"
        )
    return item


def _required_float(value: dict[str, object], field: str, line_number: int) -> float:
    item = _finite_float(value.get(field), field, line_number)
    if item < 0.0:
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} {field} must be non-negative"
        )
    return item


def _finite_float(value: object, field: str, line_number: int) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ExperimentRunValidationError(
            f"simulator control audit line {line_number} {field} must contain finite numbers"
        )
    return float(value)


__all__ = [
    "RL_SIM_CONTROL_AUDIT_SCHEMA_VERSION",
    "RlSimControlTraceExport",
    "export_rl_sim_control_trace",
]

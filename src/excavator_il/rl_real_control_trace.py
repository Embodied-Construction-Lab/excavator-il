"""Strictly convert explicit Orin RL policy samples into a paired trace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from .experiment_run import (
    EvidenceRequirement,
    ExperimentRun,
    ExperimentRunSnapshot,
    ExperimentRunValidationError,
    TaskContext,
    fingerprint_path,
)
from .rl_sim_experiment_run import (
    RL_CONTROL_TRACE_SCHEMA_VERSION,
    RL_CONTROL_TRACE_SEMANTICS,
    load_machine_profile_action_order,
    load_rl_control_trace_snapshot,
    load_trajectory_suite_snapshot,
)


ORIN_EDGE_CONTROL_AUDIT_SCHEMA_VERSION = "orin_edge_control_audit.v1"
_ACTION_ORDER = ("boom", "stick", "bucket", "swing")
_POLICY_SAMPLE_FIELDS = (
    "sample_id", "policy_action_seq", "action_order", "normalized_action",
    "commanded_normalized_action", "physical_action")


@dataclass(frozen=True)
class RlRealControlTraceExport:
    output_path: Path
    trace_run_id: str
    sample_count: int
    first_sample_id: int
    last_sample_id: int
    first_policy_action_seq: int
    last_policy_action_seq: int
    source_audit_sha256: str
    trace_sha256: str
    trajectory_suite_sha256: str
    terminal_result: str
    trace_semantics: str = RL_CONTROL_TRACE_SEMANTICS


@dataclass(frozen=True)
class RlRealExperimentRunRequest:
    experiment_run_root: Path
    machine_profile_path: Path
    trajectory_suite_path: Path
    trajectory_controller_onnx_path: Path
    control_audit_path: Path
    trace_output_path: Path
    trace_run_id: str
    trajectory_suite_sha256: str
    policy_id: str
    evaluation_scope: str
    task_variant: str
    operator_id: str
    material_id: str
    soil_reset_block_id: str | None = None
    dig_point_id: str | None = None
    host_topology: Mapping[str, Any] = MappingProxyType({})
    repository_paths: Mapping[str, Path] = MappingProxyType({})
    config_paths: Mapping[str, Path] = MappingProxyType({})
    run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_run_root", Path(
            self.experiment_run_root
        ).expanduser())
        for field_name in (
            "machine_profile_path",
            "trajectory_suite_path",
            "trajectory_controller_onnx_path",
            "control_audit_path",
            "trace_output_path",
        ):
            object.__setattr__(self, field_name, Path(
                getattr(self, field_name)
            ).expanduser())
        object.__setattr__(
            self,
            "repository_paths",
            MappingProxyType(
                {
                    str(label): Path(path).expanduser()
                    for label, path in self.repository_paths.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "config_paths",
            MappingProxyType(
                {
                    str(label): Path(path).expanduser()
                    for label, path in self.config_paths.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "host_topology",
            MappingProxyType(json.loads(json.dumps(dict(self.host_topology)))),
        )
        for field_name in (
            "policy_id",
            "trace_run_id",
            "evaluation_scope",
            "task_variant",
            "operator_id",
            "material_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ExperimentRunValidationError(
                    f"{field_name} must be non-empty text"
                )
        _sha256(self.trajectory_suite_sha256, "trajectory_suite_sha256")
        for field_name in ("soil_reset_block_id", "dig_point_id", "run_id"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ExperimentRunValidationError(
                    f"{field_name} must be non-empty text when provided"
                )
        if self.evaluation_scope not in {
            "training_internal",
            "held_out_experiment",
        }:
            raise ExperimentRunValidationError(
                "evaluation_scope must be training_internal or held_out_experiment"
            )
        if not self.policy_id.startswith("onnx_rl:"):
            raise ExperimentRunValidationError("policy_id must start with onnx_rl:")


def export_rl_real_control_trace(
    audit_path: str | Path,
    output_path: str | Path,
    *,
    trace_run_id: str,
    trajectory_suite_sha256: str,
) -> RlRealControlTraceExport:
    """Validate one Orin audit segment and atomically publish a strict trace."""

    source = Path(audit_path).expanduser()
    target = Path(output_path).expanduser()
    selected_trace_run_id = _required_text(trace_run_id, "trace_run_id")
    selected_suite_sha256 = _sha256(trajectory_suite_sha256, "trajectory_suite_sha256")
    records, source_audit_sha256 = _read_jsonl(source)
    if target.is_symlink():
        raise ExperimentRunValidationError(
            f"real trajectory trace output must be a regular file: {target}"
        )
    if source.resolve() == target.resolve() or (
        target.exists() and os.path.samefile(source, target)
    ):
        raise ExperimentRunValidationError(
            "real trajectory trace output must not overwrite the source audit"
        )
    if target.exists():
        raise ExperimentRunValidationError(
            f"real trajectory trace output already exists: {target}"
        )
    selected_records = [
        (line_number, record)
        for line_number, record in enumerate(records, start=1)
        if record.get("trace_run_id") == selected_trace_run_id
    ]
    if not selected_records:
        raise ExperimentRunValidationError(
            f"Orin control audit contains no trace_run_id={selected_trace_run_id}"
        )
    selected_line_numbers = [line_number for line_number, _ in selected_records]
    if selected_line_numbers != list(
        range(selected_line_numbers[0], selected_line_numbers[-1] + 1)
    ):
        raise ExperimentRunValidationError(
            "trace_run_id must identify one contiguous audit segment"
        )
    trace_records: list[dict[str, object]] = []
    previous_sample_id = -1
    previous_policy_action_seq = -1
    previous_stamp_s = -math.inf
    if any(
        record.get("record_type") == "audit_error"
        for _, record in selected_records
    ):
        raise ExperimentRunValidationError(
            "selected Orin audit segment contains audit_error"
        )
    terminal_records = [
        (line_number, record)
        for line_number, record in selected_records
        if record.get("record_type") == "terminal"
    ]
    if len(terminal_records) != 1 or terminal_records[0] != selected_records[-1]:
        raise ExperimentRunValidationError(
            "Orin control audit must contain exactly one final terminal record"
        )
    policy_records = selected_records[:-1]
    if not policy_records:
        raise ExperimentRunValidationError(
            "Orin control audit must contain at least one policy_sample record"
        )
    if any(record.get("record_type") != "policy_sample" for _, record in policy_records):
        raise ExperimentRunValidationError(
            "Orin control audit records before terminal must be policy_sample"
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
        ) = _validate_audit_record(
            record,
            line_number=line_number,
            trace_run_id=selected_trace_run_id,
        )
        if sample_id != len(trace_records):
            raise ExperimentRunValidationError(
                "Orin control audit sample_id values must form a contiguous "
                "prefix starting at 0"
            )
        if sample_id <= previous_sample_id:
            raise ExperimentRunValidationError(
                "Orin control audit sample_id must be unique and strictly increasing"
            )
        if policy_action_seq <= previous_policy_action_seq:
            raise ExperimentRunValidationError(
                "Orin control audit policy_action_seq must be unique and strictly increasing"
            )
        if stamp_s <= previous_stamp_s:
            raise ExperimentRunValidationError(
                "Orin control audit runtime_monotonic_s must be strictly increasing"
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
                "trajectory_suite_sha256": selected_suite_sha256,
                "bucket_tip_ros_m": list(bucket_tip_ros_m),
                "reference_waypoint_ros_m": list(reference_waypoint_ros_m),
                "waypoint_index": waypoint_index,
                "waypoint_distance_m": waypoint_distance_m,
                "episode_progress": episode_progress,
                "result": "ACTIVE",
            }
        )
        previous_sample_id = sample_id
        previous_policy_action_seq = policy_action_seq
        previous_stamp_s = stamp_s

    terminal = _validate_terminal_record(
        terminal_records[0][1],
        line_number=terminal_records[0][0],
        trace_run_id=selected_trace_run_id,
        previous_stamp_s=previous_stamp_s,
        persisted_policy_sample_count=len(policy_records),
    )
    trace_records.append(
        {
            "schema_version": RL_CONTROL_TRACE_SCHEMA_VERSION,
            "record_type": "terminal",
            "stamp_s": terminal[0],
            "elapsed_s": terminal[1],
            "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
            "trajectory_suite_sha256": selected_suite_sha256,
            "result": terminal[2],
        }
    )
    trace_sha256 = _atomic_write_jsonl(target, trace_records)
    return RlRealControlTraceExport(
        output_path=target,
        trace_run_id=selected_trace_run_id,
        sample_count=len(policy_records),
        first_sample_id=int(trace_records[0]["sample_id"]),
        last_sample_id=int(trace_records[-2]["sample_id"]),
        first_policy_action_seq=_required_int(
            policy_records[0][1], "policy_action_seq", 1
        ),
        last_policy_action_seq=_required_int(
            policy_records[-1][1],
            "policy_action_seq",
            len(policy_records),
        ),
        source_audit_sha256=source_audit_sha256,
        trace_sha256=trace_sha256,
        trajectory_suite_sha256=selected_suite_sha256,
        terminal_result=terminal[2],
    )


def record_rl_real_experiment_run(
    request: RlRealExperimentRunRequest,
) -> ExperimentRunSnapshot:
    """Export and retain one strict real-machine parent for paired evaluation."""

    if not isinstance(request, RlRealExperimentRunRequest):
        raise TypeError("request must be RlRealExperimentRunRequest")
    machine_profile_fingerprint = fingerprint_path(request.machine_profile_path)
    action_order = load_machine_profile_action_order(request.machine_profile_path)
    trajectory_suite, trajectory_suite_sha256 = load_trajectory_suite_snapshot(
        request.trajectory_suite_path
    )
    if request.trajectory_suite_sha256 != trajectory_suite_sha256:
        raise ExperimentRunValidationError(
            "explicit trajectory_suite_sha256 does not match the frozen "
            "trajectory suite artifact"
        )
    onnx_fingerprint = fingerprint_path(request.trajectory_controller_onnx_path)
    if onnx_fingerprint.object_type != "file":
        raise ExperimentRunValidationError(
            "trajectory_controller_onnx_path must be a file"
        )
    trace_export = export_rl_real_control_trace(
        request.control_audit_path,
        request.trace_output_path,
        trace_run_id=request.trace_run_id,
        trajectory_suite_sha256=request.trajectory_suite_sha256,
    )
    trace_document, trace_sha256 = load_rl_control_trace_snapshot(
        trace_export.output_path
    )
    trace = trace_document.samples
    if trace_sha256 != trace_export.trace_sha256:
        raise ExperimentRunValidationError(
            "real trajectory trace changed after export"
        )
    if {sample.trajectory_suite_sha256 for sample in trace} != {
        trajectory_suite_sha256
    }:
        raise ExperimentRunValidationError(
            "real trajectory trace trajectory_suite_sha256 does not match the "
            "trajectory suite artifact"
        )
    suite_sample_ids = set(trajectory_suite["sample_ids"])
    if set(sample.sample_id for sample in trace) - suite_sample_ids:
        raise ExperimentRunValidationError(
            "real trajectory trace contains sample_id values outside the trajectory suite"
        )
    run = ExperimentRun.create(
        request.experiment_run_root,
        run_id=request.run_id,
        run_kind="hybrid_live",
        task_context=TaskContext(
            task_variant=request.task_variant,
            soil_reset_block_id=request.soil_reset_block_id,
            dig_point_id=request.dig_point_id,
            operator_id=request.operator_id,
            material_id=request.material_id,
        ),
        policy_ids={"trajectory_controller": request.policy_id},
        host_topology=request.host_topology,
        repository_paths=request.repository_paths,
        config_paths=request.config_paths,
        machine_profile_path=request.machine_profile_path,
        evidence_requirements={
            "rl_onnx_model": EvidenceRequirement(required=True, min_count=1),
            "trajectory_suite": EvidenceRequirement(required=True, min_count=1),
            "rl_control_audit": EvidenceRequirement(required=True, min_count=1),
            "trajectory_trace": EvidenceRequirement(required=True, min_count=1),
        },
    )
    start_snapshot = run.snapshot().start
    if (
        start_snapshot["machine_profile"]["sha256"]
        != machine_profile_fingerprint.sha256
    ):
        raise ExperimentRunValidationError(
            "machine profile changed while recording evidence"
        )
    onnx_record = run.register_artifact(
        "trajectory_controller_onnx",
        request.trajectory_controller_onnx_path,
        role="rl_onnx_model",
        metadata={"policy": "trajectory_controller"},
    )
    if onnx_record["sha256"] != onnx_fingerprint.sha256:
        raise ExperimentRunValidationError(
            "trajectory controller ONNX changed while recording evidence"
        )
    suite_record = run.register_artifact(
        "trajectory_suite",
        request.trajectory_suite_path,
        role="trajectory_suite",
        metadata={"suite_id": trajectory_suite["suite_id"]},
    )
    if suite_record["sha256"] != trajectory_suite_sha256:
        raise ExperimentRunValidationError(
            "trajectory suite changed while recording evidence"
        )
    audit_record = run.register_artifact(
        "rl_control_audit",
        request.control_audit_path,
        role="rl_control_audit",
        metadata={
            "schema_version": ORIN_EDGE_CONTROL_AUDIT_SCHEMA_VERSION,
            "trajectory_controller_backend": "onnx_rl",
            "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
            "trace_run_id": trace_export.trace_run_id,
            "trajectory_suite_sha256": trace_export.trajectory_suite_sha256,
            "terminal_result": trace_document.terminal.result,
            "terminal_stamp_s": trace_document.terminal.stamp_s,
            "terminal_elapsed_s": trace_document.terminal.elapsed_s,
        },
    )
    if audit_record["sha256"] != trace_export.source_audit_sha256:
        raise ExperimentRunValidationError(
            "control audit changed while recording evidence"
        )
    trace_record = run.register_artifact(
        "trajectory_trace",
        trace_export.output_path,
        role="trajectory_trace",
        metadata={
            "sample_count": trace_export.sample_count,
            "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
            "source_audit_sha256": audit_record["sha256"],
            "trace_run_id": trace_export.trace_run_id,
            "trajectory_suite_sha256": trace_export.trajectory_suite_sha256,
        },
    )
    if trace_record["sha256"] != trace_export.trace_sha256:
        raise ExperimentRunValidationError(
            "trajectory trace changed while recording evidence"
        )
    return run.finalize(
        "success",
        metrics={
            "evaluation_scope": request.evaluation_scope,
            "machine_profile_sha256": run.snapshot().start["machine_profile"][
                "sha256"
            ],
            "action_order": list(action_order),
            "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
            "trace_run_id": trace_export.trace_run_id,
            "trajectory_controller_onnx_sha256": onnx_record["sha256"],
            "trajectory_suite_sha256": suite_record["sha256"],
            "control_audit_sha256": audit_record["sha256"],
            "trace_sha256": trace_record["sha256"],
            "sample_count": trace_export.sample_count,
            "first_sample_id": trace_export.first_sample_id,
            "last_sample_id": trace_export.last_sample_id,
            "first_policy_action_seq": trace_export.first_policy_action_seq,
            "last_policy_action_seq": trace_export.last_policy_action_seq,
        },
        summary=(
            "Strict real-machine RL control audit and trajectory evidence retained "
            "for sim-real comparison."
        ),
    )


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunValidationError(
            f"Orin control audit must be a regular file: {path}"
        )
    try:
        payload = path.read_bytes()
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ExperimentRunValidationError(
            f"cannot read Orin control audit {path}: {exc}"
        ) from exc
    if not lines or any(not line.strip() for line in lines):
        raise ExperimentRunValidationError(
            "Orin control audit must contain non-empty JSON records"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentRunValidationError(
                f"Orin control audit line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ExperimentRunValidationError(
                f"Orin control audit line {line_number} must be a JSON object"
            )
        _validate_audit_envelope(value, line_number=line_number)
        records.append(value)
    return records, hashlib.sha256(payload).hexdigest()


def _validate_audit_envelope(
    value: dict[str, Any], *, line_number: int
) -> None:
    if value.get("schema_version") != ORIN_EDGE_CONTROL_AUDIT_SCHEMA_VERSION:
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} schema_version must be "
            f"{ORIN_EDGE_CONTROL_AUDIT_SCHEMA_VERSION}"
        )
    if value.get("record_type") not in {"policy_sample", "terminal", "audit_error"}:
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} record_type is not recognized"
        )
    trace_run_id = value.get("trace_run_id")
    if not isinstance(trace_run_id, str) or not trace_run_id.strip():
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} trace_run_id must be non-empty text"
        )


def _validate_audit_record(
    value: dict[str, Any],
    *,
    line_number: int,
    trace_run_id: str,
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
    expected_text = {
        "schema_version": ORIN_EDGE_CONTROL_AUDIT_SCHEMA_VERSION,
        "mode": "control",
        "trajectory_controller_backend": "onnx_rl",
        "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
        "trace_run_id": trace_run_id,
        "record_type": "policy_sample",
        "status": "active",
        "result": "active",
    }
    for field, expected in expected_text.items():
        if value.get(field) != expected:
            raise ExperimentRunValidationError(
                f"Orin control audit line {line_number} {field} must be {expected}"
            )
    if value.get("action_order") != list(_ACTION_ORDER):
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} action_order must be "
            "[boom, stick, bucket, swing]"
        )
    sample_id = _required_int(value, "sample_id", line_number)
    policy_action_seq = _required_int(value, "policy_action_seq", line_number)
    stamp_s = _required_float(value, "runtime_monotonic_s", line_number)
    action_value = value.get("commanded_normalized_action")
    if not isinstance(action_value, list) or len(action_value) != 4:
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} commanded_normalized_action "
            "must contain four values"
        )
    action = tuple(
        _finite_float(item, "commanded_normalized_action", line_number)
        for item in action_value
    )
    if any(value < -1.0 or value > 1.0 for value in action):
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} commanded_normalized_action "
            "must be within [-1, 1]"
        )
    bucket_tip_ros_m = _point3(value, "bucket_tip_ros_m", line_number)
    reference_waypoint_ros_m = _point3(value, "reference_waypoint_ros_m", line_number)
    waypoint_index = _required_int(value, "waypoint_index", line_number)
    waypoint_distance_m = _required_float(
        value, "waypoint_distance_m", line_number
    )
    episode_progress = _required_float(value, "episode_progress", line_number)
    if episode_progress > 1.0:
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} episode_progress must be within [0, 1]"
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
    value: dict[str, Any],
    *,
    line_number: int,
    trace_run_id: str,
    previous_stamp_s: float,
    persisted_policy_sample_count: int,
) -> tuple[float, float, str]:
    expected = {
        "schema_version": ORIN_EDGE_CONTROL_AUDIT_SCHEMA_VERSION,
        "record_type": "terminal",
        "mode": "control",
        "status": "terminal",
        "trajectory_controller_backend": "onnx_rl",
        "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
        "trace_run_id": trace_run_id,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ExperimentRunValidationError(
                f"Orin control audit line {line_number} {field} must be {expected_value}"
            )
    for field in _POLICY_SAMPLE_FIELDS:
        if field in value:
            raise ExperimentRunValidationError(
                f"Orin control audit line {line_number} terminal must not contain "
                f"policy sample field {field}"
            )
    stamp_s = _required_float(value, "runtime_monotonic_s", line_number)
    if stamp_s <= previous_stamp_s:
        raise ExperimentRunValidationError(
            "Orin control audit terminal runtime_monotonic_s must follow policy samples"
        )
    elapsed_s = _required_float(value, "elapsed_s", line_number)
    policy_counts = tuple(
        _required_int(value, field, line_number)
        for field in ("expected_policy_sample_count",
                      "accepted_policy_sample_count",
                      "dropped_policy_sample_count")
    )
    expected_policy_sample_count, accepted_policy_sample_count, dropped_policy_sample_count = policy_counts
    if (
        expected_policy_sample_count
        != accepted_policy_sample_count + dropped_policy_sample_count
    ):
        raise ExperimentRunValidationError(
            "Orin control audit terminal policy sample counters are inconsistent"
        )
    if dropped_policy_sample_count != 0:
        raise ExperimentRunValidationError(
            "Orin control audit terminal reports dropped policy samples"
        )
    if (
        expected_policy_sample_count != persisted_policy_sample_count
        or accepted_policy_sample_count != persisted_policy_sample_count
    ):
        raise ExperimentRunValidationError(
            "Orin control audit terminal policy sample count does not match "
            "persisted policy samples"
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
            f"Orin control audit line {line_number} terminal result must be "
            "completed, timeout, rejected or interrupted"
        )
    return stamp_s, elapsed_s, result_map[result]


def _point3(
    value: dict[str, Any], field: str, line_number: int
) -> tuple[float, float, float]:
    item = value.get(field)
    if not isinstance(item, list) or len(item) != 3:
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} {field} must contain three values"
        )
    point = tuple(_finite_float(axis, field, line_number) for axis in item)
    return point  # type: ignore[return-value]


def _validate_segment_envelope(
    value: dict[str, Any],
    *,
    line_number: int,
) -> None:
    expected_text = {
        "schema_version": ORIN_EDGE_CONTROL_AUDIT_SCHEMA_VERSION,
        "mode": "control",
        "trajectory_controller_backend": "onnx_rl",
    }
    for field, expected in expected_text.items():
        if value.get(field) != expected:
            raise ExperimentRunValidationError(
                f"Orin control audit line {line_number} {field} must be {expected}"
            )
    status = value.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} status must be non-empty text"
        )


def _required_int(value: dict[str, Any], field: str, line_number: int) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} {field} must be a non-negative int"
        )
    return item


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentRunValidationError(f"{field} must be non-empty text")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ExperimentRunValidationError(
            f"{field} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _required_float(value: dict[str, Any], field: str, line_number: int) -> float:
    return _finite_float(value.get(field), field, line_number, nonnegative=True)


def _finite_float(
    value: object,
    field: str,
    line_number: int,
    *,
    nonnegative: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise ExperimentRunValidationError(
            f"Orin control audit line {line_number} {field} must contain {qualifier} numbers"
        )
    return float(value)


def _atomic_write_jsonl(path: Path, records: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ExperimentRunValidationError(
            f"real trajectory trace output must be a regular file: {path}"
        )
    payload = "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
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
                f"real trajectory trace output already exists: {path}"
            ) from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

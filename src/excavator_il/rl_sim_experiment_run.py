"""Strict post-hoc Experiment Run evidence for RL simulation trajectory playback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
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


RL_CONTROL_TRACE_SCHEMA_VERSION = "excavator_rl_control_trace.v3"
RL_CONTROL_TRACE_SEMANTICS = "commanded_normalized_action"
_ACTION_ORDER = ("boom", "stick", "bucket", "swing")
_EVALUATION_SCOPES = frozenset({"training_internal", "held_out_experiment"})
_TRACE_RESULTS = frozenset(
    {"ACTIVE", "COMPLETED", "TIMEOUT", "REJECTED", "INTERRUPTED"}
)
_TRACE_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "sample_id",
        "stamp_s",
        "action_order",
        "action",
        "trace_semantics",
        "trajectory_suite_sha256",
        "bucket_tip_ros_m",
        "reference_waypoint_ros_m",
        "waypoint_index",
        "waypoint_distance_m",
        "episode_progress",
        "result",
    }
)
_TRACE_TERMINAL_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "stamp_s",
        "elapsed_s",
        "trace_semantics",
        "trajectory_suite_sha256",
        "result",
    }
)


@dataclass(frozen=True)
class RlControlTraceSample:
    sample_id: int
    stamp_s: float
    action_order: tuple[str, str, str, str]
    action: tuple[float, float, float, float]
    trace_semantics: str
    trajectory_suite_sha256: str
    bucket_tip_ros_m: tuple[float, float, float]
    reference_waypoint_ros_m: tuple[float, float, float]
    waypoint_index: int
    waypoint_distance_m: float
    episode_progress: float
    result: str


@dataclass(frozen=True)
class RlControlTraceTerminal:
    stamp_s: float
    elapsed_s: float
    trace_semantics: str
    trajectory_suite_sha256: str
    result: str


@dataclass(frozen=True)
class RlControlTrace:
    samples: tuple[RlControlTraceSample, ...]
    terminal: RlControlTraceTerminal


@dataclass(frozen=True)
class RlSimExperimentRunRequest:
    experiment_run_root: Path
    machine_profile_path: Path
    trajectory_suite_path: Path
    trajectory_controller_onnx_path: Path
    trace_path: Path
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
        object.__setattr__(
            self, "experiment_run_root", Path(self.experiment_run_root).expanduser()
        )
        for field_name in (
            "machine_profile_path",
            "trajectory_suite_path",
            "trajectory_controller_onnx_path",
            "trace_path",
        ):
            object.__setattr__(
                self,
                field_name,
                Path(getattr(self, field_name)).expanduser(),
            )
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
        for field_name in ("policy_id", "evaluation_scope", "task_variant", "operator_id", "material_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ExperimentRunValidationError(f"{field_name} must be non-empty text")
        for field_name in ("soil_reset_block_id", "dig_point_id", "run_id"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ExperimentRunValidationError(f"{field_name} must be non-empty text when provided")
        if self.evaluation_scope not in _EVALUATION_SCOPES:
            raise ExperimentRunValidationError(
                "evaluation_scope must be training_internal or held_out_experiment"
            )
        if not self.policy_id.startswith("onnx_rl:"):
            raise ExperimentRunValidationError("policy_id must start with onnx_rl:")


def load_machine_profile_action_order(path: str | Path) -> tuple[str, str, str, str]:
    value = _read_json_object(Path(path), label="machine profile")
    action_order = value.get("action_order")
    if action_order != list(_ACTION_ORDER):
        raise ExperimentRunValidationError(
            "machine profile action_order must be [boom, stick, bucket, swing]"
        )
    return _ACTION_ORDER


def load_rl_control_trace(path: str | Path) -> tuple[RlControlTraceSample, ...]:
    return load_rl_control_trace_document(path).samples


def load_rl_control_trace_document(path: str | Path) -> RlControlTrace:
    return load_rl_control_trace_snapshot(path)[0]


def load_rl_control_trace_snapshot(
    path: str | Path,
) -> tuple[RlControlTrace, str]:
    trace_path = Path(path).expanduser()
    if trace_path.is_symlink() or not trace_path.is_file():
        raise ExperimentRunValidationError(
            f"trajectory trace must be a regular file: {trace_path}"
        )
    try:
        payload = trace_path.read_bytes()
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ExperimentRunValidationError(f"cannot read trajectory trace {trace_path}: {exc}") from exc
    if not lines:
        raise ExperimentRunValidationError("trajectory trace must contain at least one sample")
    samples: list[RlControlTraceSample] = []
    seen_ids: set[int] = set()
    previous_sample_id = -1
    previous_stamp = -math.inf
    trajectory_suite_sha256: str | None = None
    terminal: RlControlTraceTerminal | None = None
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentRunValidationError(
                f"trajectory trace line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ExperimentRunValidationError(
                f"trajectory trace line {line_number} must be a JSON object"
            )
        if value.get("record_type") == "terminal":
            if terminal is not None or line_number != len(lines):
                raise ExperimentRunValidationError(
                    "trajectory trace must contain exactly one final terminal record"
                )
            if set(value) != _TRACE_TERMINAL_FIELDS:
                raise ExperimentRunValidationError(
                    "trajectory trace terminal fields are invalid; expected "
                    f"{sorted(_TRACE_TERMINAL_FIELDS)}"
                )
            if value["schema_version"] != RL_CONTROL_TRACE_SCHEMA_VERSION:
                raise ExperimentRunValidationError(
                    f"trajectory trace schema_version must be {RL_CONTROL_TRACE_SCHEMA_VERSION}"
                )
            if value["trace_semantics"] != RL_CONTROL_TRACE_SEMANTICS:
                raise ExperimentRunValidationError(
                    "trajectory trace semantics must be commanded_normalized_action"
                )
            terminal_suite_sha256 = _sha256(
                value["trajectory_suite_sha256"],
                "trajectory trace trajectory_suite_sha256",
            )
            if terminal_suite_sha256 != trajectory_suite_sha256:
                raise ExperimentRunValidationError(
                    "trajectory trace terminal suite binding must match policy samples"
                )
            terminal_stamp_s = _nonnegative_float(
                value["stamp_s"], "trajectory trace terminal stamp_s"
            )
            if terminal_stamp_s <= previous_stamp:
                raise ExperimentRunValidationError(
                    "trajectory trace terminal stamp_s must follow policy samples"
                )
            terminal_elapsed_s = _nonnegative_float(
                value["elapsed_s"], "trajectory trace terminal elapsed_s"
            )
            terminal_result = value["result"]
            if terminal_result not in _TRACE_RESULTS - {"ACTIVE"}:
                raise ExperimentRunValidationError(
                    "trajectory trace terminal result must be COMPLETED, TIMEOUT, "
                    "REJECTED or INTERRUPTED"
                )
            terminal = RlControlTraceTerminal(
                stamp_s=terminal_stamp_s,
                elapsed_s=terminal_elapsed_s,
                trace_semantics=RL_CONTROL_TRACE_SEMANTICS,
                trajectory_suite_sha256=terminal_suite_sha256,
                result=terminal_result,
            )
            continue
        if set(value) != _TRACE_SAMPLE_FIELDS:
            raise ExperimentRunValidationError(
                "trajectory trace sample fields are invalid; expected "
                f"{sorted(_TRACE_SAMPLE_FIELDS)}"
            )
        if value["record_type"] != "policy_sample":
            raise ExperimentRunValidationError(
                "trajectory trace record_type must be policy_sample or terminal"
            )
        if value["schema_version"] != RL_CONTROL_TRACE_SCHEMA_VERSION:
            raise ExperimentRunValidationError(
                f"trajectory trace schema_version must be {RL_CONTROL_TRACE_SCHEMA_VERSION}"
            )
        if value["trace_semantics"] != RL_CONTROL_TRACE_SEMANTICS:
            raise ExperimentRunValidationError(
                "trajectory trace semantics must be commanded_normalized_action"
            )
        sample_suite_sha256 = _sha256(
            value["trajectory_suite_sha256"],
            "trajectory trace trajectory_suite_sha256",
        )
        if trajectory_suite_sha256 is None:
            trajectory_suite_sha256 = sample_suite_sha256
        elif sample_suite_sha256 != trajectory_suite_sha256:
            raise ExperimentRunValidationError(
                "trajectory trace must bind exactly one trajectory_suite_sha256"
            )
        sample_id = value["sample_id"]
        if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
            raise ExperimentRunValidationError("trajectory trace sample_id must be a non-negative int")
        if sample_id != len(samples):
            raise ExperimentRunValidationError(
                "trajectory trace sample_id values must form a contiguous prefix "
                "starting at 0"
            )
        if sample_id in seen_ids:
            raise ExperimentRunValidationError("trajectory trace sample_id values must be unique")
        if sample_id <= previous_sample_id:
            raise ExperimentRunValidationError(
                "trajectory trace sample_id values must be strictly increasing"
            )
        stamp_s = value["stamp_s"]
        if (
            isinstance(stamp_s, bool)
            or not isinstance(stamp_s, (int, float))
            or not math.isfinite(float(stamp_s))
            or float(stamp_s) < 0.0
        ):
            raise ExperimentRunValidationError("trajectory trace stamp_s must be a finite non-negative number")
        stamp_s = float(stamp_s)
        if stamp_s <= previous_stamp:
            raise ExperimentRunValidationError("trajectory trace stamp_s must be strictly increasing")
        action_order = value["action_order"]
        if action_order != list(_ACTION_ORDER):
            raise ExperimentRunValidationError(
                "trajectory trace action_order must be [boom, stick, bucket, swing]"
            )
        action = value["action"]
        if not isinstance(action, list) or len(action) != 4:
            raise ExperimentRunValidationError("trajectory trace action must contain four values")
        action_values = tuple(_finite_float(item, "trajectory trace action") for item in action)
        if any(item < -1.0 or item > 1.0 for item in action_values):
            raise ExperimentRunValidationError(
                "trajectory trace commanded_normalized_action values must be within [-1, 1]"
            )
        bucket_tip_ros_m = _point3(
            value["bucket_tip_ros_m"], "trajectory trace bucket_tip_ros_m"
        )
        reference_waypoint_ros_m = _point3(
            value["reference_waypoint_ros_m"],
            "trajectory trace reference_waypoint_ros_m",
        )
        waypoint_index = value["waypoint_index"]
        if (
            isinstance(waypoint_index, bool)
            or not isinstance(waypoint_index, int)
            or waypoint_index < 0
        ):
            raise ExperimentRunValidationError(
                "trajectory trace waypoint_index must be a non-negative int"
            )
        waypoint_distance_m = _finite_float(
            value["waypoint_distance_m"], "trajectory trace waypoint_distance_m"
        )
        if waypoint_distance_m < 0.0:
            raise ExperimentRunValidationError(
                "trajectory trace waypoint_distance_m must be non-negative"
            )
        geometric_distance_m = math.dist(
            bucket_tip_ros_m,
            reference_waypoint_ros_m,
        )
        if not math.isclose(
            waypoint_distance_m,
            geometric_distance_m,
            rel_tol=1e-6,
            abs_tol=1e-3,
        ):
            raise ExperimentRunValidationError(
                "trajectory trace waypoint_distance_m must match bucket-tip to "
                "reference-waypoint distance within 1e-3 m"
            )
        episode_progress = _finite_float(
            value["episode_progress"], "trajectory trace episode_progress"
        )
        if episode_progress < 0.0 or episode_progress > 1.0:
            raise ExperimentRunValidationError(
                "trajectory trace episode_progress must be within [0, 1]"
            )
        result = value["result"]
        if result != "ACTIVE":
            raise ExperimentRunValidationError(
                "trajectory trace policy sample result must be ACTIVE"
            )
        samples.append(
            RlControlTraceSample(
                sample_id=sample_id,
                stamp_s=stamp_s,
                action_order=_ACTION_ORDER,
                action=action_values,  # type: ignore[arg-type]
                trace_semantics=RL_CONTROL_TRACE_SEMANTICS,
                trajectory_suite_sha256=sample_suite_sha256,
                bucket_tip_ros_m=bucket_tip_ros_m,
                reference_waypoint_ros_m=reference_waypoint_ros_m,
                waypoint_index=waypoint_index,
                waypoint_distance_m=waypoint_distance_m,
                episode_progress=episode_progress,
                result=result,
            )
        )
        seen_ids.add(sample_id)
        previous_sample_id = sample_id
        previous_stamp = stamp_s
    if not samples:
        raise ExperimentRunValidationError(
            "trajectory trace must contain at least one policy sample"
        )
    if terminal is None:
        raise ExperimentRunValidationError(
            "trajectory trace must contain exactly one final terminal record"
        )
    return (
        RlControlTrace(samples=tuple(samples), terminal=terminal),
        hashlib.sha256(payload).hexdigest(),
    )


def record_rl_sim_experiment_run(
    request: RlSimExperimentRunRequest,
) -> ExperimentRunSnapshot:
    if not isinstance(request, RlSimExperimentRunRequest):
        raise TypeError("request must be RlSimExperimentRunRequest")
    machine_profile_fingerprint = fingerprint_path(request.machine_profile_path)
    action_order = load_machine_profile_action_order(request.machine_profile_path)
    trajectory_suite, trajectory_suite_sha256 = load_trajectory_suite_snapshot(
        request.trajectory_suite_path
    )
    trace_document, trace_sha256 = load_rl_control_trace_snapshot(request.trace_path)
    trace = trace_document.samples
    if {sample.trajectory_suite_sha256 for sample in trace} != {
        trajectory_suite_sha256
    }:
        raise ExperimentRunValidationError(
            "trajectory trace trajectory_suite_sha256 does not match the "
            "trajectory suite artifact"
        )
    if set(sample.sample_id for sample in trace) - set(trajectory_suite["sample_ids"]):
        raise ExperimentRunValidationError(
            "trajectory trace contains sample_id values outside the trajectory suite"
        )
    onnx_fingerprint = fingerprint_path(request.trajectory_controller_onnx_path)
    if onnx_fingerprint.object_type != "file":
        raise ExperimentRunValidationError("trajectory_controller_onnx_path must be a file")
    run = ExperimentRun.create(
        request.experiment_run_root,
        run_id=request.run_id,
        run_kind="evaluation",
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
    trace_record = run.register_artifact(
        "trajectory_trace",
        request.trace_path,
        role="trajectory_trace",
        metadata={
            "sample_count": len(trace),
            "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
            "trajectory_suite_sha256": trajectory_suite_sha256,
            "terminal_result": trace_document.terminal.result,
            "terminal_stamp_s": trace_document.terminal.stamp_s,
            "terminal_elapsed_s": trace_document.terminal.elapsed_s,
        },
    )
    if trace_record["sha256"] != trace_sha256:
        raise ExperimentRunValidationError(
            "trajectory trace changed while recording evidence"
        )
    return run.finalize(
        "success",
        metrics={
            "evaluation_scope": request.evaluation_scope,
            "machine_profile_sha256": run.snapshot().start["machine_profile"]["sha256"],
            "action_order": list(action_order),
            "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
            "trajectory_controller_onnx_sha256": onnx_record["sha256"],
            "trajectory_suite_sha256": suite_record["sha256"],
            "trace_sha256": trace_record["sha256"],
        },
        summary="Strict RL simulation trajectory evidence retained for sim-real comparison.",
    )


def load_trajectory_suite(path: str | Path) -> dict[str, Any]:
    return load_trajectory_suite_snapshot(path)[0]


def load_trajectory_suite_snapshot(
    path: str | Path,
) -> tuple[dict[str, Any], str]:
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunValidationError(
            f"trajectory suite must be a regular file: {path}"
        )
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentRunValidationError(
            f"cannot read trajectory suite: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ExperimentRunValidationError("trajectory suite must be a JSON object")
    if set(value) != {"suite_id", "sample_period_s", "sample_ids"}:
        raise ExperimentRunValidationError(
            "trajectory suite must contain exactly suite_id, sample_period_s "
            "and sample_ids"
        )
    suite_id = value["suite_id"]
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise ExperimentRunValidationError("trajectory suite suite_id must be non-empty text")
    sample_period_s = _finite_float(
        value["sample_period_s"],
        "trajectory suite sample_period_s",
    )
    if sample_period_s != 0.1:
        raise ExperimentRunValidationError(
            "trajectory suite sample_period_s must be 0.1"
        )
    sample_ids = value["sample_ids"]
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ExperimentRunValidationError("trajectory suite sample_ids must be a non-empty array")
    seen: set[int] = set()
    for item in sample_ids:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ExperimentRunValidationError(
                "trajectory suite sample_ids must contain non-negative ints"
            )
        if item in seen:
            raise ExperimentRunValidationError("trajectory suite sample_ids must be unique")
        seen.add(item)
    if sample_ids != list(range(len(sample_ids))):
        raise ExperimentRunValidationError(
            "trajectory suite sample_ids must start at 0 and be contiguous"
        )
    return (
        {
            "suite_id": suite_id,
            "sample_period_s": sample_period_s,
            "sample_ids": tuple(sample_ids),
        },
        hashlib.sha256(payload).hexdigest(),
    )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunValidationError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentRunValidationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentRunValidationError(f"{label} must be a JSON object")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ExperimentRunValidationError(f"{label} must contain finite numeric values")
    return float(value)


def _nonnegative_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ExperimentRunValidationError(f"{label} must be non-negative")
    return result


def _point3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ExperimentRunValidationError(f"{label} must contain three values")
    result = tuple(_finite_float(item, label) for item in value)
    return result  # type: ignore[return-value]


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ExperimentRunValidationError(
            f"{label} must be a lowercase 64-character SHA-256 digest"
        )
    return value


__all__ = [
    "RL_CONTROL_TRACE_SCHEMA_VERSION",
    "RL_CONTROL_TRACE_SEMANTICS",
    "RlControlTraceSample",
    "RlControlTraceTerminal",
    "RlControlTrace",
    "RlSimExperimentRunRequest",
    "load_machine_profile_action_order",
    "load_rl_control_trace",
    "load_rl_control_trace_document",
    "load_rl_control_trace_snapshot",
    "load_trajectory_suite",
    "load_trajectory_suite_snapshot",
    "record_rl_sim_experiment_run",
]

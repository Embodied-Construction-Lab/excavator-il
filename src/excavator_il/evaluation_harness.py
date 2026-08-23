"""Unified, evidence-backed evaluation for one or more Experiment Runs.

The small ``ExperimentRunLoader`` Interface keeps evidence stores replaceable
without weakening the strict ``experiment_run.v1`` contract.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
import csv
import io
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from ._evaluation_evidence import (
    EvidenceEvaluationError,
    evaluate_artifacts,
    evaluate_reproducibility,
    evaluate_requirements,
    verify_snapshot_artifacts,
)
from .experiment_run import EXPERIMENT_RUN_KINDS


EVALUATION_SUMMARY_SCHEMA_VERSION = "evaluation_summary.v1"
EVALUATION_AGGREGATE_SCHEMA_VERSION = "evaluation_aggregate.v1"
_CSV_FIELDS = (
    "run_id", "run_kind", "run_state", "passed",
    "evaluation_scope", "method", "policy", "task_variant",
    "soil_reset_block_id", "dig_point_id", "dirty_repository_count",
    "task_success", "requested_cycles", "completed_cycles",
    "cycle_success_rate", "episode_count", "training_frame_count",
    "camera_front_rate_hz_p50",
    "camera_front_age_ms_p95",
    "camera_front_sequence_gap_count",
    "camera_front_queue_drop_count",
    "camera_dump_rate_hz_p50",
    "camera_dump_age_ms_p95",
    "camera_dump_sequence_gap_count",
    "camera_dump_queue_drop_count",
    "act_step_count",
    "inference_rate_hz",
    "max_state_to_decision_ms",
    "deadline_dropped_state_count",
    "rl_to_act_handoff_p50_ms",
    "rl_to_act_handoff_p95_ms",
    "rl_to_act_handoff_max_ms",
    "act_to_rl_handoff_p50_ms",
    "act_to_rl_handoff_p95_ms",
    "act_to_rl_handoff_max_ms",
    "intervention_count",
    "runtime_abort_count",
    "post_terminal_nonzero_count",
    "payload_mass_kg",
    "fill_ratio",
    "spillage_mass_kg",
    "spillage_ratio",
    "phase_duration_s_json",
    "unavailable_reasons_json",
    "failure_reasons_json",
)


class EvaluationError(ValueError):
    """Experiment evidence cannot support a trustworthy evaluation."""


class ExperimentRunLoader(Protocol):
    """Load one finalized strict ``experiment_run.v1`` snapshot."""
    def load(self, run_path: str | Path) -> object: ...


class FileExperimentRunLoader:
    """Adapter for the repository's authoritative Experiment Run loader."""
    def load(self, run_path: str | Path) -> object:
        try:
            from .experiment_run import load_experiment_run
        except ImportError as exc:  # pragma: no cover - installation corruption
            raise EvaluationError("experiment_run loader is unavailable") from exc
        return load_experiment_run(Path(run_path))


def evaluate_experiment_run(
    run_path: str | Path,
    *,
    loader: ExperimentRunLoader | None = None,
) -> dict[str, object]:
    """Evaluate one finalized Experiment Run through its public evidence API."""

    adapter = loader or FileExperimentRunLoader()
    snapshot = adapter.load(run_path)
    start = _mapping(_attribute(snapshot, "start"), "snapshot.start")
    run_id = _text(_attribute(snapshot, "run_id"), "snapshot.run_id")
    if start.get("schema_version") != "experiment_run.v1":
        raise EvaluationError("snapshot.start schema_version must be experiment_run.v1")
    if start.get("run_id") != run_id:
        raise EvaluationError("snapshot run_id does not match snapshot.start")
    state = _text(_attribute(snapshot, "state"), "snapshot.state")
    if state not in {"success", "failure"}:
        raise EvaluationError("Experiment Run must be finalized before evaluation")
    final_value = _attribute(snapshot, "final")
    if final_value is None:
        raise EvaluationError("Experiment Run must be finalized before evaluation")
    final = _mapping(final_value, "snapshot.final")
    if _attribute(snapshot, "manifest") is None:
        raise EvaluationError("finalized Experiment Run must include a manifest")
    if _text(final.get("status"), "final.status") != state:
        raise EvaluationError("snapshot.state and final.status must match")
    evaluation_scope = _evaluation_scope(final)
    events = _records(_attribute(snapshot, "events"), "snapshot.events")
    artifacts = _records(_attribute(snapshot, "artifacts"), "snapshot.artifacts")
    snapshot_run_dir_value = _attribute(snapshot, "run_dir")
    snapshot_run_dir = (
        None
        if snapshot_run_dir_value is None
        else Path(snapshot_run_dir_value)
    )
    run_kind = _text(start.get("run_kind"), "start.run_kind")
    if run_kind not in EXPERIMENT_RUN_KINDS:
        raise EvaluationError(
            f"start.run_kind must be one of {sorted(EXPERIMENT_RUN_KINDS)}"
        )
    try:
        artifact_failure = verify_snapshot_artifacts(snapshot)
    except EvidenceEvaluationError as exc:
        raise EvaluationError(str(exc)) from exc
    trusted_artifacts = artifacts if artifact_failure is None else ()

    grouping = _grouping(start)
    task = _task_metrics(state, final)
    timing = _timing_metrics(events)
    measurements = _measurement_metrics(final)
    try:
        artifact_evaluation = evaluate_artifacts(
            trusted_artifacts,
            run_dir=snapshot_run_dir,
        )
        reproducibility, reproducibility_failures = evaluate_reproducibility(
            start,
            evaluation_scope=evaluation_scope,
        )
        requirement_evaluation = evaluate_requirements(
            start.get("evidence_requirements", {}),
            artifacts,
            artifact_evaluation.evaluated_role_counts,
        )
    except EvidenceEvaluationError as exc:
        raise EvaluationError(str(exc)) from exc
    safety, safety_failures = _safety_metrics(events)
    failure_reasons = (
        ([] if state == "success" else ["Experiment Run did not succeed"])
        + ([] if artifact_failure is None else [artifact_failure])
        + list(requirement_evaluation.failure_reasons)
        + list(artifact_evaluation.failure_reasons)
        + list(reproducibility_failures)
        + safety_failures
    )

    return {
        "schema_version": EVALUATION_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "run_state": state,
        "evaluation_scope": evaluation_scope,
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "grouping": grouping,
        "reproducibility": reproducibility,
        "task": task,
        "timing": timing,
        "data": artifact_evaluation.data,
        "runtime": artifact_evaluation.runtime,
        "handoffs": artifact_evaluation.handoffs,
        "safety": safety,
        "measurements": measurements,
        "evidence": {
            "artifact_count": len(artifacts),
            "artifacts_verified": artifact_failure is None,
            "evaluated_roles": list(artifact_evaluation.evaluated_roles),
            "evaluated_role_counts": dict(
                requirement_evaluation.evaluated_role_counts
            ),
            "missing_required_roles": list(
                requirement_evaluation.missing_required_roles
            ),
            "unevaluated_required_roles": list(
                requirement_evaluation.unevaluated_required_roles
            ),
            "integrity_verified": {
                "artifact_set": artifact_failure is None,
                "roles": list(artifact_evaluation.integrity_verified_roles),
                "role_counts": dict(
                    artifact_evaluation.integrity_verified_role_counts
                ),
            },
        },
    }


def evaluate_experiment_runs(
    run_paths: Iterable[str | Path],
    *,
    loader: ExperimentRunLoader | None = None,
    aggregate_mode: str = "homogeneous",
) -> dict[str, object]:
    """Evaluate and deterministically aggregate finalized Experiment Runs."""

    paths = tuple(Path(path) for path in run_paths)
    if not paths:
        raise EvaluationError("at least one Experiment Run path is required")
    summaries = sorted(
        (evaluate_experiment_run(path, loader=loader) for path in paths),
        key=lambda summary: str(summary["run_id"]),
    )
    run_ids = [str(summary["run_id"]) for summary in summaries]
    if len(set(run_ids)) != len(run_ids):
        raise EvaluationError("Experiment Run ids must be unique in one aggregate")
    _validate_aggregate_mode(summaries, aggregate_mode)

    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for summary in summaries:
        grouping = _mapping(summary["grouping"], "evaluation grouping")
        key = (
            summary["evaluation_scope"],
            grouping["method"],
            grouping["policy"],
            grouping["task_variant"],
            grouping["soil_reset_block_id"],
            grouping["dig_point_id"],
        )
        grouped[key].append(summary)

    groups = [
        _aggregate_group(key, grouped[key])
        for key in sorted(
            grouped,
            key=lambda item: tuple(
                "" if value is None else str(value) for value in item
            ),
        )
    ]
    return {
        "schema_version": EVALUATION_AGGREGATE_SCHEMA_VERSION,
        "aggregate_mode": aggregate_mode,
        "scope_policy": (
            "one evaluation_scope per aggregate; training_internal and "
            "held_out_experiment are never mixed"
        ),
        "run_count": len(summaries),
        "passed_run_count": sum(
            1 for summary in summaries if summary["passed"] is True
        ),
        "failed_run_count": sum(
            1 for summary in summaries if summary["passed"] is not True
        ),
        "groups": groups,
        "runs": summaries,
    }


def write_evaluation_outputs(
    aggregate: Mapping[str, object],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Write byte-deterministic JSON and CSV representations of an aggregate."""

    if aggregate.get("schema_version") != EVALUATION_AGGREGATE_SCHEMA_VERSION:
        raise EvaluationError(
            f"aggregate schema_version must be {EVALUATION_AGGREGATE_SCHEMA_VERSION}"
        )
    runs = aggregate.get("runs")
    if not isinstance(runs, list):
        raise EvaluationError("aggregate.runs must be an array")
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_bytes = (
        json.dumps(
            aggregate,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for summary in sorted(runs, key=lambda item: str(item["run_id"])):
        writer.writerow(_flatten_summary(_mapping(summary, "aggregate run")))
    _atomic_write_bytes(json_target, json_bytes)
    _atomic_write_bytes(csv_target, buffer.getvalue().encode("utf-8"))
    return json_target, csv_target


def _evaluation_scope(final: Mapping[str, object]) -> str:
    metrics = _mapping(final.get("metrics", {}), "final.metrics")
    value = metrics.get("evaluation_scope")
    if value is None:
        raise EvaluationError("final.metrics.evaluation_scope is required")
    scope = _text(value, "final.metrics.evaluation_scope")
    if scope not in {"training_internal", "held_out_experiment"}:
        raise EvaluationError(
            "evaluation_scope must be training_internal or held_out_experiment"
        )
    return scope


def _aggregate_group(
    key: tuple[object, ...], summaries: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    scope, method, policy, task_variant, soil_block, dig_point = key
    completed = [
        _mapping(summary["task"], "summary.task").get("completed_cycles")
        for summary in summaries
    ]
    completed_values = [int(value) for value in completed if isinstance(value, int)]
    return {
        "evaluation_scope": scope,
        "method": method,
        "policy": policy,
        "task_variant": task_variant,
        "soil_reset_block_id": soil_block,
        "dig_point_id": dig_point,
        "run_count": len(summaries),
        "passed_run_count": sum(
            1 for summary in summaries if summary["passed"] is True
        ),
        "task_success_rate": sum(
            1
            for summary in summaries
            if _mapping(summary["task"], "summary.task").get("success") is True
        )
        / len(summaries),
        "completed_cycles": sum(completed_values) if completed_values else None,
        "episode_count": sum(
            int(_mapping(summary["data"], "summary.data")["episode_count"])
            for summary in summaries
        ),
        "training_frame_count": sum(
            int(_mapping(summary["data"], "summary.data")["training_frame_count"])
            for summary in summaries
        ),
    }


def _flatten_summary(summary: Mapping[str, object]) -> dict[str, object]:
    grouping = _mapping(summary["grouping"], "summary.grouping")
    task = _mapping(summary["task"], "summary.task")
    data = _mapping(summary["data"], "summary.data")
    cameras = _mapping(data.get("cameras", {}), "summary.data.cameras")
    runtime = _mapping(summary["runtime"], "summary.runtime")
    act = _mapping(runtime.get("act", {}), "summary.runtime.act")
    inference = _mapping(runtime.get("inference", {}), "summary.runtime.inference")
    deadline = _mapping(runtime.get("deadline", {}), "summary.runtime.deadline")
    handoffs = _mapping(summary["handoffs"], "summary.handoffs")
    directions = _mapping(handoffs.get("directions", {}), "summary.handoffs.directions")
    safety = _mapping(summary["safety"], "summary.safety")
    measurements = _mapping(summary["measurements"], "summary.measurements")
    reproducibility = _mapping(
        summary["reproducibility"], "summary.reproducibility"
    )
    timing = _mapping(summary["timing"], "summary.timing")
    row: dict[str, object] = {
        "run_id": summary["run_id"],
        "run_kind": summary["run_kind"],
        "run_state": summary["run_state"],
        "passed": summary["passed"],
        "evaluation_scope": summary["evaluation_scope"],
        "method": grouping["method"],
        "policy": grouping["policy"],
        "task_variant": grouping["task_variant"],
        "soil_reset_block_id": grouping["soil_reset_block_id"],
        "dig_point_id": grouping["dig_point_id"],
        "dirty_repository_count": reproducibility["dirty_repository_count"],
        "task_success": task["success"],
        "requested_cycles": task["requested_cycles"],
        "completed_cycles": task["completed_cycles"],
        "cycle_success_rate": task["cycle_success_rate"],
        "episode_count": data["episode_count"],
        "training_frame_count": data["training_frame_count"],
        "act_step_count": act.get("step_count"),
        "inference_rate_hz": inference.get("estimated_rate_hz"),
        "max_state_to_decision_ms": inference.get("max_state_to_decision_ms"),
        "deadline_dropped_state_count": deadline.get("dropped_state_count"),
        "intervention_count": safety["intervention_count"],
        "runtime_abort_count": safety["runtime_abort_count"],
        "post_terminal_nonzero_count": safety["post_terminal_nonzero_count"],
        "phase_duration_s_json": json.dumps(
            timing.get("phases", {}), sort_keys=True, separators=(",", ":")
        ),
        "failure_reasons_json": json.dumps(
            summary["failure_reasons"], sort_keys=True, separators=(",", ":")
        ),
        "unavailable_reasons_json": json.dumps(
            _collect_unavailable_reasons(summary),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for camera in ("camera_front", "camera_dump"):
        metrics = _mapping(cameras.get(camera, {}), f"summary.data.cameras.{camera}")
        rate = _mapping(metrics.get("rate_hz", {}), f"{camera}.rate_hz")
        age = _mapping(metrics.get("age_ms", {}), f"{camera}.age_ms")
        row[f"{camera}_rate_hz_p50"] = rate.get("p50")
        row[f"{camera}_age_ms_p95"] = age.get("p95")
        row[f"{camera}_sequence_gap_count"] = metrics.get(
            "sequence_gap_count"
        )
        row[f"{camera}_queue_drop_count"] = metrics.get("queue_drop_count")
    direction_fields = {
        "rl_to_act": "rl_follow/velocity_reference->act_dig/manual_action",
        "act_to_rl": "act_dig/manual_action->rl_follow/velocity_reference",
    }
    for prefix, direction in direction_fields.items():
        metrics = _mapping(directions.get(direction, {}), f"handoff.{direction}")
        for percentile in ("p50", "p95", "max"):
            row[f"{prefix}_handoff_{percentile}_ms"] = metrics.get(
                f"{percentile}_ms"
            )
    for name in (
        "payload_mass_kg",
        "fill_ratio",
        "spillage_mass_kg",
        "spillage_ratio",
    ):
        row[name] = _mapping(measurements[name], f"measurements.{name}").get("value")
    return {field: row.get(field) for field in _CSV_FIELDS}


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _collect_unavailable_reasons(
    value: object,
    *,
    path: str = "",
) -> dict[str, str]:
    reasons: dict[str, str] = {}
    if isinstance(value, Mapping):
        unavailable_reason = value.get("unavailable_reason")
        if isinstance(unavailable_reason, str) and unavailable_reason:
            reasons[path or "$summary"] = unavailable_reason
        named_reasons = value.get("unavailable_reasons")
        if isinstance(named_reasons, Mapping):
            for name in sorted(named_reasons):
                reason = named_reasons[name]
                if isinstance(reason, str) and reason:
                    key = ".".join(filter(None, (path, str(name))))
                    reasons[key] = reason
        for name in sorted(value, key=str):
            if name in {"unavailable_reason", "unavailable_reasons"}:
                continue
            nested_path = ".".join(filter(None, (path, str(name))))
            reasons.update(
                _collect_unavailable_reasons(value[name], path=nested_path)
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            nested_path = ".".join(filter(None, (path, str(index))))
            reasons.update(_collect_unavailable_reasons(item, path=nested_path))
    return reasons


def _validate_aggregate_mode(
    summaries: Sequence[Mapping[str, object]], mode: str
) -> None:
    allowed_modes = {"homogeneous", "live_task", "collection_dataset"}
    if mode not in allowed_modes:
        raise EvaluationError(
            "aggregate_mode must be homogeneous, live_task, or collection_dataset"
        )
    scopes = {summary["evaluation_scope"] for summary in summaries}
    if len(scopes) != 1:
        raise EvaluationError(f"{mode} aggregate cannot mix evaluation scopes")
    if mode == "homogeneous":
        kinds = {summary["run_kind"] for summary in summaries}
        if len(kinds) != 1:
            raise EvaluationError(
                "homogeneous aggregate cannot mix Experiment Run kinds"
            )
        return
    if mode == "live_task":
        if any(
            summary["run_kind"] != "hybrid_live"
            or summary["evaluation_scope"] != "held_out_experiment"
            for summary in summaries
        ):
            raise EvaluationError(
                "live_task aggregate accepts only held-out hybrid_live runs"
            )
        return
    if mode == "collection_dataset":
        if any(summary["run_kind"] != "collection_episode" for summary in summaries):
            raise EvaluationError(
                "collection_dataset aggregate accepts only collection_episode runs"
            )
        return


def _attribute(value: object, name: str) -> object:
    if not hasattr(value, name):
        raise EvaluationError(f"Experiment Run snapshot is missing {name}")
    return getattr(value, name)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{field} must be an object")
    return value


def _records(value: object, field: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (tuple, list)):
        raise EvaluationError(f"{field} must be an array")
    return tuple(_mapping(item, field) for item in value)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{field} must be non-empty text")
    return value


def _grouping(start: Mapping[str, object]) -> dict[str, str | None]:
    context = _mapping(start.get("task_context"), "start.task_context")
    policy_ids = _mapping(start.get("policy_ids"), "start.policy_ids")
    canonical_policy: list[str] = []
    canonical_method: list[str] = []
    for role in sorted(policy_ids):
        policy_id = _text(policy_ids[role], f"policy_ids.{role}")
        canonical_policy.append(
            f"{_text(role, 'policy role')}={policy_id}"
        )
        canonical_method.append(f"{role}={policy_id.partition(':')[0]}")
    return {
        "method": "|".join(canonical_method) or "no_policy",
        "policy": "|".join(canonical_policy),
        "task_variant": _text(
            context.get("task_variant"), "task_context.task_variant"
        ),
        "soil_reset_block_id": _optional_text(
            context.get("soil_reset_block_id"),
            "task_context.soil_reset_block_id",
        ),
        "dig_point_id": _optional_text(
            context.get("dig_point_id"), "task_context.dig_point_id"
        ),
    }


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _task_metrics(
    state: str, final: Mapping[str, object]
) -> dict[str, object]:
    metrics = _mapping(final.get("metrics", {}), "final.metrics")
    requested = _optional_nonnegative_int(
        metrics.get("requested_cycles"), "requested_cycles"
    )
    completed = _optional_nonnegative_int(
        metrics.get("completed_cycles"), "completed_cycles"
    )
    if requested == 0:
        raise EvaluationError("requested_cycles must be positive when present")
    if requested is not None and completed is not None and completed > requested:
        raise EvaluationError("completed_cycles cannot exceed requested_cycles")
    rate = (
        None
        if requested is None or completed is None
        else completed / requested
    )
    result: dict[str, object] = {
        "success": state == "success",
        "requested_cycles": requested,
        "completed_cycles": completed,
        "cycle_success_rate": rate,
    }
    if requested is None or completed is None:
        result["unavailable_reasons"] = {
            "requested_cycles": (
                "final metrics did not record requested_cycles"
                if requested is None
                else None
            ),
            "completed_cycles": (
                "final metrics did not record completed_cycles"
                if completed is None
                else None
            ),
            "cycle_success_rate": "both cycle counts are required",
        }
    return result


def _optional_nonnegative_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{field} must be a non-negative integer")
    return value


def _timing_metrics(events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    phases: dict[str, list[float]] = defaultdict(list)
    cycles: list[float] = []
    phase_starts: dict[tuple[int | None, str], int] = {}
    cycle_starts: dict[int, int] = {}
    previous_sequence = -1
    previous_stamp = -1
    for event in events:
        if event.get("schema_version") != "experiment_run_event.v1":
            raise EvaluationError(
                "event schema_version must be experiment_run_event.v1"
            )
        sequence = _nonnegative_int(event.get("sequence"), "event.sequence")
        stamp = _nonnegative_int(event.get("monotonic_ns"), "event.monotonic_ns")
        if sequence != previous_sequence + 1:
            raise EvaluationError("event sequences must be contiguous from zero")
        if stamp < previous_stamp:
            raise EvaluationError("event monotonic_ns must not regress")
        previous_sequence = sequence
        previous_stamp = stamp
        event_type = _text(event.get("event_type"), "event.event_type")
        payload = _mapping(event.get("payload"), "event.payload")
        cycle_index = _optional_nonnegative_int(
            payload.get("cycle_index"), "cycle_index"
        )
        if event_type == "cycle_started":
            if cycle_index is None or cycle_index in cycle_starts:
                raise EvaluationError("cycle_started requires a unique cycle_index")
            cycle_starts[cycle_index] = stamp
        elif event_type == "cycle_completed":
            if cycle_index is None or cycle_index not in cycle_starts:
                raise EvaluationError("cycle_completed has no matching cycle_started")
            cycles.append((stamp - cycle_starts.pop(cycle_index)) / 1_000_000_000.0)
        elif event_type in {"phase_started", "phase_completed"}:
            phase = _text(payload.get("phase"), "event.payload.phase")
            key = (cycle_index, phase)
            if event_type == "phase_started":
                if key in phase_starts:
                    raise EvaluationError(
                        "phase_started must have a unique active phase"
                    )
                phase_starts[key] = stamp
            else:
                if key not in phase_starts and cycle_index is None:
                    candidates = [
                        active_key
                        for active_key in phase_starts
                        if active_key[1] == phase
                    ]
                    if len(candidates) == 1:
                        key = candidates[0]
                if key not in phase_starts:
                    raise EvaluationError(
                        "phase_completed has no matching phase_started"
                    )
                phases[phase].append(
                    (stamp - phase_starts.pop(key)) / 1_000_000_000.0
                )
    if phase_starts or cycle_starts:
        raise EvaluationError("timing evidence contains an unfinished phase or cycle")
    return {
        "cycles": _duration_summary(cycles),
        "phases": {
            phase: _duration_summary(values)
            for phase, values in sorted(phases.items())
        },
    }


def _nonnegative_int(value: object, field: str) -> int:
    result = _optional_nonnegative_int(value, field)
    if result is None:
        raise EvaluationError(f"{field} is required")
    return result


def _duration_summary(values: Sequence[float]) -> dict[str, object]:
    ordered = sorted(values)
    result: dict[str, object] = {
        "count": len(ordered),
        "duration_s": {
            "p50": _nearest_rank(ordered, 0.50),
            "p95": _nearest_rank(ordered, 0.95),
            "max": ordered[-1] if ordered else None,
            "total": sum(ordered) if ordered else None,
        },
    }
    if not ordered:
        result["unavailable_reason"] = "no completed timing interval was recorded"
    return result


def _nearest_rank(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return values[math.ceil(len(values) * probability) - 1]


def _measurement_metrics(final: Mapping[str, object]) -> dict[str, object]:
    metrics = _mapping(final.get("metrics", {}), "final.metrics")
    external = _mapping(
        metrics.get("external_measurements", {}), "external_measurements"
    )
    specifications = {
        "payload_mass_kg": "kg",
        "fill_ratio": "ratio",
        "spillage_mass_kg": "kg",
        "spillage_ratio": "ratio",
    }
    return {
        name: _measurement(external.get(name), name=name, expected_unit=unit)
        for name, unit in specifications.items()
    }


def _measurement(value: object, *, name: str, expected_unit: str) -> dict[str, object]:
    if value is None:
        return {
            "value": None,
            "unit": expected_unit,
            "method": None,
            "source": None,
            "unavailable_reason": "not explicitly measured for this run",
        }
    item = _mapping(value, f"external_measurements.{name}")
    number = item.get("value")
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise EvaluationError(f"external_measurements.{name}.value must be numeric")
    number = float(number)
    if not math.isfinite(number) or number < 0:
        raise EvaluationError(
            f"external_measurements.{name}.value must be finite and non-negative"
        )
    if expected_unit == "ratio" and number > 1:
        raise EvaluationError(
            f"external_measurements.{name}.value must be at most 1"
        )
    if item.get("unit") != expected_unit:
        raise EvaluationError(
            f"external_measurements.{name}.unit must be {expected_unit}"
        )
    return {
        "value": number,
        "unit": expected_unit,
        "method": _text(item.get("method"), f"external_measurements.{name}.method"),
        "source": _text(item.get("source"), f"external_measurements.{name}.source"),
        "unavailable_reason": None,
    }


def _event_count(events: Sequence[Mapping[str, object]], event_type: str) -> int:
    return sum(1 for event in events if event.get("event_type") == event_type)


def _safety_metrics(
    events: Sequence[Mapping[str, object]],
) -> tuple[dict[str, int], list[str]]:
    intervention_count = _event_count(events, "operator_intervention")
    runtime_abort_count = _event_count(events, "runtime_abort")
    post_terminal_nonzero_count = _event_count(
        events, "post_terminal_nonzero_command"
    )
    failures: list[str] = []
    if runtime_abort_count:
        failures.append(f"runtime abort event count is {runtime_abort_count}")
    if post_terminal_nonzero_count:
        failures.append(
            "post-terminal nonzero command event count is "
            f"{post_terminal_nonzero_count}"
        )
    return (
        {
            "intervention_count": intervention_count,
            "runtime_abort_count": runtime_abort_count,
            "post_terminal_nonzero_count": post_terminal_nonzero_count,
        },
        failures,
    )

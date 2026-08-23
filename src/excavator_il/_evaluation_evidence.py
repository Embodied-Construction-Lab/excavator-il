"""Strict adapters from registered artifacts to evaluation metrics.

This private module is the only place where the evaluation harness knows the
shape of Episode, ACT Runtime, and resident handoff evidence.  It delegates
semantic validation to their authoritative analyzers and only aggregates the
validated reports.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .act_runtime_log import inspect_act_runtime_log
from ._experiment_run_types import EXPERIMENT_ARTIFACT_SCHEMA_VERSION
from .raw_episode import EpisodeValidationError, validate_episode
from .resident_handoff_metrics import HandoffMetricsError, analyze_handoff_logs


class EvidenceEvaluationError(ValueError):
    """Registered artifact metadata cannot be interpreted safely."""


@dataclass(frozen=True)
class ArtifactEvaluation:
    data: Mapping[str, object]
    runtime: Mapping[str, object]
    handoffs: Mapping[str, object]
    evaluated_roles: tuple[str, ...]
    evaluated_role_counts: Mapping[str, int]
    integrity_verified_roles: tuple[str, ...]
    integrity_verified_role_counts: Mapping[str, int]
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceRequirementEvaluation:
    failure_reasons: tuple[str, ...]
    missing_required_roles: tuple[str, ...]
    unevaluated_required_roles: tuple[str, ...]
    evaluated_role_counts: Mapping[str, int]


_ANALYZER_EVALUATED_ROLES = frozenset(
    {"raw_episode", "runtime_log", "mission_log"}
)
_INTEGRITY_ONLY_EVALUATED_ROLES = frozenset(
    {
        "act_deployment_manifest",
        "act_policy_checkpoint",
        "quality_report",
        "rl_onnx_model",
    }
)
_CANONICALLY_EVALUATED_ROLES = (
    _ANALYZER_EVALUATED_ROLES | _INTEGRITY_ONLY_EVALUATED_ROLES
)


def evaluate_artifacts(
    artifacts: Sequence[Mapping[str, object]],
    *,
    run_dir: Path | None,
) -> ArtifactEvaluation:
    """Evaluate trusted artifacts by analyzer or explicit integrity-only role."""

    data, data_roles, data_failures = _evaluate_episodes(artifacts, run_dir)
    runtime, runtime_roles, runtime_failures = _evaluate_act_runtime(
        artifacts, run_dir
    )
    handoffs, handoff_roles, handoff_failures = _evaluate_handoffs(
        artifacts, run_dir
    )
    integrity_verified_role_counts = Counter(
        artifact.get("role")
        for artifact in artifacts
        if artifact.get("role") in _INTEGRITY_ONLY_EVALUATED_ROLES
    )
    evaluated_role_counts = Counter(data_roles + runtime_roles + handoff_roles)
    evaluated_role_counts.update(integrity_verified_role_counts)
    return ArtifactEvaluation(
        data=data,
        runtime=runtime,
        handoffs=handoffs,
        evaluated_roles=tuple(sorted(evaluated_role_counts)),
        evaluated_role_counts={
            role: evaluated_role_counts[role]
            for role in sorted(evaluated_role_counts)
        },
        integrity_verified_roles=tuple(sorted(integrity_verified_role_counts)),
        integrity_verified_role_counts={
            role: integrity_verified_role_counts[role]
            for role in sorted(integrity_verified_role_counts)
        },
        failure_reasons=tuple(
            data_failures + runtime_failures + handoff_failures
        ),
    )


def evaluate_requirements(
    requirements_value: object,
    artifacts: Sequence[Mapping[str, object]],
    evaluated_role_counts: Mapping[str, int],
) -> EvidenceRequirementEvaluation:
    """Require registered canonical evidence with its declared validation mode."""

    observed: dict[str, int] = defaultdict(int)
    for artifact in artifacts:
        if artifact.get("schema_version") != EXPERIMENT_ARTIFACT_SCHEMA_VERSION:
            raise EvidenceEvaluationError(
                "artifact schema_version must be "
                f"{EXPERIMENT_ARTIFACT_SCHEMA_VERSION}"
            )
        observed[_text(artifact.get("role"), "artifact.role")] += 1

    requirements = _mapping(
        requirements_value, "start.evidence_requirements"
    )
    specifications = tuple(
        _evidence_requirement(requirements, raw_role)
        for raw_role in sorted(requirements)
    )
    failures: list[str] = []
    missing: list[str] = []
    unevaluated: list[str] = []
    for role, required, minimum_count in specifications:
        registered_count = observed.get(role, 0)
        if not required or registered_count >= minimum_count:
            continue
        missing.append(role)
        failures.append(
            f"required artifact role {role} expected at least "
            f"{minimum_count} record(s); observed {registered_count}"
        )

    missing_set = frozenset(missing)
    for role, required, minimum_count in specifications:
        if not required or role in missing_set:
            continue
        if role not in _CANONICALLY_EVALUATED_ROLES:
            unevaluated.append(role)
            failures.append(
                f"required artifact role {role} is not supported by the "
                "canonical evaluation harness"
            )
            continue
        evaluated_count = evaluated_role_counts.get(role, 0)
        if evaluated_count >= minimum_count:
            continue
        unevaluated.append(role)
        failures.append(
            f"required artifact role {role} expected at least "
            f"{minimum_count} canonically evaluated record(s); observed "
            f"{evaluated_count}"
        )

    return EvidenceRequirementEvaluation(
        failure_reasons=tuple(failures),
        missing_required_roles=tuple(missing),
        unevaluated_required_roles=tuple(unevaluated),
        evaluated_role_counts={
            role: evaluated_role_counts.get(role, 0)
            for role in sorted(_CANONICALLY_EVALUATED_ROLES)
        },
    )


def evaluate_reproducibility(
    start: Mapping[str, object],
    *,
    evaluation_scope: str,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Report source-control state and reject dirty held-out evidence."""

    repositories = _mapping(start.get("repositories"), "start.repositories")
    dirty_repositories: list[str] = []
    for raw_name in sorted(repositories):
        name = _text(raw_name, "repository name")
        record = _mapping(repositories[raw_name], f"repositories.{name}")
        dirty = record.get("dirty")
        if not isinstance(dirty, bool):
            raise EvidenceEvaluationError(
                f"repositories.{name}.dirty must be boolean"
            )
        if dirty:
            dirty_repositories.append(name)
    failures: tuple[str, ...] = ()
    if evaluation_scope == "held_out_experiment" and dirty_repositories:
        failures = (
            "held_out_experiment is not reproducible because repositories "
            f"are dirty: {', '.join(dirty_repositories)}",
        )
    return (
        {
            "repository_count": len(repositories),
            "dirty_repository_count": len(dirty_repositories),
            "dirty_repositories": dirty_repositories,
            "source_control_clean": not dirty_repositories,
        },
        failures,
    )


def verify_snapshot_artifacts(snapshot: object) -> str | None:
    """Verify registered bytes through the Experiment Run public interface."""

    verifier = getattr(snapshot, "verify_artifacts", None)
    if not callable(verifier):
        raise EvidenceEvaluationError(
            "snapshot.verify_artifacts must be callable"
        )
    try:
        verifier()
    except Exception as exc:
        return f"artifact integrity verification failed: {exc}"
    return None


def _evidence_requirement(
    requirements: Mapping[str, object],
    raw_role: object,
) -> tuple[str, bool, int]:
    role = _text(raw_role, "evidence requirement role")
    specification = _mapping(
        requirements[role], f"evidence_requirements.{role}"
    )
    required = specification.get("required")
    if not isinstance(required, bool):
        raise EvidenceEvaluationError(
            f"evidence_requirements.{role}.required must be boolean"
        )
    minimum_count = _nonnegative_int(
        specification.get("min_count"),
        f"evidence_requirements.{role}.min_count",
    )
    if required and minimum_count < 1:
        raise EvidenceEvaluationError(
            f"evidence_requirements.{role}.min_count must be positive when required"
        )
    return role, required, minimum_count


def _evaluate_episodes(
    artifacts: Sequence[Mapping[str, object]],
    run_dir: Path | None,
) -> tuple[dict[str, object], list[str], list[str]]:
    episodes = tuple(
        artifact for artifact in artifacts if artifact.get("role") == "raw_episode"
    )
    training_frames = 0
    valid_episodes = 0
    observations: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    failures: list[str] = []
    for artifact in episodes:
        path = _artifact_snapshot_path(artifact, run_dir)
        try:
            report = validate_episode(path)
            quality = _read_json(path / "quality_report.json", optional=True)
            valid_episodes += 1
            training_frames += report.step_count
            if quality:
                for camera, metrics in _camera_quality(quality, report).items():
                    observations[camera].append(metrics)
        except (
            EpisodeValidationError,
            EvidenceEvaluationError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            failures.append(f"raw_episode artifact {path} failed validation: {exc}")
    available_cameras = {
        camera: _aggregate_camera(values)
        for camera, values in sorted(observations.items())
    }
    cameras = {
        "camera_front": _unavailable(
            "no front camera quality evidence was registered"
        ),
        "camera_dump": _unavailable(
            "no dump camera quality evidence was registered"
        ),
        **available_cameras,
    }
    return (
        {
            "episode_count": valid_episodes,
            "registered_episode_count": len(episodes),
            "training_frame_count": training_frames,
            "cameras": cameras,
        },
        ["raw_episode"] * valid_episodes,
        failures,
    )


def _evaluate_act_runtime(
    artifacts: Sequence[Mapping[str, object]],
    run_dir: Path | None,
) -> tuple[dict[str, object], list[str], list[str]]:
    runtime_logs = tuple(
        artifact
        for artifact in artifacts
        if artifact.get("role") == "runtime_log"
        and _metadata(artifact).get("analyzer") == "act_runtime"
    )
    if not runtime_logs:
        return _empty_runtime("no ACT Runtime log artifact was registered"), [], []

    reports = []
    failures: list[str] = []
    for artifact in runtime_logs:
        path = _artifact_snapshot_path(artifact, run_dir)
        mode = _text(_metadata(artifact).get("mode"), "runtime_log.metadata.mode")
        try:
            report = inspect_act_runtime_log(path, mode=mode)
        except (OSError, UnicodeError, ValueError) as exc:
            failures.append(f"runtime_log artifact {path} failed validation: {exc}")
            continue
        reports.append(report)
        failures.extend(
            f"runtime_log artifact {path}: {reason}"
            for reason in report.failure_reasons
        )
    if not reports:
        return _empty_runtime("no valid ACT Runtime log evidence"), [], failures

    total_steps = sum(report.step_count for report in reports)
    act = {
        "available": True,
        "passed": all(report.passed for report in reports),
        "log_count": len(reports),
        "modes": sorted({report.mode for report in reports}),
        "step_count": total_steps,
        "command_event_count": sum(report.command_event_count for report in reports),
        "serial_write_count": sum(report.serial_write_count for report in reports),
        "nonzero_serial_write_count": sum(
            report.nonzero_serial_write_count for report in reports
        ),
        "failure_reasons": [
            reason for report in reports for reason in report.failure_reasons
        ],
    }
    inference = {
        "available": True,
        "estimated_rate_hz": _weighted_mean(
            tuple(
                (report.estimated_step_rate_hz, report.step_count)
                for report in reports
            )
        ),
        "max_state_to_decision_ms": max(
            report.max_state_to_decision_ms for report in reports
        ),
        "max_camera_age_ms": max(report.max_camera_age_ms for report in reports),
    }
    deadline = {
        "available": True,
        "dropped_state_count": sum(report.dropped_state_count for report in reports),
        "deadline_miss_count": None,
        "unavailable_reason": (
            "ACT Runtime evidence does not record a per-step deadline-miss counter"
        ),
    }
    return (
        {"act": act, "inference": inference, "deadline": deadline},
        ["runtime_log"] * len(reports),
        failures,
    )


def _empty_runtime(reason: str) -> dict[str, object]:
    return {
        "act": _unavailable(reason),
        "inference": _unavailable(reason),
        "deadline": _unavailable(reason),
    }


def _evaluate_handoffs(
    artifacts: Sequence[Mapping[str, object]],
    run_dir: Path | None,
) -> tuple[dict[str, object], list[str], list[str]]:
    owner_logs = tuple(
        _artifact_snapshot_path(artifact, run_dir)
        for artifact in artifacts
        if artifact.get("role") == "mission_log"
        and _metadata(artifact).get("analyzer") == "resident_handoff"
    )
    if not owner_logs:
        return _unavailable("no resident handoff log artifact was registered"), [], []
    try:
        report = analyze_handoff_logs(owner_logs)
    except (HandoffMetricsError, OSError, UnicodeError, ValueError) as exc:
        return (
            _unavailable("resident handoff evidence failed validation"),
            [],
            [f"mission_log artifact failed validation: {exc}"],
        )
    return (
        {
            "available": True,
            "sample_count": report["sample_count"],
            "directions": report["directions"],
            "benchmark_passed": report["passed"],
            "benchmark_failure_reasons": report["failure_reasons"],
            "thresholds": report["thresholds"],
        },
        ["mission_log"] * len(owner_logs),
        [
            f"resident handoff benchmark: {reason}"
            for reason in report["failure_reasons"]
        ],
    )


def _artifact_snapshot_path(
    artifact: Mapping[str, object], run_dir: Path | None
) -> Path:
    if run_dir is None:
        raise EvidenceEvaluationError(
            "snapshot.run_dir is required to evaluate artifact snapshots"
        )
    relative = Path(_text(artifact.get("snapshot_path"), "artifact.snapshot_path"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise EvidenceEvaluationError("artifact.snapshot_path must be a safe relative path")
    return run_dir / relative


def _metadata(artifact: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(artifact.get("metadata", {}), "artifact.metadata")


def _read_json(path: Path, *, optional: bool) -> Mapping[str, object]:
    if optional and not path.is_file():
        return {}
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _camera_quality(
    quality: Mapping[str, object], report: object
) -> dict[str, Mapping[str, object]]:
    canonical = quality.get("camera_streams")
    if canonical is not None:
        streams = _mapping(canonical, "quality_report.camera_streams")
        unsupported = set(streams) - {"front", "dump"}
        if unsupported:
            raise EvidenceEvaluationError(
                "quality_report.camera_streams has unsupported roles: "
                f"{sorted(unsupported)}"
            )
        expected = set(getattr(report, "cameras", {}))
        missing = expected - set(streams)
        if missing:
            raise EvidenceEvaluationError(
                f"quality_report.camera_streams is missing roles: {sorted(missing)}"
            )
        return {
            f"camera_{role}": _mapping(
                streams[role], f"quality_report.camera_streams.{role}"
            )
            for role in sorted(streams)
        }

    timing = _mapping(quality.get("stream_timing", {}), "quality_report.stream_timing")
    front = _mapping(
        timing.get("camera_front", {}), "quality_report.stream_timing.camera_front"
    )
    gaps = _mapping(quality.get("sequence_gaps", {}), "quality_report.sequence_gaps")
    return {
        "camera_front": {
            "frame_count": front.get(
                "count", getattr(report, "camera_frame_count", None)
            ),
            "estimated_rate_hz": front.get("estimated_rate_hz"),
            "age_ms": _mapping(
                quality.get("camera_age_ms", {}), "quality_report.camera_age_ms"
            ),
            "sequence_gap_count": gaps.get("camera", 0),
            "queue_drop_count": quality.get("camera_queue_drop_count"),
        }
    }


def _aggregate_camera(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    frame_counts = tuple(
        _nonnegative_int(item.get("frame_count"), "camera.frame_count")
        for item in observations
    )
    rates = sorted(
        _finite_nonnegative(item.get("estimated_rate_hz"), "camera.rate")
        for item in observations
    )
    ages = tuple(_mapping(item.get("age_ms"), "camera.age_ms") for item in observations)
    return {
        "frame_count": sum(frame_counts),
        "rate_hz": {
            "p50": _nearest_rank(rates, 0.50),
            "p95": _nearest_rank(rates, 0.95),
            "min": rates[0],
            "max": rates[-1],
        },
        "age_ms": {
            "p50": _nearest_rank(
                sorted(
                    _finite_nonnegative(age.get("p50"), "camera.age_ms.p50")
                    for age in ages
                ),
                0.50,
            ),
            "p95": _nearest_rank(
                sorted(
                    _finite_nonnegative(age.get("p95"), "camera.age_ms.p95")
                    for age in ages
                ),
                0.95,
            ),
            "max": max(
                _finite_nonnegative(age.get("max"), "camera.age_ms.max")
                for age in ages
            ),
        },
        "sequence_gap_count": sum(
            _nonnegative_int(
                item.get("sequence_gap_count", 0), "camera.sequence_gap_count"
            )
            for item in observations
        ),
        "queue_drop_count": sum(
            _nonnegative_int(item.get("queue_drop_count"), "camera.queue_drop_count")
            for item in observations
        ),
    }


def _weighted_mean(values: Sequence[tuple[float, int]]) -> float:
    total = sum(weight for _value, weight in values)
    if total <= 0:
        raise EvidenceEvaluationError("cannot aggregate an ACT Runtime without steps")
    return sum(value * weight for value, weight in values) / total


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceEvaluationError(f"{field} must be an object")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceEvaluationError(f"{field} must be non-empty text")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceEvaluationError(f"{field} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceEvaluationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise EvidenceEvaluationError(f"{field} must be finite and non-negative")
    return result


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    return values[math.ceil(len(values) * probability) - 1]


def _unavailable(reason: str) -> dict[str, object]:
    return {"available": False, "unavailable_reason": reason}

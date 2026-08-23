"""Fail-closed validation for resuming a deterministic collection Run."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from .experiment_run import (
    ExperimentRun,
    ExperimentRunSnapshot,
    ExperimentRunValidationError,
    TaskContext,
    capture_repository_state,
    fingerprint_path,
)

if TYPE_CHECKING:
    from .collection_experiment_run import CollectionExperimentRunRequest


COLLECTION_SOURCE_BINDING_CONFIG_LABEL = "collection_source_binding"
_COLLECTION_SOURCE_BINDING_SCHEMA_VERSION = (
    "excavator_collection_source_binding.v1"
)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def capture_collection_source_binding(
    episode_path: Path, quality_path: Path
) -> dict[str, Any]:
    """Capture paths and bytes that may be registered after a crash."""

    sources = {
        "raw_episode": episode_path.resolve(strict=True),
        "quality_report": quality_path.resolve(strict=True),
    }
    return {
        "schema_version": _COLLECTION_SOURCE_BINDING_SCHEMA_VERSION,
        "artifacts": {
            artifact_id: {
                "source_path": str(source_path),
                **asdict(fingerprint_path(source_path)),
            }
            for artifact_id, source_path in sources.items()
        },
    }


def _source_binding_bytes(binding: Mapping[str, Any]) -> bytes:
    encoded = json.dumps(
        _thaw_json(binding),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"{encoded}\n".encode("utf-8")


@contextmanager
def materialize_collection_source_binding(binding: Mapping[str, Any]):
    """Expose a short-lived file for ExperimentRun's config snapshot API."""

    data = _source_binding_bytes(binding)
    with NamedTemporaryFile(
        mode="w+b",
        prefix="excavator-collection-source-",
        suffix=".json",
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        yield Path(handle.name)


def _validate_source_binding_snapshot(
    record: Any,
    binding: Mapping[str, Any],
    run_id: str,
) -> None:
    expected = _source_binding_bytes(binding)
    if (
        not isinstance(record, Mapping)
        or record.get("sha256") != hashlib.sha256(expected).hexdigest()
        or record.get("size_bytes") != len(expected)
    ):
        raise ExperimentRunValidationError(
            f"existing collection source binding does not match input: {run_id}"
        )


def _validate_file_snapshot(
    record: Any,
    source_path: Path,
    *,
    label: str,
    run_id: str,
) -> None:
    current = fingerprint_path(source_path)
    expected_source = source_path.resolve(strict=True)
    if (
        not isinstance(record, Mapping)
        or current.object_type != "file"
        or Path(str(record.get("source_path"))) != expected_source
        or record.get("sha256") != current.sha256
        or record.get("size_bytes") != current.size_bytes
    ):
        raise ExperimentRunValidationError(
            f"existing {label} snapshot does not match input: {run_id}"
        )


def _validate_start(
    snapshot: ExperimentRunSnapshot,
    request: CollectionExperimentRunRequest,
    task_context: TaskContext,
    run_id: str,
) -> None:
    current_repositories = {
        label: asdict(capture_repository_state(repository_path))
        for label, repository_path in request.repository_paths.items()
    }
    expected_fields = (
        ("run_kind", "run_kind", "collection_episode"),
        ("task_context", "TaskContext", asdict(task_context)),
        ("policy_ids", "policy_ids", dict(request.policy_ids)),
        ("host_topology", "host_topology", _thaw_json(request.host_topology)),
        ("repositories", "repository state", current_repositories),
        (
            "evidence_requirements",
            "evidence requirements",
            {
                "quality_report": {"required": True, "min_count": 1},
                "raw_episode": {"required": True, "min_count": 1},
            },
        ),
    )
    for field, label, expected in expected_fields:
        if _thaw_json(snapshot.start[field]) != expected:
            raise ExperimentRunValidationError(
                f"existing collection {label} does not match input: {run_id}"
            )

    config_snapshots = snapshot.start["config_snapshots"]
    if set(config_snapshots) != {
        "collection",
        "campaign_provenance",
        COLLECTION_SOURCE_BINDING_CONFIG_LABEL,
    }:
        raise ExperimentRunValidationError(
            f"existing collection config snapshots do not match contract: {run_id}"
        )
    current_files = (
        (
            config_snapshots["collection"],
            request.collection_config_path,
            "collection config",
        ),
        (
            config_snapshots["campaign_provenance"],
            request.campaign_provenance_path,
            "campaign provenance",
        ),
        (
            snapshot.start["machine_profile"],
            request.machine_profile_path,
            "machine profile",
        ),
    )
    for record, current_path, label in current_files:
        _validate_file_snapshot(
            record, current_path, label=label, run_id=run_id
        )


def _validate_artifacts(
    snapshot: ExperimentRunSnapshot,
    metadata: Mapping[str, Any],
    episode_path: Path,
    quality_path: Path,
    run_id: str,
) -> None:
    artifacts = {str(item["artifact_id"]): item for item in snapshot.artifacts}
    episode_id = str(metadata["episode_id"])
    expected_artifacts = {
        "raw_episode": {
            "source_path": episode_path.resolve(strict=True),
            "role": "raw_episode",
            "metadata": {"episode_id": episode_id},
        },
        "quality_report": {
            "source_path": quality_path.resolve(strict=True),
            "role": "quality_report",
            "metadata": {"episode_id": episode_id},
        },
    }
    expected_ids = set(expected_artifacts)
    actual_ids = set(artifacts)
    ids_match = (
        actual_ids.issubset(expected_ids)
        if snapshot.state == "active"
        else actual_ids == expected_ids
    )
    if not ids_match:
        raise ExperimentRunValidationError(
            f"existing collection artifacts do not match contract: {run_id}"
        )

    snapshot.verify_artifacts()
    for artifact_id, artifact in artifacts.items():
        expected = expected_artifacts[artifact_id]
        expected_path = expected["source_path"]
        if (
            Path(str(artifact["source_path"])) != expected_path
            or artifact["role"] != expected["role"]
            or dict(artifact["metadata"]) != expected["metadata"]
        ):
            raise ExperimentRunValidationError(
                f"existing {artifact_id} contract does not match Episode: {run_id}"
            )
        current = fingerprint_path(expected_path)
        if (
            current.object_type != artifact["object_type"]
            or current.sha256 != artifact["sha256"]
            or current.size_bytes != artifact["size_bytes"]
            or current.file_count != artifact["file_count"]
        ):
            raise ExperimentRunValidationError(
                f"existing {artifact_id} source fingerprint mismatch: {run_id}"
            )


def load_validated_collection_run(
    request: CollectionExperimentRunRequest,
    *,
    run_id: str,
    metadata: Mapping[str, Any],
    task_context: TaskContext,
    episode_path: Path,
    quality_path: Path,
    source_binding: Mapping[str, Any],
) -> ExperimentRunSnapshot:
    """Load an active/final Run only when it matches the current command."""

    snapshot = ExperimentRun.load(
        request.experiment_run_root, run_id
    ).snapshot()
    _validate_start(snapshot, request, task_context, run_id)
    _validate_artifacts(snapshot, metadata, episode_path, quality_path, run_id)
    if snapshot.state == "active" and len(snapshot.artifacts) < 2:
        _validate_source_binding_snapshot(
            snapshot.start["config_snapshots"][
                COLLECTION_SOURCE_BINDING_CONFIG_LABEL
            ],
            source_binding,
            run_id,
        )
    if snapshot.state == "active":
        return snapshot

    succeeded = metadata["status"] == "complete" and metadata["success"] is True
    expected_state = "success" if succeeded else "failure"
    expected_metrics = {
        "episode_id": metadata["episode_id"],
        "episode_status": metadata["status"],
        "episode_success": metadata["success"],
        "evaluation_scope": "training_internal",
    }
    if (
        snapshot.state != expected_state
        or dict(snapshot.final["metrics"]) != expected_metrics
    ):
        raise ExperimentRunValidationError(
            f"existing collection result does not match Episode: {run_id}"
        )
    return snapshot

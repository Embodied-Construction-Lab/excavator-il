"""Versioned value objects for :mod:`excavator_il.experiment_run`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final


EXPERIMENT_RUN_SCHEMA_VERSION: Final = "experiment_run.v1"
EXPERIMENT_EVENT_SCHEMA_VERSION: Final = "experiment_run_event.v1"
EXPERIMENT_ARTIFACT_SCHEMA_VERSION: Final = "experiment_run_artifact.v2"
EXPERIMENT_MANIFEST_SCHEMA_VERSION: Final = "experiment_run_manifest.v1"
EXPERIMENT_INDEX_SCHEMA_VERSION: Final = "experiment_run_index.v1"
EXPERIMENT_RUN_KINDS: Final = frozenset(
    {
        "reproducible_baseline",
        "collection_episode",
        "hybrid_live",
        "training",
        "evaluation",
        "diagnostic",
        "replay",
    }
)


class ExperimentRunError(RuntimeError):
    """Base error for Experiment Run evidence operations."""


class ExperimentRunValidationError(ExperimentRunError, ValueError):
    """Evidence or input does not conform to the versioned contract."""


class ExperimentRunFinalizedError(ExperimentRunError):
    """A caller attempted to mutate an already published run."""


@dataclasses.dataclass(frozen=True)
class TaskContext:
    task_variant: str
    soil_reset_block_id: str | None
    dig_point_id: str | None
    operator_id: str
    material_id: str | None


@dataclasses.dataclass(frozen=True)
class EvidenceRequirement:
    required: bool
    min_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.required, bool):
            raise ExperimentRunValidationError("evidence required must be bool")
        if (
            isinstance(self.min_count, bool)
            or not isinstance(self.min_count, int)
            or self.min_count < 0
        ):
            raise ExperimentRunValidationError("evidence min_count must be a non-negative int")
        if self.required and self.min_count < 1:
            raise ExperimentRunValidationError(
                "required evidence must have min_count of at least one"
            )


@dataclasses.dataclass(frozen=True)
class RepositoryState:
    source_path: str
    commit: str
    dirty: bool


@dataclasses.dataclass(frozen=True)
class PathFingerprint:
    object_type: str
    sha256: str
    size_bytes: int
    file_count: int


@dataclasses.dataclass(frozen=True)
class ExperimentRunSnapshot:
    run_id: str
    run_dir: Path
    state: str
    start: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    artifacts: tuple[Mapping[str, Any], ...]
    final: Mapping[str, Any] | None
    manifest: Mapping[str, Any] | None

    def verify_artifacts(self) -> None:
        """Re-hash every Run-owned artifact snapshot and reject any drift."""

        # Local import keeps the public API acyclic while the lifecycle module
        # re-exports these frozen value objects.
        from ._experiment_artifact_store import verify_registered_artifacts

        verify_registered_artifacts(self.artifacts, self.run_dir)

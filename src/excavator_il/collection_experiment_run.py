"""Experiment Run evidence adapter for one completed collection Episode.

The adapter is deliberately independent from Collector and UI lifecycles.  A
post-validation hook supplies an immutable, completed raw Episode; this module
only reads it, snapshots the experiment inputs, and publishes content hashes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from ._collection_experiment_resume import (
    COLLECTION_SOURCE_BINDING_CONFIG_LABEL,
    capture_collection_source_binding,
    load_validated_collection_run,
    materialize_collection_source_binding,
)
from .collector.config import (
    validate_collection_protocol,
    validate_target_source_provenance,
)
from .experiment_run import (
    EvidenceRequirement,
    ExperimentRun,
    ExperimentRunSnapshot,
    ExperimentRunValidationError,
    TaskContext,
    capture_repository_state,
    fingerprint_path,
)


_EPISODE_SCHEMA_VERSION = "excavator_demo_raw.v2"
_CAMPAIGN_PROVENANCE_SCHEMA_VERSION = (
    "excavator_collection_campaign_provenance.v1"
)
COLLECTION_EVIDENCE_CONFIG_SCHEMA_VERSION = (
    "excavator_collection_evidence_config.v2"
)
_TERMINAL_EPISODE_STATUSES = frozenset({"complete", "failed", "aborted"})
_PROTOCOL_FIELDS = frozenset(
    {"task_variant", "soil_reset_block_id", "dig_point_id"}
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_root",
        "collection_config",
        "machine_profile",
        "campaign_provenance",
        "repository_paths",
        "policy_ids",
        "host_topology",
    }
)
_CAMPAIGN_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "frozen_baseline",
        "airylidar_mission_targets",
        "f407_firmware_commit",
        "machine_profile_sha256",
        "dig_targets_m",
    }
)
_FROZEN_BASELINE_FIELDS = frozenset(
    {"baseline_id", "tag", "repository_commits"}
)
_FROZEN_REPOSITORIES = frozenset(
    {"excavator_il", "excavator_orin_runtime", "airylidar", "f407"}
)
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MISSION_TARGET_FIELDS = frozenset({"repository", "path", "sha256"})
_DIG_TARGET_IDS = frozenset({"dig_01", "dig_02", "dig_03"})


@dataclass(frozen=True)
class CollectionExperimentRunConfig:
    """Versioned Orin-local inputs shared by all collection Episodes."""

    evidence_root: Path
    collection_config_path: Path
    machine_profile_path: Path
    campaign_provenance_path: Path
    repository_paths: Mapping[str, Path]
    policy_ids: Mapping[str, str]
    host_topology: Mapping[str, Any]

    def request_for_episode(
        self, raw_episode_path: str | Path
    ) -> "CollectionExperimentRunRequest":
        return CollectionExperimentRunRequest(
            experiment_run_root=self.evidence_root,
            raw_episode_path=Path(raw_episode_path),
            collection_config_path=self.collection_config_path,
            machine_profile_path=self.machine_profile_path,
            campaign_provenance_path=self.campaign_provenance_path,
            repository_paths=self.repository_paths,
            policy_ids=self.policy_ids,
            host_topology=self.host_topology,
        )


def _config_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentRunValidationError(f"{label} must be an object")
    return value


def _config_path(value: Any, label: str, base: Path) -> Path:
    raw = _required_text(value, label)
    candidate = Path(raw).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _path_mapping(value: Any, label: str, base: Path) -> Mapping[str, Path]:
    mapping = _config_object(value, label)
    if not mapping:
        raise ExperimentRunValidationError(f"{label} must not be empty")
    result: dict[str, Path] = {}
    for key, raw_path in mapping.items():
        name = _required_text(key, f"{label} key")
        result[name] = _config_path(raw_path, f"{label}.{name}", base)
    return MappingProxyType(result)


def _text_mapping(value: Any, label: str) -> Mapping[str, str]:
    mapping = _config_object(value, label)
    if not mapping:
        raise ExperimentRunValidationError(f"{label} must not be empty")
    result: dict[str, str] = {}
    for key, raw_value in mapping.items():
        name = _required_text(key, f"{label} key")
        result[name] = _required_text(raw_value, f"{label}.{name}")
    return MappingProxyType(result)


def _json_mapping(value: Any, label: str) -> Mapping[str, Any]:
    mapping = _config_object(value, label)
    if not mapping:
        raise ExperimentRunValidationError(f"{label} must not be empty")
    try:
        thawed = _thaw_json(mapping)
        normalized = json.loads(
            json.dumps(thawed, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentRunValidationError(
            f"{label} must contain finite JSON values"
        ) from exc
    return _freeze_json(normalized)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def load_collection_experiment_run_config(
    path: str | Path,
) -> CollectionExperimentRunConfig:
    """Load the strict v2 Orin collection-evidence configuration."""

    config_path = Path(path).expanduser()
    root = _read_json_object(config_path, "collection evidence config")
    if set(root) != _CONFIG_FIELDS:
        raise ExperimentRunValidationError(
            "collection evidence config fields are invalid; expected "
            f"{sorted(_CONFIG_FIELDS)}"
        )
    if root["schema_version"] != COLLECTION_EVIDENCE_CONFIG_SCHEMA_VERSION:
        raise ExperimentRunValidationError(
            "schema_version must be "
            f"{COLLECTION_EVIDENCE_CONFIG_SCHEMA_VERSION}"
        )
    base = config_path.resolve(strict=True).parent
    return CollectionExperimentRunConfig(
        evidence_root=_config_path(root["evidence_root"], "evidence_root", base),
        collection_config_path=_config_path(
            root["collection_config"], "collection_config", base
        ),
        machine_profile_path=_config_path(
            root["machine_profile"], "machine_profile", base
        ),
        campaign_provenance_path=_config_path(
            root["campaign_provenance"], "campaign_provenance", base
        ),
        repository_paths=_path_mapping(
            root["repository_paths"], "repository_paths", base
        ),
        policy_ids=_text_mapping(root["policy_ids"], "policy_ids"),
        host_topology=_json_mapping(root["host_topology"], "host_topology"),
    )


@dataclass(frozen=True)
class CollectionExperimentRunRequest:
    """Explicit inputs needed to publish evidence for one demonstration."""

    experiment_run_root: Path
    raw_episode_path: Path
    collection_config_path: Path
    machine_profile_path: Path
    campaign_provenance_path: Path
    repository_paths: Mapping[str, Path]
    policy_ids: Mapping[str, str]
    host_topology: Mapping[str, Any]
    run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_run_root", Path(self.experiment_run_root).expanduser()
        )
        object.__setattr__(
            self, "raw_episode_path", Path(self.raw_episode_path).expanduser()
        )
        object.__setattr__(
            self,
            "collection_config_path",
            Path(self.collection_config_path).expanduser(),
        )
        object.__setattr__(
            self, "machine_profile_path", Path(self.machine_profile_path).expanduser()
        )
        object.__setattr__(
            self,
            "campaign_provenance_path",
            Path(self.campaign_provenance_path).expanduser(),
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
        object.__setattr__(self, "policy_ids", MappingProxyType(dict(self.policy_ids)))
        object.__setattr__(
            self,
            "host_topology",
            _json_mapping(self.host_topology, "host_topology"),
        )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunValidationError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentRunValidationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentRunValidationError(f"{label} must be a JSON object")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentRunValidationError(f"{label} must be non-empty text")
    return value


def _exact_object(
    value: Any, *, fields: frozenset[str], label: str
) -> Mapping[str, Any]:
    mapping = _config_object(value, label)
    if set(mapping) != fields:
        raise ExperimentRunValidationError(
            f"{label} fields are invalid; expected {sorted(fields)}"
        )
    return mapping


def _load_campaign_provenance(path: Path) -> dict[str, Any]:
    provenance = _read_json_object(path, "campaign provenance")
    if set(provenance) != _CAMPAIGN_PROVENANCE_FIELDS:
        raise ExperimentRunValidationError(
            "campaign provenance fields are invalid; expected "
            f"{sorted(_CAMPAIGN_PROVENANCE_FIELDS)}"
        )
    if provenance["schema_version"] != _CAMPAIGN_PROVENANCE_SCHEMA_VERSION:
        raise ExperimentRunValidationError(
            "campaign provenance schema_version must be "
            f"{_CAMPAIGN_PROVENANCE_SCHEMA_VERSION}"
        )
    _required_text(provenance["campaign_id"], "campaign provenance campaign_id")
    baseline = _exact_object(
        provenance["frozen_baseline"],
        fields=_FROZEN_BASELINE_FIELDS,
        label="campaign provenance frozen_baseline",
    )
    _required_text(baseline["baseline_id"], "frozen_baseline.baseline_id")
    _required_text(baseline["tag"], "frozen_baseline.tag")
    commits = _exact_object(
        baseline["repository_commits"],
        fields=_FROZEN_REPOSITORIES,
        label="frozen_baseline.repository_commits",
    )
    for repository, commit in commits.items():
        if not isinstance(commit, str) or _GIT_COMMIT_RE.fullmatch(commit) is None:
            raise ExperimentRunValidationError(
                f"frozen_baseline.repository_commits.{repository} must be a "
                "lowercase 40-character Git commit"
            )
    mission = _exact_object(
        provenance["airylidar_mission_targets"],
        fields=_MISSION_TARGET_FIELDS,
        label="campaign provenance airylidar_mission_targets",
    )
    if mission["repository"] != "airylidar":
        raise ExperimentRunValidationError(
            "airylidar_mission_targets.repository must be airylidar"
        )
    mission_path = _required_text(
        mission["path"], "airylidar_mission_targets.path"
    )
    parsed_mission_path = PurePosixPath(mission_path)
    if (
        parsed_mission_path.is_absolute()
        or ".." in parsed_mission_path.parts
        or parsed_mission_path.as_posix() != mission_path
    ):
        raise ExperimentRunValidationError(
            "airylidar_mission_targets.path must be a normalized "
            "repository-relative path"
        )
    mission_sha = mission["sha256"]
    if not isinstance(mission_sha, str) or _SHA256_RE.fullmatch(mission_sha) is None:
        raise ExperimentRunValidationError(
            "airylidar_mission_targets.sha256 must be lowercase SHA-256"
        )
    firmware_commit = provenance["f407_firmware_commit"]
    if (
        not isinstance(firmware_commit, str)
        or _GIT_COMMIT_RE.fullmatch(firmware_commit) is None
    ):
        raise ExperimentRunValidationError(
            "f407_firmware_commit must be a lowercase 40-character Git commit"
        )
    if firmware_commit != commits["f407"]:
        raise ExperimentRunValidationError(
            "f407_firmware_commit must match frozen_baseline.repository_commits.f407"
        )
    machine_profile_sha = provenance["machine_profile_sha256"]
    if (
        not isinstance(machine_profile_sha, str)
        or _SHA256_RE.fullmatch(machine_profile_sha) is None
    ):
        raise ExperimentRunValidationError(
            "machine_profile_sha256 must be lowercase SHA-256"
        )
    dig_targets = _exact_object(
        provenance["dig_targets_m"],
        fields=_DIG_TARGET_IDS,
        label="campaign provenance dig_targets_m",
    )
    for target_id, coordinates in dig_targets.items():
        if not isinstance(coordinates, list) or len(coordinates) != 3:
            raise ExperimentRunValidationError(
                f"dig_targets_m.{target_id} must contain three coordinates"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in coordinates
        ):
            raise ExperimentRunValidationError(
                f"dig_targets_m.{target_id} must contain finite coordinates"
            )
    return provenance


def _validate_episode_campaign_binding(
    metadata: Mapping[str, Any],
    task_context: TaskContext,
    campaign_provenance: Mapping[str, Any],
) -> None:
    try:
        target_source = validate_target_source_provenance(
            metadata.get("target_source_provenance")
        )
    except ValueError as exc:
        raise ExperimentRunValidationError(str(exc)) from exc
    mission_source = campaign_provenance["airylidar_mission_targets"]
    baseline_commit = campaign_provenance["frozen_baseline"][
        "repository_commits"
    ]["airylidar"]
    if target_source["commit"] != baseline_commit:
        raise ExperimentRunValidationError(
            "episode.json target source commit does not match campaign provenance"
        )
    if target_source["path"] != mission_source["path"]:
        raise ExperimentRunValidationError(
            "episode.json target source path does not match campaign provenance"
        )
    if target_source["sha256"] != mission_source["sha256"]:
        raise ExperimentRunValidationError(
            "episode.json target source SHA-256 does not match campaign provenance"
        )
    if metadata.get("firmware_commit") != campaign_provenance["f407_firmware_commit"]:
        raise ExperimentRunValidationError(
            "episode.json firmware_commit does not match campaign provenance"
        )
    if (
        metadata.get("machine_profile_hash")
        != campaign_provenance["machine_profile_sha256"]
    ):
        raise ExperimentRunValidationError(
            "episode.json machine_profile_hash does not match campaign provenance"
        )

    target_id = task_context.dig_point_id
    dig_targets = campaign_provenance["dig_targets_m"]
    if target_id not in dig_targets:
        raise ExperimentRunValidationError(
            "episode.json dig_point_id is not defined by campaign provenance"
        )
    actual_target = metadata.get("dig_target_m")
    expected_target = dig_targets[target_id]
    if (
        not isinstance(actual_target, list)
        or len(actual_target) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in actual_target
        )
        or any(
            not math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9
            )
            for actual, expected in zip(actual_target, expected_target, strict=True)
        )
    ):
        raise ExperimentRunValidationError(
            "episode.json dig_target_m does not match campaign provenance dig point"
        )


def _require_clean_repositories(
    repository_paths: Mapping[str, Path],
) -> None:
    for label, repository_path in repository_paths.items():
        state = capture_repository_state(repository_path)
        if state.dirty:
            raise ExperimentRunValidationError(
                f"formal collection repository must be clean: {label}"
            )


def _load_episode(
    raw_episode_path: Path,
) -> tuple[dict[str, Any], TaskContext, Path]:
    if raw_episode_path.is_symlink() or not raw_episode_path.is_dir():
        raise ExperimentRunValidationError(
            f"raw Episode must be a directory: {raw_episode_path}"
        )
    episode_path = raw_episode_path.resolve(strict=True)
    metadata = _read_json_object(episode_path / "episode.json", "episode.json")
    if metadata.get("schema_version") != _EPISODE_SCHEMA_VERSION:
        raise ExperimentRunValidationError(
            f"episode.json schema_version must be {_EPISODE_SCHEMA_VERSION}"
        )
    episode_id = _required_text(metadata.get("episode_id"), "episode_id")
    if episode_id != episode_path.name:
        raise ExperimentRunValidationError(
            "episode.json episode_id must match the raw Episode directory"
        )
    if metadata.get("task") != "ExecuteDig":
        raise ExperimentRunValidationError("episode.json task must be ExecuteDig")
    if metadata.get("recording_purpose") != "demonstration":
        raise ExperimentRunValidationError(
            "collection evidence requires recording_purpose=demonstration"
        )
    status = metadata.get("status")
    if status not in _TERMINAL_EPISODE_STATUSES:
        raise ExperimentRunValidationError(
            "episode.json status must be complete, failed, or aborted"
        )
    if not isinstance(metadata.get("success"), bool):
        raise ExperimentRunValidationError("episode.json success must be boolean")

    protocol = metadata.get("collection_protocol")
    if not isinstance(protocol, Mapping) or set(protocol) != _PROTOCOL_FIELDS:
        raise ExperimentRunValidationError(
            "episode.json collection_protocol must contain exactly task_variant, "
            "soil_reset_block_id and dig_point_id"
        )
    try:
        normalized_protocol = validate_collection_protocol(
            task_variant=protocol["task_variant"],
            soil_reset_block_id=protocol["soil_reset_block_id"],
            dig_point_id=protocol["dig_point_id"],
        )
    except ValueError as exc:
        raise ExperimentRunValidationError(
            f"invalid collection_protocol: {exc}"
        ) from exc

    context = TaskContext(
        task_variant=normalized_protocol["task_variant"],
        soil_reset_block_id=normalized_protocol["soil_reset_block_id"],
        dig_point_id=normalized_protocol["dig_point_id"],
        operator_id=_required_text(metadata.get("operator_id"), "operator_id"),
        material_id=_required_text(metadata.get("material_id"), "material_id"),
    )
    quality_path = episode_path / "quality_report.json"
    quality = _read_json_object(quality_path, "quality_report.json")
    quality_episode_id = quality.get("episode_id")
    if quality_episode_id != episode_id:
        raise ExperimentRunValidationError(
            "quality_report.json episode_id does not match episode.json"
        )
    if (
        metadata["status"] == "complete"
        and metadata["success"] is True
        and quality.get("passed") is not True
    ):
        raise ExperimentRunValidationError(
            "quality_report.json passed must be true for successful evidence"
        )
    return metadata, context, quality_path


def record_collection_experiment_run(
    request: CollectionExperimentRunRequest,
) -> ExperimentRunSnapshot:
    """Publish immutable evidence for one already completed raw Episode."""

    if not isinstance(request, CollectionExperimentRunRequest):
        raise TypeError("request must be CollectionExperimentRunRequest")
    campaign_provenance = _load_campaign_provenance(
        request.campaign_provenance_path
    )
    profile_fingerprint = fingerprint_path(request.machine_profile_path)
    if (
        profile_fingerprint.object_type != "file"
        or profile_fingerprint.sha256
        != campaign_provenance["machine_profile_sha256"]
    ):
        raise ExperimentRunValidationError(
            "machine profile file SHA-256 does not match campaign provenance"
        )
    metadata, task_context, quality_path = _load_episode(request.raw_episode_path)
    _validate_episode_campaign_binding(
        metadata, task_context, campaign_provenance
    )
    _require_clean_repositories(request.repository_paths)
    episode_path = request.raw_episode_path.resolve(strict=True)
    episode_id = str(metadata["episode_id"])
    deterministic_run_id = f"collection_{episode_id}"
    if request.run_id is not None and request.run_id != deterministic_run_id:
        raise ExperimentRunValidationError(
            f"collection run_id must be {deterministic_run_id}"
        )
    source_binding = capture_collection_source_binding(
        episode_path, quality_path
    )

    finalized_path = (
        request.experiment_run_root.expanduser()
        / "runs"
        / deterministic_run_id
    )
    if finalized_path.is_dir() and not finalized_path.is_symlink():
        return load_validated_collection_run(
            request,
            run_id=deterministic_run_id,
            metadata=metadata,
            task_context=task_context,
            episode_path=episode_path,
            quality_path=quality_path,
            source_binding=source_binding,
        )

    registered_artifact_ids: frozenset[str] = frozenset()
    with materialize_collection_source_binding(source_binding) as binding_path:
        try:
            run = ExperimentRun.create(
                request.experiment_run_root,
                run_id=deterministic_run_id,
                run_kind="collection_episode",
                task_context=task_context,
                policy_ids=request.policy_ids,
                host_topology=request.host_topology,
                repository_paths=request.repository_paths,
                config_paths={
                    "collection": request.collection_config_path,
                    "campaign_provenance": request.campaign_provenance_path,
                    COLLECTION_SOURCE_BINDING_CONFIG_LABEL: binding_path,
                },
                machine_profile_path=request.machine_profile_path,
                evidence_requirements={
                    "raw_episode": EvidenceRequirement(required=True, min_count=1),
                    "quality_report": EvidenceRequirement(required=True, min_count=1),
                },
            )
        except FileExistsError:
            existing = load_validated_collection_run(
                request,
                run_id=deterministic_run_id,
                metadata=metadata,
                task_context=task_context,
                episode_path=episode_path,
                quality_path=quality_path,
                source_binding=source_binding,
            )
            if existing.state != "active":
                return existing
            registered_artifact_ids = frozenset(
                str(artifact["artifact_id"])
                for artifact in existing.artifacts
            )
            run = ExperimentRun.load(
                request.experiment_run_root, deterministic_run_id
            )
    if "raw_episode" not in registered_artifact_ids:
        run.register_artifact(
            "raw_episode",
            episode_path,
            role="raw_episode",
            metadata={"episode_id": episode_id},
        )
    if "quality_report" not in registered_artifact_ids:
        run.register_artifact(
            "quality_report",
            quality_path,
            role="quality_report",
            metadata={"episode_id": episode_id},
        )
    succeeded = metadata["status"] == "complete" and metadata["success"] is True
    final_status = "success" if succeeded else "failure"
    return run.finalize(
        final_status,
        metrics={
            "episode_id": episode_id,
            "episode_status": metadata["status"],
            "episode_success": metadata["success"],
            "evaluation_scope": "training_internal",
        },
        summary=(
            "Completed demonstration Episode retained with required evidence."
            if succeeded
            else "Non-success demonstration Episode retained as failure evidence."
        ),
    )


__all__ = [
    "COLLECTION_EVIDENCE_CONFIG_SCHEMA_VERSION",
    "CollectionExperimentRunConfig",
    "CollectionExperimentRunRequest",
    "load_collection_experiment_run_config",
    "record_collection_experiment_run",
]

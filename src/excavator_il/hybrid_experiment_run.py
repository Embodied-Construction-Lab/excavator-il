"""Explicit Experiment Run adapter for the hybrid Mission supervisor.

The active hybrid configuration identifies the operator, material, selected
runtime backend, and config files, but it does not identify a local evidence
root, every participating Git worktree, the local machine profile snapshot, or
the exact deployed policy revisions.  Those values therefore stay explicit in
``HybridExperimentRunConfig`` instead of being guessed from checkout layout.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._hybrid_evidence_lifecycle import HybridMissionEvidenceLifecycle
from ._hybrid_experiment_contract import (
    ARTIFACT_CONFIG_FIELDS as _ARTIFACT_CONFIG_FIELDS,
    AUTOMATIC_CONFIG_LABELS as _AUTOMATIC_CONFIG_LABELS,
    EVALUATION_SCOPES as _EVALUATION_SCOPES,
    HOST_CONFIG_FIELDS as _HOST_CONFIG_FIELDS,
    HYBRID_CONFIG_FIELDS as _HYBRID_CONFIG_FIELDS,
    HYBRID_EXPERIMENT_RUN_CONFIG_SCHEMA_VERSION,
    HYBRID_POLICY_EVIDENCE_ROLES,
    HybridEvidenceArtifact,
    POLICY_LABELS as _POLICY_LABELS,
    REPOSITORY_LABELS as _REPOSITORY_LABELS,
    REQUIREMENT_CONFIG_FIELDS as _REQUIREMENT_CONFIG_FIELDS,
    TASK_CONTEXT_CONFIG_FIELDS as _TASK_CONTEXT_CONFIG_FIELDS,
    config_path as _config_path,
    config_identifier as _config_identifier,
    config_sha256 as _config_sha256,
    config_text as _config_text,
    dig_policy_family as _dig_policy_family,
    expected_policy_evidence_roles as _expected_policy_evidence_roles,
    fixed_action_profile_id as _fixed_action_profile_id,
    optional_config_text as _optional_config_text,
    path_mapping as _path_mapping,
    require_no_symlink_components as _require_no_symlink_components,
    required_count as _required_count,
    requirement as _requirement,
    strict_object as _strict_object,
    string_mapping as _string_mapping,
    trajectory_controller_family as _trajectory_controller_family,
    validate_repository_scope as _validate_repository_scope,
)
from .experiment_run import EvidenceRequirement, ExperimentRun, TaskContext
from .guided_episode import GuidedEpisodeConfig
from .hybrid_mission import HybridMissionConfig


class HybridEvidenceIncompleteError(RuntimeError):
    """A Mission ended, but its required evidence could not be published as success."""

    def __init__(self, message: str, *, finalized: bool = False) -> None:
        super().__init__(message)
        self.finalized = finalized


def _require_run_id(run: Any) -> str:
    value = getattr(run, "run_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("evidence run must expose a non-empty run_id")
    return value


@dataclass(frozen=True)
class HybridMissionRunRequest:
    """Immutable inputs required to create evidence for one Mission run."""

    config_path: Path
    dig_target_id: str
    automatic: bool
    requested_cycles: int


@dataclass(frozen=True)
class HybridExperimentRunConfig:
    """Host-local, reproducibility-critical inputs absent from Mission config."""

    evidence_root: Path
    machine_profile_path: Path
    repository_paths: Mapping[str, Path]
    config_paths: Mapping[str, Path]
    policy_ids: Mapping[str, str]
    host_topology: Mapping[str, Any]
    mission_id: str
    mission_sha256: str
    evaluation_scope: str = "training_internal"
    evidence_requirements: Mapping[
        str, EvidenceRequirement | Mapping[str, Any]
    ] = field(default_factory=dict)
    task_variant: str = "dig_transport_dump"
    soil_reset_block_id: str | None = None
    operator_id: str | None = None
    material_id: str | None = None
    artifacts: tuple[HybridEvidenceArtifact, ...] = ()
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "HybridExperimentRunConfig":
        """Load the versioned host-local evidence contract without guessing paths."""

        source_path = Path(path).expanduser().absolute()
        _require_no_symlink_components(source_path, "hybrid evidence config")
        config_path = source_path.resolve()
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load hybrid evidence config {config_path}: {exc}"
            ) from exc
        root = _strict_object(
            parsed,
            "hybrid evidence config",
            expected_fields=_HYBRID_CONFIG_FIELDS,
        )
        if root["schema_version"] != HYBRID_EXPERIMENT_RUN_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{HYBRID_EXPERIMENT_RUN_CONFIG_SCHEMA_VERSION}"
            )
        base = config_path.parent
        task = _strict_object(
            root["task_context"],
            "task_context",
            expected_fields=_TASK_CONTEXT_CONFIG_FIELDS,
        )
        hosts = _strict_object(
            root["host_topology"],
            "host_topology",
            expected_fields=frozenset({"pc", "orin"}),
        )
        host_topology: dict[str, Any] = {}
        for label in ("pc", "orin"):
            host = _strict_object(
                hosts[label],
                f"host_topology.{label}",
                expected_fields=_HOST_CONFIG_FIELDS,
            )
            host_topology[label] = {
                "host": _config_text(host["host"], f"host_topology.{label}.host"),
                "role": _config_text(host["role"], f"host_topology.{label}.role"),
            }
        policy_ids = _string_mapping(
            root["policy_ids"],
            "policy_ids",
            expected_labels=_POLICY_LABELS,
        )
        expected_requirement_roles = frozenset(
            _expected_policy_evidence_roles(
                dig_policy_family=_dig_policy_family(policy_ids["dig_policy"]),
                trajectory_controller_family=_trajectory_controller_family(
                    policy_ids["trajectory_controller"]
                ),
            )
        )
        raw_requirements = _strict_object(
            root["evidence_requirements"], "evidence_requirements"
        )
        if set(raw_requirements) != expected_requirement_roles:
            raise ValueError(
                "evidence_requirements must contain exactly the configured "
                "policy evidence roles"
            )
        requirements: dict[str, EvidenceRequirement] = {}
        for role in sorted(expected_requirement_roles):
            item = _strict_object(
                raw_requirements[role],
                f"evidence_requirements.{role}",
                expected_fields=_REQUIREMENT_CONFIG_FIELDS,
            )
            required = item["required"]
            minimum = item["min_count"]
            if not isinstance(required, bool):
                raise ValueError(
                    f"evidence_requirements.{role}.required must be boolean"
                )
            if isinstance(minimum, bool) or not isinstance(minimum, int):
                raise ValueError(
                    f"evidence_requirements.{role}.min_count must be an integer"
                )
            requirements[role] = EvidenceRequirement(
                required=required,
                min_count=minimum,
            )
        raw_artifacts = root["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise ValueError("artifacts must be an array")
        artifacts: list[HybridEvidenceArtifact] = []
        for index, value in enumerate(raw_artifacts):
            item = _strict_object(
                value,
                f"artifacts[{index}]",
                expected_fields=_ARTIFACT_CONFIG_FIELDS,
            )
            metadata = _strict_object(
                item["metadata"], f"artifacts[{index}].metadata"
            )
            artifacts.append(
                HybridEvidenceArtifact(
                    artifact_id=_config_text(
                        item["artifact_id"], f"artifacts[{index}].artifact_id"
                    ),
                    source_path=_config_path(
                        item["source_path"],
                        f"artifacts[{index}].source_path",
                        base,
                    ),
                    role=_config_text(item["role"], f"artifacts[{index}].role"),
                    metadata=metadata,
                )
            )
        return cls(
            evidence_root=_config_path(root["evidence_root"], "evidence_root", base),
            machine_profile_path=_config_path(
                root["machine_profile_path"], "machine_profile_path", base
            ),
            repository_paths=_path_mapping(
                root["repository_paths"],
                "repository_paths",
                base,
                expected_labels=_REPOSITORY_LABELS,
            ),
            config_paths=_path_mapping(root["config_paths"], "config_paths", base),
            policy_ids=policy_ids,
            host_topology=host_topology,
            mission_id=_config_identifier(root["mission_id"], "mission_id"),
            mission_sha256=_config_sha256(
                root["mission_sha256"], "mission_sha256"
            ),
            evaluation_scope=_config_text(
                root["evaluation_scope"], "evaluation_scope"
            ),
            evidence_requirements=requirements,
            task_variant=_config_text(
                task["task_variant"], "task_context.task_variant"
            ),
            soil_reset_block_id=_optional_config_text(
                task["soil_reset_block_id"], "task_context.soil_reset_block_id"
            ),
            operator_id=_optional_config_text(
                task["operator_id"], "task_context.operator_id"
            ),
            material_id=_optional_config_text(
                task["material_id"], "task_context.material_id"
            ),
            artifacts=tuple(artifacts),
            source_path=source_path,
        )

    def __post_init__(self) -> None:
        evidence_root = Path(self.evidence_root).expanduser()
        profile_path = Path(self.machine_profile_path).expanduser()
        repositories = {
            str(label): Path(path).expanduser()
            for label, path in self.repository_paths.items()
        }
        configs = {
            str(label): Path(path).expanduser()
            for label, path in self.config_paths.items()
        }
        reserved = _AUTOMATIC_CONFIG_LABELS.intersection(configs)
        if reserved:
            raise ValueError(
                "config_paths must not replace automatically captured labels: "
                + ", ".join(sorted(reserved))
            )
        if not isinstance(self.task_variant, str) or not self.task_variant.strip():
            raise ValueError("task_variant must be non-empty text")
        if self.evaluation_scope not in _EVALUATION_SCOPES:
            raise ValueError(
                "evaluation_scope must be training_internal or held_out_experiment"
            )
        for field_name in ("soil_reset_block_id", "operator_id", "material_id"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be non-empty text when provided")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, HybridEvidenceArtifact) for item in artifacts):
            raise ValueError("artifacts must contain HybridEvidenceArtifact values")
        artifact_ids = tuple(item.artifact_id for item in artifacts)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifact ids must be unique")
        object.__setattr__(self, "evidence_root", evidence_root)
        object.__setattr__(self, "machine_profile_path", profile_path)
        object.__setattr__(self, "repository_paths", MappingProxyType(repositories))
        object.__setattr__(self, "config_paths", MappingProxyType(configs))
        object.__setattr__(
            self,
            "policy_ids",
            MappingProxyType(dict(self.policy_ids)),
        )
        object.__setattr__(
            self,
            "host_topology",
            MappingProxyType(dict(self.host_topology)),
        )
        object.__setattr__(
            self,
            "mission_id",
            _config_identifier(self.mission_id, "mission_id"),
        )
        object.__setattr__(
            self,
            "mission_sha256",
            _config_sha256(self.mission_sha256, "mission_sha256"),
        )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(
            self,
            "source_path",
            None
            if self.source_path is None
            else Path(self.source_path).expanduser().absolute(),
        )
        requirements = dict(self.evidence_requirements)
        dig_policy_family = _dig_policy_family(self.policy_ids.get("dig_policy"))
        controller_family = _trajectory_controller_family(
            self.policy_ids.get("trajectory_controller")
        )
        expected_policy_roles = frozenset(
            _expected_policy_evidence_roles(
                dig_policy_family=dig_policy_family,
                trajectory_controller_family=controller_family,
            )
        )
        unexpected_requirement_roles = frozenset(requirements) - expected_policy_roles
        if unexpected_requirement_roles:
            raise ValueError(
                "evidence_requirements roles do not match configured policy "
                "families: "
                + ", ".join(sorted(unexpected_requirement_roles))
            )
        required_counts = {
            role: _required_count(
                requirements.setdefault(
                    role,
                    EvidenceRequirement(required=True, min_count=1),
                ),
                role,
            )
            for role in expected_policy_roles
        }
        forbidden_policy_roles = frozenset(HYBRID_POLICY_EVIDENCE_ROLES) - (
            expected_policy_roles
        )
        forbidden_bound_roles = sorted(
            {
                item.role
                for item in artifacts
                if item.role in forbidden_policy_roles
            }
        )
        if forbidden_bound_roles:
            raise ValueError(
                "configured policy families must not bind unsupported policy "
                "artifacts: "
                + ", ".join(forbidden_bound_roles)
            )
        observed_roles = Counter(item.role for item in artifacts)
        missing_bindings = [
            role
            for role in required_counts
            if observed_roles[role] < required_counts[role]
        ]
        if missing_bindings:
            raise ValueError(
                "hybrid evidence artifacts must bind required roles: "
                + ", ".join(missing_bindings)
            )
        object.__setattr__(
            self,
            "evidence_requirements",
            MappingProxyType(requirements),
        )

    def validate_repository_scope(self) -> None:
        """Reject loaded provenance paths outside their declared workspace."""

        if self.source_path is None:
            raise ValueError(
                "source_path is required for repository-scoped hybrid evidence"
            )
        _validate_repository_scope(
            source_path=self.source_path,
            evidence_root=self.evidence_root,
            machine_profile_path=self.machine_profile_path,
            repository_paths=self.repository_paths,
            config_paths=self.config_paths,
            artifact_paths=tuple(
                artifact.source_path for artifact in self.artifacts
            ),
        )


class _ConfiguredHybridExperimentRun:
    def __init__(
        self,
        run: Any,
        *,
        artifacts: tuple[HybridEvidenceArtifact, ...],
        evidence_requirements: Mapping[
            str, EvidenceRequirement | Mapping[str, Any]
        ],
        evaluation_scope: str,
        mission_id: str,
        mission_sha256: str,
        experiment_profile_id: str | None = None,
        initial_events: tuple[tuple[str, Mapping[str, Any]], ...] = (),
        prestart_artifacts: tuple[HybridEvidenceArtifact, ...] = (),
        path_is_available: Callable[[Path], bool],
    ) -> None:
        self._run = run
        self._artifacts = artifacts
        self._path_is_available = path_is_available
        self._evaluation_scope = evaluation_scope
        self._mission_id = mission_id
        self._mission_sha256 = mission_sha256
        self._experiment_profile_id = experiment_profile_id
        self._registered_artifact_ids: set[str] = set()
        self._registered_roles: Counter[str] = Counter()
        self._reported_unavailable_ids: set[str] = set()
        self._required_roles = {
            role: minimum
            for role, specification in evidence_requirements.items()
            for required, minimum in (_requirement(specification, role),)
            if required
        }
        self._initial_events = deque(initial_events)
        self._prestart_artifacts = deque(prestart_artifacts)

    @property
    def run_id(self) -> str:
        return _require_run_id(self._run)

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        self._flush_prestart_evidence()
        return self._run.append_event(event_type, payload)

    def _flush_prestart_evidence(self) -> None:
        while self._initial_events:
            event_type, payload = self._initial_events[0]
            self._run.append_event(event_type, payload)
            self._initial_events.popleft()
        while self._prestart_artifacts:
            artifact = self._prestart_artifacts[0]
            self.register_artifact(
                artifact.artifact_id,
                artifact.source_path,
                role=artifact.role,
                metadata=artifact.metadata,
            )
            self._prestart_artifacts.popleft()

    def register_artifact(
        self,
        artifact_id: str,
        source_path: str | Path,
        *,
        role: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        record = self._run.register_artifact(
            artifact_id,
            source_path,
            role=role,
            metadata=metadata,
        )
        self._registered_artifact_ids.add(artifact_id)
        self._registered_roles[role] += 1
        return record

    def finalize(
        self,
        status: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        summary: str | None = None,
    ) -> Any:
        self._flush_prestart_evidence()
        final_metrics = {
            **dict(metrics or {}),
            "evaluation_scope": self._evaluation_scope,
            "mission_id": self._mission_id,
            "mission_sha256": self._mission_sha256,
            **(
                {"experiment_profile_id": self._experiment_profile_id}
                if self._experiment_profile_id is not None
                else {}
            ),
        }
        for artifact in self._artifacts:
            if artifact.artifact_id in self._registered_artifact_ids:
                continue
            available = self._path_is_available(artifact.source_path)
            if available:
                self.register_artifact(
                    artifact.artifact_id,
                    artifact.source_path,
                    role=artifact.role,
                    metadata=artifact.metadata,
                )
                continue
            if artifact.artifact_id not in self._reported_unavailable_ids:
                self.append_event(
                    "artifact_unavailable",
                    {
                        "run_id": self.run_id,
                        "artifact_id": artifact.artifact_id,
                        "role": artifact.role,
                        "path": str(artifact.source_path),
                    },
                )
                self._reported_unavailable_ids.add(artifact.artifact_id)
        missing_roles = [
            role
            for role, minimum in sorted(self._required_roles.items())
            if self._registered_roles[role] < minimum
        ]
        if status == "success" and missing_roles:
            self.append_event(
                "evidence_incomplete",
                {
                    "run_id": self.run_id,
                    "missing_required_roles": missing_roles,
                },
            )
            missing_text = ", ".join(missing_roles)
            failure_metrics = {
                **final_metrics,
                "evidence_complete": False,
                "intended_status": status,
            }
            self._run.finalize(
                "failure",
                metrics=failure_metrics,
                summary=(
                    f"{summary}; evidence incomplete: {missing_text}"
                    if summary
                    else f"evidence incomplete: {missing_text}"
                ),
            )
            raise HybridEvidenceIncompleteError(
                f"required hybrid evidence is unavailable: {missing_text}",
                finalized=True,
            )
        return self._run.finalize(status, metrics=final_metrics, summary=summary)


class HybridExperimentRunFactory:
    """Create one ``hybrid_live`` Experiment Run from explicit host settings."""

    def __init__(
        self,
        config: HybridExperimentRunConfig,
        *,
        run_creator: Callable[..., Any] = ExperimentRun.create,
        hybrid_config_loader: Callable[[str | Path], Any] = HybridMissionConfig.load,
        guided_config_loader: Callable[[str | Path], Any] = GuidedEpisodeConfig.load,
        path_is_available: Callable[[Path], bool] = lambda path: path.exists(),
        runtime_config_label: str = "hybrid_mission",
        runtime_backend: str | None = None,
    ) -> None:
        if runtime_config_label not in {
            "hybrid_mission",
            "resident_fixed_cycle",
        }:
            raise ValueError("runtime_config_label is unsupported")
        if runtime_backend is not None:
            _config_text(runtime_backend, "runtime_backend")
        self._config = config
        self._run_creator = run_creator
        self._hybrid_config_loader = hybrid_config_loader
        self._guided_config_loader = guided_config_loader
        self._path_is_available = path_is_available
        self._runtime_config_label = runtime_config_label
        self._runtime_backend = runtime_backend
        self._config.validate_repository_scope()
        self._experiment_profile = self._load_experiment_profile()
        self._experiment_profile_id = (
            None
            if self._experiment_profile is None
            else self._experiment_profile.profile_id
        )

    def _load_experiment_profile(self) -> Any | None:
        profile_path = self._config.config_paths.get("experiment_profile")
        if profile_path is None:
            return None
        from .icra2027_experiment_profile import Icra2027ExperimentProfile

        profile = Icra2027ExperimentProfile.load(profile_path)
        profile.preflight()
        expected_scope = (
            "held_out_experiment"
            if profile.readiness == "ready"
            else "training_internal"
        )
        if self._config.evaluation_scope != expected_scope:
            raise ValueError(
                f"{profile.readiness} experiment_profile requires "
                f"{expected_scope} evidence scope"
            )
        return profile

    def _require_profile_runtime_binding(self, config_path: Path) -> None:
        if self._experiment_profile is None:
            return
        expected = self._experiment_profile.bindings[
            "resident_fixed_cycle_config"
        ]
        if expected is None or Path(config_path).expanduser().resolve() != expected:
            raise ValueError(
                "experiment_profile runtime config does not match the Mission "
                f"request: expected {expected}, got "
                f"{Path(config_path).expanduser().resolve()}"
            )

    def _require_fixed_action_policy_binding(self) -> None:
        policy_id = self._config.policy_ids["dig_policy"]
        if _dig_policy_family(policy_id) != "fixed_action":
            return
        expected_profile_id = policy_id.partition(":")[2]
        for artifact in self._config.artifacts:
            if artifact.role != "fixed_action_profile":
                continue
            actual_profile_id = _fixed_action_profile_id(artifact.source_path)
            if actual_profile_id != expected_profile_id:
                raise ValueError(
                    "fixed_action policy identity does not match bound "
                    "fixed_action_profile: "
                    f"expected {expected_profile_id}, got {actual_profile_id}"
                )

    def preflight(self) -> None:
        """Validate static host-local evidence inputs without starting hardware."""

        self._config.validate_repository_scope()
        if not self._config.machine_profile_path.is_file():
            raise ValueError(
                "machine_profile_path does not exist as a file: "
                f"{self._config.machine_profile_path}"
            )
        for label, path in self._config.repository_paths.items():
            if not path.is_dir():
                raise ValueError(
                    f"repository_paths.{label} does not exist as a directory: {path}"
                )
        for label, path in self._config.config_paths.items():
            if not path.is_file():
                raise ValueError(
                    f"config_paths.{label} does not exist as a file: {path}"
                )
        for artifact in self._config.artifacts:
            if not self._path_is_available(artifact.source_path):
                raise ValueError(
                    f"artifact {artifact.role} does not exist: {artifact.source_path}"
                )
            if not (
                artifact.source_path.is_file() or artifact.source_path.is_dir()
            ):
                raise ValueError(
                    f"artifact {artifact.role} must be a file or directory: "
                    f"{artifact.source_path}"
                )
        self._require_fixed_action_policy_binding()
        evidence_parent = self._config.evidence_root.parent
        if not evidence_parent.is_dir():
            raise ValueError(
                f"evidence_root parent does not exist: {evidence_parent}"
            )
        if (
            self._config.evidence_root.exists()
            and not self._config.evidence_root.is_dir()
        ):
            raise ValueError(
                f"evidence_root must be a directory: {self._config.evidence_root}"
            )

    def __call__(self, request: HybridMissionRunRequest) -> Any:
        self._config.validate_repository_scope()
        self._require_profile_runtime_binding(request.config_path)
        hybrid = self._hybrid_config_loader(request.config_path)
        if (
            self._runtime_config_label == "resident_fixed_cycle"
            and getattr(hybrid, "expected_mission_id", None) != self._config.mission_id
        ):
            raise ValueError("runtime mission_id does not match evidence mission_id")
        if (
            self._runtime_config_label == "resident_fixed_cycle"
            and getattr(hybrid, "expected_mission_sha256", None)
            != self._config.mission_sha256
        ):
            raise ValueError("runtime Mission digest does not match evidence")
        guided_path = Path(hybrid.guided_config)
        guided = self._guided_config_loader(guided_path)
        available_required_roles = Counter(
            artifact.role
            for artifact in self._config.artifacts
            if artifact.role in HYBRID_POLICY_EVIDENCE_ROLES
            and self._path_is_available(artifact.source_path)
        )
        missing_required_roles = [
            role
            for role, specification in sorted(
                self._config.evidence_requirements.items()
            )
            for required, minimum in (_requirement(specification, role),)
            if (
                role in HYBRID_POLICY_EVIDENCE_ROLES
                and required
                and available_required_roles[role] < minimum
            )
        ]
        if missing_required_roles:
            raise HybridEvidenceIncompleteError(
                "required hybrid evidence is unavailable before Mission start: "
                + ", ".join(missing_required_roles)
            )
        self._require_fixed_action_policy_binding()
        operator_id = self._config.operator_id or guided.operator_id
        material_id = self._config.material_id or guided.material_id
        config_paths = {
            **self._config.config_paths,
            "hybrid_evidence": self._config.source_path,
            "guided_episode": guided_path,
            self._runtime_config_label: request.config_path,
        }
        run = self._run_creator(
            self._config.evidence_root,
            run_kind="hybrid_live",
            task_context=TaskContext(
                task_variant=self._config.task_variant,
                soil_reset_block_id=self._config.soil_reset_block_id,
                dig_point_id=request.dig_target_id,
                operator_id=operator_id,
                material_id=material_id,
            ),
            policy_ids=self._config.policy_ids,
            host_topology=self._config.host_topology,
            repository_paths=self._config.repository_paths,
            config_paths=config_paths,
            machine_profile_path=self._config.machine_profile_path,
            evidence_requirements=self._config.evidence_requirements,
        )
        return _ConfiguredHybridExperimentRun(
            run,
            artifacts=self._config.artifacts,
            evidence_requirements=self._config.evidence_requirements,
            evaluation_scope=self._config.evaluation_scope,
            mission_id=self._config.mission_id,
            mission_sha256=self._config.mission_sha256,
            experiment_profile_id=self._experiment_profile_id,
            initial_events=(
                (
                    "runtime_selected",
                    {
                        "run_id": _require_run_id(run),
                        "runtime_backend": (
                            self._runtime_backend
                            if self._runtime_backend is not None
                            else str(hybrid.runtime_backend)
                        ),
                        "automatic": request.automatic,
                        "requested_cycles": request.requested_cycles,
                        "mission_id": self._config.mission_id,
                        "mission_sha256": self._config.mission_sha256,
                        **(
                            {"experiment_profile_id": self._experiment_profile_id}
                            if self._experiment_profile_id is not None
                            else {}
                        ),
                    },
                ),
            ),
            prestart_artifacts=tuple(
                artifact
                for artifact in self._config.artifacts
                if artifact.role in HYBRID_POLICY_EVIDENCE_ROLES
            ),
            path_is_available=self._path_is_available,
        )

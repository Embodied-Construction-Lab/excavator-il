"""Private parsing and policy-evidence rules for hybrid Experiment Runs.

This module keeps the versioned host-local contract and its policy-family
validation cohesive.  Public callers continue to use
``excavator_il.hybrid_experiment_run``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from .experiment_run import EvidenceRequirement


AUTOMATIC_CONFIG_LABELS = frozenset(
    {"guided_episode", "hybrid_mission", "resident_fixed_cycle"}
)
HYBRID_POLICY_EVIDENCE_ROLES = (
    "act_deployment_manifest",
    "act_policy_checkpoint",
    "fixed_action_profile",
    "rl_onnx_model",
)
HYBRID_EXPERIMENT_RUN_CONFIG_SCHEMA_VERSION = (
    "excavator_hybrid_evidence_config.v2"
)
HYBRID_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_root",
        "machine_profile_path",
        "mission_id",
        "mission_sha256",
        "repository_paths",
        "config_paths",
        "policy_ids",
        "host_topology",
        "evaluation_scope",
        "evidence_requirements",
        "task_context",
        "artifacts",
    }
)
REPOSITORY_LABELS = frozenset(
    {"excavator_il", "excavator_orin_runtime", "airy_lidar", "rl_excavator"}
)
POLICY_LABELS = frozenset({"dig_policy", "trajectory_controller"})
TASK_CONTEXT_CONFIG_FIELDS = frozenset(
    {"task_variant", "soil_reset_block_id", "operator_id", "material_id"}
)
ARTIFACT_CONFIG_FIELDS = frozenset(
    {"artifact_id", "source_path", "role", "metadata"}
)
REQUIREMENT_CONFIG_FIELDS = frozenset({"required", "min_count"})
HOST_CONFIG_FIELDS = frozenset({"host", "role"})
EVALUATION_SCOPES = frozenset({"training_internal", "held_out_experiment"})
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")

_DIG_POLICY_FAMILY_EVIDENCE_ROLES = {
    "lerobot_act": (
        "act_deployment_manifest",
        "act_policy_checkpoint",
    ),
    "fixed_action": ("fixed_action_profile",),
}
_TRAJECTORY_CONTROLLER_FAMILY_EVIDENCE_ROLES = {
    "onnx_rl": ("rl_onnx_model",),
    "cartesian_p": (),
}


def strict_object(
    value: object,
    field: str,
    *,
    expected_fields: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    if expected_fields is not None and set(value) != expected_fields:
        raise ValueError(
            f"{field} must contain exactly: {', '.join(sorted(expected_fields))}"
        )
    return value


def config_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def config_identifier(value: object, field: str) -> str:
    identifier = config_text(value, field)
    if _SAFE_IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(f"{field} must be a safe identifier")
    return identifier


def config_sha256(value: object, field: str) -> str:
    result = config_text(value, field)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{field} must be lowercase hexadecimal")
    return result


@dataclass(frozen=True)
class HybridEvidenceArtifact:
    artifact_id: str
    source_path: Path
    role: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.artifact_id, "artifact_id"),
            (self.role, "role"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        object.__setattr__(self, "source_path", Path(self.source_path).expanduser())
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


def optional_config_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return config_text(value, field)


def config_path(value: object, field: str, base: Path) -> Path:
    text = config_text(value, field)
    lexical = (base / text).expanduser().absolute()
    require_no_symlink_components(lexical, field)
    return lexical.resolve()


def require_no_symlink_components(path: Path, field: str) -> None:
    """Reject a path whose existing leaf or parent redirects through a symlink."""

    candidate = Path(path).expanduser().absolute()
    while True:
        if candidate.is_symlink():
            raise ValueError(f"{field} must not contain a symlink: {candidate}")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def require_scoped_path(
    path: Path,
    roots: tuple[Path, ...],
    field: str,
    *,
    scope_description: str,
) -> Path:
    """Return the canonical path only when it stays under a declared root."""

    require_no_symlink_components(path, field)
    candidate = Path(path).expanduser().resolve()
    canonical_roots = tuple(Path(root).expanduser().resolve() for root in roots)
    if not any(
        candidate == root or candidate.is_relative_to(root)
        for root in canonical_roots
    ):
        raise ValueError(
            f"{field} must be repository-scoped under {scope_description}: "
            f"{candidate}"
        )
    return candidate


def validate_repository_scope(
    *,
    source_path: Path,
    evidence_root: Path,
    machine_profile_path: Path,
    repository_paths: Mapping[str, Path],
    config_paths: Mapping[str, Path],
    artifact_paths: tuple[Path, ...],
) -> None:
    """Bind one loaded evidence config to a single declared sibling workspace."""

    if set(repository_paths) != REPOSITORY_LABELS:
        raise ValueError(
            "repository_paths must contain exactly the four experiment repositories"
        )
    repositories = {
        label: Path(path).expanduser().resolve()
        for label, path in repository_paths.items()
    }
    if len(set(repositories.values())) != len(repositories):
        raise ValueError("repository_paths must identify distinct repository roots")
    excavator_root = repositories["excavator_il"]
    workspace_root = excavator_root.parent
    for label, root in repositories.items():
        require_no_symlink_components(root, f"repository_paths.{label}")
        if root.parent != workspace_root:
            raise ValueError(
                f"repository_paths.{label} must be a sibling under the declared "
                f"workspace root: {workspace_root}"
            )
    require_scoped_path(
        source_path,
        (excavator_root / "config",),
        "hybrid evidence config",
        scope_description="repository_paths.excavator_il/config",
    )
    require_scoped_path(
        evidence_root,
        (workspace_root,),
        "evidence_root",
        scope_description="the declared workspace root",
    )
    require_scoped_path(
        machine_profile_path,
        (workspace_root,),
        "machine_profile_path",
        scope_description="the declared workspace root",
    )
    repository_roots = tuple(repositories.values())
    for label, path in config_paths.items():
        require_scoped_path(
            path,
            repository_roots,
            f"config_paths.{label}",
            scope_description="a declared repository root",
        )
    for index, path in enumerate(artifact_paths):
        require_scoped_path(
            path,
            repository_roots,
            f"artifacts[{index}].source_path",
            scope_description="a declared repository root",
        )


def path_mapping(
    value: object,
    field: str,
    base: Path,
    *,
    expected_labels: frozenset[str] | None = None,
) -> dict[str, Path]:
    raw = strict_object(value, field, expected_fields=expected_labels)
    if not raw:
        raise ValueError(f"{field} must not be empty")
    return {
        config_text(label, f"{field} label"): config_path(
            path, f"{field}.{label}", base
        )
        for label, path in raw.items()
    }


def string_mapping(
    value: object,
    field: str,
    *,
    expected_labels: frozenset[str],
) -> dict[str, str]:
    raw = strict_object(value, field, expected_fields=expected_labels)
    return {
        label: config_text(raw[label], f"{field}.{label}")
        for label in sorted(expected_labels)
    }


def requirement(value: object, role: str) -> tuple[bool, int]:
    if isinstance(value, EvidenceRequirement):
        required, minimum = value.required, value.min_count
    elif isinstance(value, Mapping):
        required, minimum = value.get("required"), value.get("min_count")
    else:
        raise ValueError(f"evidence requirement {role} must be an object")
    if not isinstance(required, bool):
        raise ValueError(f"evidence requirement {role}.required must be boolean")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise ValueError(
            f"evidence requirement {role}.min_count must be non-negative"
        )
    return required, minimum


def required_count(value: object, role: str) -> int:
    required, minimum = requirement(value, role)
    if not required:
        raise ValueError(f"hybrid policy evidence role {role} must be required")
    if minimum < 1:
        raise ValueError(
            f"hybrid policy evidence role {role} must require at least one artifact"
        )
    return minimum


def trajectory_controller_family(policy_id: object) -> str:
    value = config_text(policy_id, "policy_ids.trajectory_controller")
    for family in ("onnx_rl", "cartesian_p"):
        if value.startswith(f"{family}:") and value != f"{family}:":
            return family
    raise ValueError(
        "policy_ids.trajectory_controller must identify onnx_rl or cartesian_p"
    )


def dig_policy_family(policy_id: object) -> str:
    value = config_text(policy_id, "policy_ids.dig_policy")
    for family in _DIG_POLICY_FAMILY_EVIDENCE_ROLES:
        if value.startswith(f"{family}:") and value != f"{family}:":
            return family
    raise ValueError(
        "policy_ids.dig_policy must identify lerobot_act or fixed_action"
    )


def fixed_action_profile_id(path: Path) -> str:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load fixed_action_profile {path}: {exc}") from exc
    profile = strict_object(parsed, "fixed_action_profile")
    return config_text(profile.get("profile_id"), "fixed_action_profile.profile_id")


def expected_policy_evidence_roles(
    *,
    dig_policy_family: str,
    trajectory_controller_family: str,
) -> tuple[str, ...]:
    return (
        *_DIG_POLICY_FAMILY_EVIDENCE_ROLES[dig_policy_family],
        *_TRAJECTORY_CONTROLLER_FAMILY_EVIDENCE_ROLES[
            trajectory_controller_family
        ],
    )

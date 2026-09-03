"""Versioned method identity for ICRA 2027 comparison experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from ._hybrid_experiment_contract import (
    config_path as _repository_path,
    require_scoped_path as _require_scoped_path,
)


ICRA2027_EXPERIMENT_PROFILE_SCHEMA_VERSION = (
    "excavator_icra2027_experiment_profile.v2"
)

_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "expected_mission_id",
        "expected_mission_sha256",
        "study_kind",
        "readiness",
        "reference_profile_id",
        "isolated_factor",
        "method_factors",
        "bindings",
        "required_metrics",
    }
)
_METHOD_FACTOR_FIELDS = frozenset(
    {
        "software_architecture",
        "target_selection",
        "trajectory_tracking",
        "task_policy",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "collection_ui_config",
        "resident_fixed_cycle_config",
        "hybrid_evidence_config",
    }
)
_STUDY_KINDS = frozenset({"live_mission", "component_benchmark"})
_READINESS_VALUES = frozenset({"commissioning", "planned", "ready"})
_METHOD_FACTOR_LEVELS = {
    "software_architecture": frozenset({"regime_factorized"}),
    "target_selection": frozenset({"fixed_catalog", "tadps"}),
    "trajectory_tracking": frozenset(
        {"cartesian_p", "not_applicable", "tc_btf"}
    ),
    "task_policy": frozenset(
        {
            "act_dig_lift_fixed_dump",
            "fixed_dig_fixed_dump",
            "not_applicable",
        }
    ),
}


def _strict_object(
    value: object,
    field: str,
    expected_fields: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    if set(value) != expected_fields:
        raise ValueError(
            f"{field} must contain exactly: "
            + ", ".join(sorted(expected_fields))
        )
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _optional_sha256(value: object, field: str) -> str | None:
    result = _optional_text(value, field)
    if result is not None and re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{field} must be lowercase hexadecimal")
    return result


def _optional_path(value: object, field: str, base: Path) -> Path | None:
    text = _optional_text(value, field)
    if text is None:
        return None
    return _repository_path(text, field, base)


def _choice(value: object, field: str, choices: frozenset[str]) -> str:
    text = _text(value, field)
    if text not in choices:
        raise ValueError(
            f"{field} must be one of: " + ", ".join(sorted(choices))
        )
    return text


@dataclass(frozen=True)
class Icra2027ExperimentProfile:
    """Immutable, reviewable identity of one experiment condition."""

    source_path: Path
    profile_id: str
    expected_mission_id: str | None
    expected_mission_sha256: str | None
    study_kind: str
    readiness: str
    reference_profile_id: str | None
    isolated_factor: str | None
    method_factors: Mapping[str, str]
    bindings: Mapping[str, Path | None]
    required_metrics: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Icra2027ExperimentProfile":
        source_path = _repository_path(
            str(path), "experiment profile", Path.cwd()
        )
        try:
            parsed = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load experiment profile {source_path}: {exc}"
            ) from exc
        root = _strict_object(parsed, "experiment profile", _PROFILE_FIELDS)
        if root["schema_version"] != ICRA2027_EXPERIMENT_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{ICRA2027_EXPERIMENT_PROFILE_SCHEMA_VERSION}"
            )
        factors = _strict_object(
            root["method_factors"], "method_factors", _METHOD_FACTOR_FIELDS
        )
        raw_bindings = _strict_object(
            root["bindings"], "bindings", _BINDING_FIELDS
        )
        raw_metrics = root["required_metrics"]
        if not isinstance(raw_metrics, list) or not raw_metrics:
            raise ValueError("required_metrics must be a non-empty array")
        metrics = tuple(
            _text(value, f"required_metrics[{index}]")
            for index, value in enumerate(raw_metrics)
        )
        if len(set(metrics)) != len(metrics):
            raise ValueError("required_metrics must not contain duplicates")
        base = source_path.parent
        return cls(
            source_path=source_path,
            profile_id=_text(root["profile_id"], "profile_id"),
            expected_mission_id=_optional_text(
                root["expected_mission_id"], "expected_mission_id"
            ),
            expected_mission_sha256=_optional_sha256(
                root["expected_mission_sha256"], "expected_mission_sha256"
            ),
            study_kind=_choice(root["study_kind"], "study_kind", _STUDY_KINDS),
            readiness=_choice(root["readiness"], "readiness", _READINESS_VALUES),
            reference_profile_id=_optional_text(
                root["reference_profile_id"], "reference_profile_id"
            ),
            isolated_factor=_optional_text(
                root["isolated_factor"], "isolated_factor"
            ),
            method_factors=MappingProxyType(
                {
                    name: _choice(
                        factors[name],
                        f"method_factors.{name}",
                        _METHOD_FACTOR_LEVELS[name],
                    )
                    for name in sorted(_METHOD_FACTOR_FIELDS)
                }
            ),
            bindings=MappingProxyType(
                {
                    name: _optional_path(
                        raw_bindings[name], f"bindings.{name}", base
                    )
                    for name in sorted(_BINDING_FIELDS)
                }
            ),
            required_metrics=metrics,
        )

    def preflight(self) -> None:
        """Reject non-runnable or incomplete profiles before hardware startup."""

        if self.readiness == "planned":
            raise ValueError("profile is planned, not runnable")
        for name, path in self.bindings.items():
            if path is None:
                raise ValueError(f"bindings.{name} is required for a ready profile")
            if not path.is_file():
                raise ValueError(f"bindings.{name} does not exist as a file: {path}")
        self._validate_runtime_binding()
        self._validate_evidence_binding()
        self._validate_collection_ui_binding()
        self._validate_repository_scope()

    def _validate_repository_scope(self) -> None:
        from .hybrid_experiment_run import HybridExperimentRunConfig

        evidence_path = self.bindings["hybrid_evidence_config"]
        if evidence_path is None:
            raise ValueError("hybrid_evidence_config binding is missing")
        evidence = HybridExperimentRunConfig.load(evidence_path)
        excavator_root = evidence.repository_paths.get("excavator_il")
        if excavator_root is None:
            raise ValueError(
                "hybrid evidence must declare repository_paths.excavator_il"
            )
        _require_scoped_path(
            self.source_path,
            (excavator_root / "config/experiments/icra2027",),
            "experiment profile",
            scope_description=(
                "repository_paths.excavator_il/config/experiments/icra2027"
            ),
        )
        for name, binding in self.bindings.items():
            if binding is None:
                continue
            _require_scoped_path(
                binding,
                (excavator_root / "config",),
                f"bindings.{name}",
                scope_description="repository_paths.excavator_il/config",
            )
        evidence.validate_repository_scope()

    def _validate_runtime_binding(self) -> None:
        from .resident_fixed_cycle_system import ResidentFixedCyclePcConfig

        runtime_path = self.bindings["resident_fixed_cycle_config"]
        if runtime_path is None:
            raise ValueError("resident_fixed_cycle_config binding is missing")
        runtime = ResidentFixedCyclePcConfig.load(runtime_path)
        expected = self.expected_mission_id
        if expected is None:
            raise ValueError(
                "expected_mission_id is required for a runnable profile"
            )
        if runtime.expected_mission_id != expected:
            raise ValueError(
                "resident expected_mission_id does not match experiment profile: "
                f"expected {expected}, got {runtime.expected_mission_id}"
            )
        expected_sha256 = self.expected_mission_sha256
        if expected_sha256 is None:
            raise ValueError(
                "expected_mission_sha256 is required for a runnable profile"
            )
        if runtime.expected_mission_sha256 != expected_sha256:
            raise ValueError(
                "resident expected_mission_sha256 does not match experiment profile"
            )
        expected_act_worker = (
            self.method_factors["task_policy"] != "fixed_dig_fixed_dump"
        )
        if runtime.expected_act_worker_required is not expected_act_worker:
            raise ValueError(
                "resident ACT worker requirement does not match experiment "
                f"task_policy: expected {expected_act_worker}, got "
                f"{runtime.expected_act_worker_required}"
            )
        expected_controller = {
            "tc_btf": "onnx_rl",
            "cartesian_p": "cartesian_p",
        }.get(self.method_factors["trajectory_tracking"])
        if expected_controller is None:
            raise ValueError(
                "trajectory_tracking has no runnable resident controller binding"
            )
        if runtime.trajectory_controller_backend != expected_controller:
            raise ValueError(
                "resident trajectory_controller_backend does not match "
                "experiment trajectory_tracking: "
                f"expected {expected_controller}, got "
                f"{runtime.trajectory_controller_backend}"
            )

    def _validate_evidence_binding(self) -> None:
        from .hybrid_experiment_run import HybridExperimentRunConfig
        from .resident_fixed_cycle_system import ResidentFixedCyclePcConfig

        evidence_path = self.bindings["hybrid_evidence_config"]
        runtime_path = self.bindings["resident_fixed_cycle_config"]
        if evidence_path is None:
            raise ValueError("hybrid_evidence_config binding is missing")
        if runtime_path is None:
            raise ValueError("resident_fixed_cycle_config binding is missing")
        evidence = HybridExperimentRunConfig.load(evidence_path)
        runtime = ResidentFixedCyclePcConfig.load(runtime_path)
        expected_scope = (
            "held_out_experiment"
            if self.readiness == "ready"
            else "training_internal"
        )
        if evidence.evaluation_scope != expected_scope:
            raise ValueError(
                f"{self.readiness} experiment profile requires "
                f"{expected_scope} evidence"
            )
        if evidence.mission_id != self.expected_mission_id:
            raise ValueError(
                "evidence mission_id does not match experiment profile: "
                f"expected {self.expected_mission_id}, got {evidence.mission_id}"
            )
        if evidence.mission_sha256 != self.expected_mission_sha256:
            raise ValueError(
                "evidence Mission digest does not match experiment profile"
            )
        bound_profile = evidence.config_paths.get("experiment_profile")
        if bound_profile is None or bound_profile.resolve() != self.source_path:
            raise ValueError(
                "evidence config experiment_profile must bind this profile"
            )
        expected_policy_prefix = {
            "tc_btf": "onnx_rl:",
            "cartesian_p": "cartesian_p:",
        }.get(self.method_factors["trajectory_tracking"])
        controller_policy = evidence.policy_ids["trajectory_controller"]
        if expected_policy_prefix is None or not controller_policy.startswith(
            expected_policy_prefix
        ):
            raise ValueError(
                "evidence trajectory_controller identity does not match "
                "experiment trajectory_tracking"
            )
        expected_dig_policy_prefix = (
            "fixed_action:"
            if self.method_factors["task_policy"] == "fixed_dig_fixed_dump"
            else "lerobot_act:"
        )
        if not evidence.policy_ids["dig_policy"].startswith(
            expected_dig_policy_prefix
        ):
            raise ValueError(
                "evidence dig_policy identity does not match experiment "
                "task_policy"
            )
        self._validate_edge_runtime_evidence(evidence, runtime)

    def _validate_collection_ui_binding(self) -> None:
        from .collection_ui_config import load_collection_ui_config

        ui_path = self.bindings["collection_ui_config"]
        evidence_path = self.bindings["hybrid_evidence_config"]
        runtime_path = self.bindings["resident_fixed_cycle_config"]
        if ui_path is None:
            raise ValueError("collection_ui_config binding is missing")
        if evidence_path is None:
            raise ValueError("hybrid_evidence_config binding is missing")
        if runtime_path is None:
            raise ValueError("resident_fixed_cycle_config binding is missing")
        ui = load_collection_ui_config(ui_path)
        if ui.resident_fixed_cycle_config != runtime_path:
            raise ValueError(
                "collection_ui_config resident_fixed_cycle_config must bind "
                "this profile runtime"
            )
        if ui.hybrid_evidence_config != evidence_path:
            raise ValueError(
                "collection_ui_config hybrid_evidence_config must bind this "
                "profile evidence"
            )

    @staticmethod
    def _validate_edge_runtime_evidence(evidence: Any, runtime: Any) -> None:
        repository = evidence.repository_paths.get("excavator_orin_runtime")
        bound_path = evidence.config_paths.get("edge_runtime")
        if repository is None or bound_path is None:
            raise ValueError(
                "evidence edge_runtime must bind the Orin controller config"
            )
        expected_path = repository.joinpath(
            *runtime.edge_runtime_config.parts
        ).resolve()
        if bound_path.resolve() != expected_path:
            raise ValueError(
                "evidence edge_runtime does not match the resident runtime: "
                f"expected {expected_path}, got {bound_path.resolve()}"
            )
        try:
            parsed = json.loads(expected_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load evidence edge_runtime config {expected_path}: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("evidence edge_runtime config must be an object")
        backend = parsed.get("trajectory_controller_backend")
        if backend != runtime.trajectory_controller_backend:
            raise ValueError(
                "evidence edge_runtime backend does not match the resident runtime: "
                f"expected {runtime.trajectory_controller_backend}, got {backend}"
            )


@dataclass(frozen=True)
class Icra2027ExperimentSuite:
    """A comparison matrix whose ablations declare exactly one changed factor."""

    root: Path
    profiles: Mapping[str, Icra2027ExperimentProfile]

    @classmethod
    def load_directory(cls, path: str | Path) -> "Icra2027ExperimentSuite":
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"experiment profile directory does not exist: {root}")
        loaded: dict[str, Icra2027ExperimentProfile] = {}
        for profile_path in sorted(root.glob("*.json")):
            profile = Icra2027ExperimentProfile.load(profile_path)
            if profile.profile_id != profile_path.stem:
                raise ValueError(
                    f"profile_id must match file name: {profile_path.name}"
                )
            if profile.profile_id in loaded:
                raise ValueError(f"duplicate profile_id: {profile.profile_id}")
            loaded[profile.profile_id] = profile
        if not loaded:
            raise ValueError("experiment profile directory contains no profiles")
        for profile in loaded.values():
            reference_id = profile.reference_profile_id
            if reference_id is None:
                if profile.isolated_factor is not None:
                    raise ValueError(
                        f"reference profile {profile.profile_id} "
                        "cannot isolate a factor"
                    )
                continue
            if reference_id not in loaded:
                raise ValueError(
                    f"profile {profile.profile_id} references unknown profile "
                    f"{reference_id}"
                )
            changed = {
                name
                for name in _METHOD_FACTOR_FIELDS
                if profile.method_factors[name]
                != loaded[reference_id].method_factors[name]
            }
            expected = (
                set()
                if profile.isolated_factor is None
                else {profile.isolated_factor}
            )
            if changed != expected:
                raise ValueError(
                    f"profile {profile.profile_id} changes factors "
                    f"{sorted(changed)} but declares {profile.isolated_factor}"
                )
        return cls(root=root, profiles=MappingProxyType(dict(sorted(loaded.items()))))


def _suite_report(
    suite_path: Path,
    require_ready: str | None,
) -> tuple[dict[str, Any], int]:
    try:
        suite = Icra2027ExperimentSuite.load_directory(suite_path)
        failures: list[str] = []
        for profile in suite.profiles.values():
            if profile.readiness == "planned":
                continue
            try:
                profile.preflight()
            except ValueError as exc:
                failures.append(f"profile {profile.profile_id}: {exc}")
        if require_ready is not None:
            required = suite.profiles.get(require_ready)
            if required is None:
                failures.append(f"unknown profile: {require_ready}")
            elif required.readiness != "ready":
                failures.append(
                    f"profile {require_ready} is {required.readiness}, not ready"
                )
        report = {
            "schema_version": "excavator_icra2027_experiment_suite_report.v1",
            "suite_root": str(suite.root),
            "profile_count": len(suite.profiles),
            "ready_profiles": [
                name
                for name, profile in suite.profiles.items()
                if profile.readiness == "ready"
            ],
            "planned_profiles": [
                name
                for name, profile in suite.profiles.items()
                if profile.readiness == "planned"
            ],
            "commissioning_profiles": [
                name
                for name, profile in suite.profiles.items()
                if profile.readiness == "commissioning"
            ],
            "profiles": {
                name: {
                    "study_kind": profile.study_kind,
                    "expected_mission_id": profile.expected_mission_id,
                    "expected_mission_sha256": profile.expected_mission_sha256,
                    "readiness": profile.readiness,
                    "reference_profile_id": profile.reference_profile_id,
                    "isolated_factor": profile.isolated_factor,
                    "method_factors": dict(profile.method_factors),
                    "required_metrics": list(profile.required_metrics),
                }
                for name, profile in suite.profiles.items()
            },
            "failure_reasons": failures,
            "passed": not failures,
        }
    except ValueError as exc:
        report = {
            "schema_version": "excavator_icra2027_experiment_suite_report.v1",
            "suite_root": str(suite_path.expanduser().resolve()),
            "profile_count": 0,
            "ready_profiles": [],
            "planned_profiles": [],
            "commissioning_profiles": [],
            "profiles": {},
            "failure_reasons": [str(exc)],
            "passed": False,
        }
    return report, 0 if report["passed"] else 2


def main(argv: list[str] | None = None) -> int:
    """Inspect the experiment matrix without starting hardware."""

    parser = argparse.ArgumentParser(
        description="Validate the ICRA 2027 experiment profile matrix."
    )
    parser.add_argument(
        "suite",
        nargs="?",
        default="config/experiments/icra2027",
        type=Path,
    )
    parser.add_argument("--require-ready", metavar="PROFILE_ID")
    arguments = parser.parse_args(argv)
    report, exit_code = _suite_report(
        arguments.suite,
        arguments.require_ready,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return exit_code

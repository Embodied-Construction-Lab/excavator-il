#!/usr/bin/env python3
"""Statically verify ACT dig-transport-dump reference assets on either PC or Orin.

The check opens configuration, checkpoint, manifest, and machine-profile files
only.  It never starts Docker, cameras, serial devices, or a Mission runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any

_REPOSITORY = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from excavator_il.act_runtime_config import load_act_runtime_config
from excavator_il.resident_fixed_cycle_system import ResidentFixedCyclePcConfig


ORIN_EXCAVATOR_IL_ROOT = PurePosixPath(
    "/home/jetson16/workspace_excavator/excavator-il"
)
RUNTIME_CONFIG = "act_runtime.icra2027_transport_dump_dual_rgb.orin.json"
RESIDENT_CONFIG = "resident_fixed_cycle.act_dig_transport_dump_reference.commissioning.pc.json"
EVIDENCE_MANIFEST = (
    "act_deployment.icra2027_transport_dump_dual_rgb_step115000.json"
)
COMMISSIONED_MODEL_DIRECTORY = (
    ORIN_EXCAVATOR_IL_ROOT
    / "models"
    / "icra2027_transport_dump_dual_rgb_step115000"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_asset_path(repository_root: Path, host_path: PurePosixPath) -> Path:
    try:
        relative = host_path.relative_to(ORIN_EXCAVATOR_IL_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"engineering-reference host path is outside {ORIN_EXCAVATOR_IL_ROOT}: {host_path}"
        ) from exc
    return repository_root.joinpath(*relative.parts)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_manifest(
    *, manifest_path: Path, checkpoint_path: Path, machine_profile_path: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "excavator_act_deployment.v2":
        raise ValueError("ACT dig-transport-dump reference deployment manifest schema is invalid")
    checkpoint = manifest.get("checkpoint")
    evaluation = manifest.get("evaluation")
    data = manifest.get("data")
    contract = manifest.get("contract")
    sections = (checkpoint, evaluation, data, contract)
    if not all(isinstance(item, dict) for item in sections):
        raise ValueError("ACT dig-transport-dump reference deployment manifest sections are invalid")
    if (
        checkpoint.get("selected") is not True
        or checkpoint.get("selection_reason")
        != "lowest safe validation deployment-prior L1"
    ):
        raise ValueError("ACT dig-transport-dump reference checkpoint is not validation-selected")
    file_hashes = checkpoint.get("files_sha256")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("ACT dig-transport-dump reference checkpoint hashes are invalid")
    actual_files = {path.name for path in checkpoint_path.iterdir() if path.is_file()}
    if actual_files != set(file_hashes):
        raise ValueError("ACT dig-transport-dump reference checkpoint file set mismatch")
    for name, expected in file_hashes.items():
        if Path(name).name != name or not _is_sha256(expected):
            raise ValueError("ACT dig-transport-dump reference checkpoint hash entry is invalid")
        if _sha256(checkpoint_path / name) != expected:
            raise ValueError(f"ACT dig-transport-dump reference checkpoint SHA-256 mismatch: {name}")
    validation_frames = evaluation.get("validation_frame_count")
    gross_count = evaluation.get("gross_out_of_range_sample_count")
    saturated_count = evaluation.get("saturated_value_count")
    if (
        evaluation.get("all_finite") is not True
        or isinstance(gross_count, bool)
        or not isinstance(gross_count, int)
        or gross_count != 0
        or isinstance(saturated_count, bool)
        or not isinstance(saturated_count, int)
        or saturated_count < 0
        or evaluation.get("max_tolerated_normalized_magnitude") != 1.25
        or not isinstance(validation_frames, int)
        or isinstance(validation_frames, bool)
        or validation_frames <= 0
    ):
        raise ValueError("ACT dig-transport-dump reference validation evidence is unsafe")
    try:
        prior_l1 = float(evaluation["deployment_prior_l1"])
        maximum_l1 = float(evaluation["max_deployment_prior_l1"])
        action_min = float(evaluation["action_min"])
        action_max = float(evaluation["action_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ACT dig-transport-dump reference validation evidence is invalid") from exc
    if not (
        all(
            math.isfinite(value)
            for value in (prior_l1, maximum_l1, action_min, action_max)
        )
        and 0.0 <= prior_l1 <= maximum_l1
        and -1.25 <= action_min <= action_max <= 1.25
    ):
        raise ValueError("ACT dig-transport-dump reference validation evidence is unsafe")
    if data.get("pipeline_validation_present") is not False or any(
        not _is_sha256(data.get(field))
        for field in (
            "train_dataset_sha256",
            "validation_dataset_sha256",
            "source_dataset_sha256",
        )
    ):
        raise ValueError("ACT dig-transport-dump reference dataset provenance is invalid")
    expected_contract = {
        "action_order": ["boom", "stick", "bucket", "swing"],
        "action_fields": [
            "action_boom",
            "action_stick",
            "action_bucket",
            "action_swing",
        ],
        "state_fields": [
            "boom_pos_m",
            "stick_pos_m",
            "bucket_pos_m",
            "boom_vel_mps",
            "stick_vel_mps",
            "bucket_vel_mps",
            "boom_angle_rad",
            "arm_angle_rad",
            "bucket_angle_rad",
            "swing_angle_rad",
            "swing_vel_radps",
        ],
        "action_dim": 4,
        "state_dim": 11,
        "front_rgb_chw": [3, 480, 640],
        "dump_rgb_chw": [3, 480, 640],
        "chunk_size": 20,
        "n_action_steps": 10,
        "temporal_ensemble_coeff": None,
        "input_feature_keys": [
            "observation.images.dump",
            "observation.images.front",
            "observation.state",
        ],
    }
    if any(
        contract.get(field) != expected
        for field, expected in expected_contract.items()
    ):
        raise ValueError("ACT dig-transport-dump reference observation/action contract is invalid")
    machine_profile = json.loads(machine_profile_path.read_text(encoding="utf-8"))
    if machine_profile.get("action_order") != expected_contract["action_order"]:
        raise ValueError("machine profile action order is invalid")
    if _sha256(machine_profile_path) != manifest.get("machine_profile_sha256"):
        raise ValueError("machine profile SHA-256 mismatch")
    return manifest


def inspect_assets(
    *, repository_root: Path, machine_profile_path: Path
) -> dict[str, Any]:
    repository = repository_root.expanduser().resolve()
    machine_profile = machine_profile_path.expanduser().resolve()
    config_root = repository / "config"
    runtime_path = config_root / RUNTIME_CONFIG
    resident_path = config_root / RESIDENT_CONFIG
    evidence_manifest_path = config_root / EVIDENCE_MANIFEST

    runtime = load_act_runtime_config(runtime_path)
    resident = ResidentFixedCyclePcConfig.load(resident_path)
    if runtime.camera_roles != ("front", "dump"):
        raise ValueError("ACT dig-transport-dump reference runtime must require front and dump RGB")
    if resident.expected_mission_id != "engineering_act_transport_reference":
        raise ValueError(
            "resident config is not the ACT dig-transport-dump reference Mission"
        )
    if resident.act_checkpoint_host_path is None:
        raise ValueError("resident config is missing the ACT checkpoint host path")
    if resident.act_deployment_host_path is None:
        raise ValueError("resident config is missing the ACT deployment host path")

    expected_checkpoint_host_path = COMMISSIONED_MODEL_DIRECTORY / "checkpoint"
    expected_deployment_host_path = COMMISSIONED_MODEL_DIRECTORY / "deployment"
    if resident.act_checkpoint_host_path != expected_checkpoint_host_path:
        raise ValueError(
            "resident ACT checkpoint host path is not the commissioned asset"
        )
    if resident.act_deployment_host_path != expected_deployment_host_path:
        raise ValueError(
            "resident ACT deployment host path is not the commissioned asset"
        )

    expected_runtime_host_path = ORIN_EXCAVATOR_IL_ROOT / "config" / RUNTIME_CONFIG
    if resident.act_runtime_config != expected_runtime_host_path:
        raise ValueError("resident ACT runtime config path is not canonical")
    checkpoint = _local_asset_path(
        repository, resident.act_checkpoint_host_path
    )
    deployment = _local_asset_path(
        repository, resident.act_deployment_host_path
    )
    deployment_manifest_path = deployment / "deployment_manifest.json"
    manifest = _verify_manifest(
        manifest_path=deployment_manifest_path,
        checkpoint_path=checkpoint,
        machine_profile_path=machine_profile,
    )
    manifest_hashes = manifest["checkpoint"]["files_sha256"]
    if dict(runtime.checkpoint_files_sha256) != manifest_hashes:
        raise ValueError("runtime checkpoint hashes do not match deployment manifest")
    if runtime.checkpoint_model_sha256 != manifest_hashes.get("model.safetensors"):
        raise ValueError("runtime model SHA-256 does not match deployment manifest")
    if runtime.checkpoint_path != Path("/opt/act-checkpoint"):
        raise ValueError("runtime checkpoint container path is not canonical")
    if runtime.deployment_manifest_path != Path(
        "/opt/act-deployment/deployment_manifest.json"
    ):
        raise ValueError("runtime deployment manifest container path is not canonical")
    if runtime.machine_profile_path != Path(
        "/opt/excavator-config/machine_profile.json"
    ):
        raise ValueError("runtime machine profile container path is not canonical")
    if deployment_manifest_path.read_bytes() != evidence_manifest_path.read_bytes():
        raise ValueError(
            "deployed manifest does not match the PC evidence manifest"
        )

    return {
        "schema_version": "excavator_act_dig_transport_dump_reference_asset_preflight.v1",
        "passed": True,
        "repository_root": str(repository),
        "machine_profile_path": str(machine_profile),
        "camera_roles": list(runtime.camera_roles),
        "checkpoint_path": str(checkpoint),
        "deployment_path": str(deployment),
        "checkpoint_file_count": len(manifest_hashes),
        "checkpoint_model_sha256": runtime.checkpoint_model_sha256,
        "deployment_manifest_sha256": _sha256(deployment_manifest_path),
        "machine_profile_sha256": _sha256(machine_profile),
        "deployment_manifest_matches_evidence": True,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=repository)
    parser.add_argument(
        "--machine-profile",
        type=Path,
        default=repository.parent / "shared/machine_profile.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = inspect_assets(
            repository_root=args.repository_root,
            machine_profile_path=args.machine_profile,
        )
    except (OSError, TypeError, ValueError) as exc:
        report = {
            "schema_version": "excavator_act_dig_transport_dump_reference_asset_preflight.v1",
            "passed": False,
            "failure_reasons": [str(exc)],
        }
        print(json.dumps(report, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

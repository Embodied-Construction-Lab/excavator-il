"""Strict deployment provenance verification for online ACT motion."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .checkpoint_evaluation import ACT_ACTION_ORDER, DEPLOYMENT_MANIFEST_SCHEMA_VERSION
from .lerobot_conversion import STATE_FIELDS
from .raw_episode import ACTION_FIELDS


TRAINING_LOSS_DEPLOYMENT_MANIFEST_SCHEMA_VERSION = "excavator_act_deployment.v3"
VALIDATION_SELECTION_REASON = "lowest safe validation deployment-prior L1"
TRAINING_LOSS_SELECTION_REASON = "operator-authorized lowest saved training loss"


def verify_deployment_manifest(
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    machine_profile_path: str | Path,
) -> dict[str, Any]:
    """Validate explicit checkpoint provenance and semantic contracts before motion."""

    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("ACT deployment manifest is unavailable or invalid") from exc
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        DEPLOYMENT_MANIFEST_SCHEMA_VERSION,
        TRAINING_LOSS_DEPLOYMENT_MANIFEST_SCHEMA_VERSION,
    }:
        raise ValueError("ACT deployment manifest schema is invalid")
    checkpoint = manifest.get("checkpoint")
    evaluation = manifest.get("evaluation")
    selection = manifest.get("selection")
    data = manifest.get("data")
    contract = manifest.get("contract")
    if not all(isinstance(item, dict) for item in (checkpoint, data, contract)):
        raise ValueError("ACT deployment manifest sections are invalid")
    if checkpoint.get("selected") is not True:
        raise ValueError("ACT deployment manifest checkpoint is not selected")
    selection_reason = checkpoint.get("selection_reason")
    if selection_reason == VALIDATION_SELECTION_REASON:
        if schema_version != DEPLOYMENT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("ACT deployment manifest schema is invalid")
        _verify_validation_evaluation(evaluation)
    elif selection_reason == TRAINING_LOSS_SELECTION_REASON:
        if schema_version != TRAINING_LOSS_DEPLOYMENT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("ACT deployment manifest schema is invalid")
        _verify_training_loss_selection(selection)
    else:
        raise ValueError("ACT deployment manifest selection reason is invalid")

    if data.get("pipeline_validation_present") is not False:
        raise ValueError("ACT deployment manifest contains pipeline-validation data")
    for field in (
        "train_dataset_sha256",
        "validation_dataset_sha256",
        "source_dataset_sha256",
    ):
        value = data.get(field)
        if not _is_sha256(value):
            raise ValueError(f"ACT deployment manifest {field} is invalid")
    if tuple(contract.get("action_order", ())) != ACT_ACTION_ORDER:
        raise ValueError("ACT deployment manifest action order is invalid")
    if tuple(contract.get("action_fields", ())) != ACTION_FIELDS:
        raise ValueError("ACT deployment manifest action fields are invalid")
    if tuple(contract.get("state_fields", ())) != STATE_FIELDS:
        raise ValueError("ACT deployment manifest state fields are invalid")
    expected_contract = {
        "state_dim": len(STATE_FIELDS),
        "action_dim": len(ACTION_FIELDS),
        "front_rgb_chw": [3, 480, 640],
        "chunk_size": 20,
        "n_action_steps": 10,
        "input_feature_keys": [
            "observation.images.front",
            "observation.state",
        ],
        "temporal_ensemble_coeff": None,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"ACT deployment manifest {field} is invalid")
    _verify_checkpoint_files(Path(checkpoint_path), checkpoint.get("files_sha256"))
    profile_path = Path(machine_profile_path)
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("machine profile is unavailable or invalid") from exc
    if tuple(profile.get("action_order", ())) != ACT_ACTION_ORDER:
        raise ValueError("machine profile action order is invalid")
    actual_profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    if actual_profile_sha != manifest.get("machine_profile_sha256"):
        raise ValueError("machine profile SHA-256 mismatch")
    return manifest


def _verify_validation_evaluation(evaluation: Any) -> None:
    if not isinstance(evaluation, dict):
        raise ValueError("ACT deployment manifest evaluation is invalid")
    if (
        isinstance(evaluation.get("validation_frame_count"), bool)
        or not isinstance(evaluation.get("validation_frame_count"), int)
        or evaluation["validation_frame_count"] <= 0
        or evaluation.get("all_finite") is not True
        or evaluation.get("out_of_range_sample_count") != 0
    ):
        raise ValueError("ACT deployment manifest evaluation is unsafe")
    try:
        l1 = float(evaluation["deployment_prior_l1"])
        action_min = float(evaluation["action_min"])
        action_max = float(evaluation["action_max"])
        max_l1 = float(evaluation["max_deployment_prior_l1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ACT deployment manifest evaluation is invalid") from exc
    if (
        not all(math.isfinite(value) for value in (l1, action_min, action_max, max_l1))
        or l1 < 0
        or max_l1 < 0
        or l1 > max_l1
        or action_min < -1.000001
        or action_max > 1.000001
        or action_min > action_max
    ):
        raise ValueError("ACT deployment manifest evaluation is unsafe")


def _verify_training_loss_selection(selection: Any) -> None:
    expected_fields = {
        "method",
        "checkpoint_step",
        "training_loss",
        "training_log_sha256",
        "validation_performed",
    }
    if not isinstance(selection, dict) or set(selection) != expected_fields:
        raise ValueError("ACT deployment manifest training-loss selection is invalid")
    step = selection.get("checkpoint_step")
    loss = selection.get("training_loss")
    if (
        selection.get("method") != "training_loss"
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(float(loss))
        or float(loss) < 0
        or not _is_sha256(selection.get("training_log_sha256"))
        or selection.get("validation_performed") is not False
    ):
        raise ValueError("ACT deployment manifest training-loss selection is invalid")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_checkpoint_files(root: Path, file_hashes: Any) -> None:
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("ACT deployment manifest checkpoint hashes are invalid")
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != set(file_hashes):
        raise ValueError("ACT deployment checkpoint file set mismatch")
    for name, expected in file_hashes.items():
        if Path(name).name != name or not _is_sha256(expected):
            raise ValueError("ACT deployment checkpoint hash entry is invalid")
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"ACT deployment checkpoint SHA-256 mismatch: {name}")

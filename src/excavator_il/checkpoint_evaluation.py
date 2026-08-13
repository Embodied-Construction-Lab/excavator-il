"""Offline validation and fail-closed selection for LeRobot ACT checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import packaging.version  # noqa: F401 - safetensors accesses packaging.version lazily
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors

from .act_smoke import _validate_excavator_act_contract
from .lerobot_conversion import STATE_FIELDS
from .raw_episode import ACTION_FIELDS
from .training_split import MATERIALIZED_SPLIT_SCHEMA_VERSION, _dataset_fingerprint


@dataclass(frozen=True)
class CheckpointValidationMetric:
    checkpoint_path: Path
    checkpoint_files_sha256: tuple[tuple[str, str], ...]
    validation_frame_count: int
    deployment_prior_l1: float
    action_min: float
    action_max: float
    all_finite: bool
    out_of_range_sample_count: int


@dataclass(frozen=True)
class CheckpointEvaluationResult:
    selected_checkpoint: Path | None
    selection_reason: str
    checkpoints: tuple[CheckpointValidationMetric, ...]
    split_root: Path
    split_provenance: tuple[tuple[str, str], ...]


DEPLOYMENT_MANIFEST_SCHEMA_VERSION = "excavator_act_deployment.v1"
ACT_ACTION_ORDER = ("boom", "stick", "bucket", "swing")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_file_hashes(checkpoint: Path) -> dict[str, str]:
    files = sorted(path for path in checkpoint.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"checkpoint contains no files: {checkpoint}")
    return {path.name: _sha256_file(path) for path in files}


def write_act_deployment_manifest(
    *,
    result: CheckpointEvaluationResult,
    split_root: str | Path,
    machine_profile_path: str | Path,
    output_path: str | Path,
    max_deployment_prior_l1: float,
) -> Path:
    """Atomically bind one evaluator-selected checkpoint to its full ACT contract."""

    selected = result.selected_checkpoint
    if selected is None:
        raise ValueError("cannot deploy when no checkpoint passed evaluation")
    checkpoint = selected.resolve()
    metric = next(
        (item for item in result.checkpoints if item.checkpoint_path.resolve() == checkpoint),
        None,
    )
    if (
        metric is None
        or not metric.all_finite
        or metric.out_of_range_sample_count != 0
        or not math.isfinite(metric.deployment_prior_l1)
    ):
        raise ValueError("selected checkpoint does not have a safe evaluation metric")
    if (
        not math.isfinite(max_deployment_prior_l1)
        or max_deployment_prior_l1 < 0
        or metric.deployment_prior_l1 > max_deployment_prior_l1
    ):
        raise ValueError("selected checkpoint deployment-prior L1 exceeds threshold")
    root = Path(split_root).resolve()
    provenance_path = root / "split_provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("materialized split provenance is unavailable") from exc
    if provenance.get("schema_version") != MATERIALIZED_SPLIT_SCHEMA_VERSION:
        raise ValueError("materialized split provenance schema is invalid")
    evaluated_provenance = dict(result.split_provenance)
    current_provenance = {
        field: provenance.get(field)
        for field in (
            "source_dataset_sha256",
            "train_repo_id",
            "validation_repo_id",
            "train_dataset_sha256",
            "validation_dataset_sha256",
        )
    }
    if root != result.split_root or current_provenance != evaluated_provenance:
        raise ValueError("deployment split is different from checkpoint evaluation")
    train_root = root / "train"
    validation_root = root / "validation"
    if any(
        (candidate / "pipeline_validation.json").exists()
        for candidate in (root, train_root, validation_root)
    ):
        raise ValueError("pipeline-validation data cannot produce a deployment manifest")
    if (
        _dataset_fingerprint(train_root) != provenance.get("train_dataset_sha256")
        or _dataset_fingerprint(validation_root)
        != provenance.get("validation_dataset_sha256")
    ):
        raise ValueError("materialized dataset fingerprint mismatch")
    profile_path = Path(machine_profile_path).resolve()
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("machine profile is unavailable or invalid") from exc
    if tuple(profile.get("action_order", ())) != ACT_ACTION_ORDER:
        raise ValueError("machine profile action order is not authoritative")
    try:
        policy_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("selected checkpoint config is unavailable or invalid") from exc
    current_hashes = _checkpoint_file_hashes(checkpoint)
    if current_hashes != dict(metric.checkpoint_files_sha256):
        raise ValueError("checkpoint changed since checkpoint evaluation")
    manifest = {
        "schema_version": DEPLOYMENT_MANIFEST_SCHEMA_VERSION,
        "checkpoint": {
            "path_at_evaluation": str(checkpoint),
            "selected": True,
            "selection_reason": result.selection_reason,
            "files_sha256": current_hashes,
        },
        "evaluation": {
            "validation_frame_count": metric.validation_frame_count,
            "deployment_prior_l1": metric.deployment_prior_l1,
            "action_min": metric.action_min,
            "action_max": metric.action_max,
            "all_finite": metric.all_finite,
            "out_of_range_sample_count": metric.out_of_range_sample_count,
            "max_deployment_prior_l1": max_deployment_prior_l1,
        },
        "data": {
            "pipeline_validation_present": False,
            "train_repo_id": provenance.get("train_repo_id"),
            "validation_repo_id": provenance.get("validation_repo_id"),
            "train_dataset_sha256": provenance.get("train_dataset_sha256"),
            "validation_dataset_sha256": provenance.get("validation_dataset_sha256"),
            "source_dataset_sha256": provenance.get("source_dataset_sha256"),
        },
        "contract": {
            "action_order": list(ACT_ACTION_ORDER),
            "action_fields": list(ACTION_FIELDS),
            "state_fields": list(STATE_FIELDS),
            "state_dim": len(STATE_FIELDS),
            "action_dim": len(ACTION_FIELDS),
            "front_rgb_chw": policy_config["input_features"][
                "observation.images.front"
            ]["shape"],
            "chunk_size": policy_config.get("chunk_size"),
            "n_action_steps": policy_config.get("n_action_steps"),
        },
        "machine_profile_sha256": _sha256_file(profile_path),
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def _evaluate_checkpoint(
    *,
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    device: str,
    batch_size: int,
    num_workers: int,
) -> CheckpointValidationMetric:
    initial_hashes = _checkpoint_file_hashes(checkpoint)
    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(checkpoint)
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )
    _validate_excavator_act_contract(policy.config, dataset)
    policy.to(device)
    policy.config.device = device
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    absolute_error_sum = 0.0
    valid_value_count = 0
    action_min = float("inf")
    action_max = float("-inf")
    out_of_range_sample_count = 0
    all_finite = True
    policy.eval()
    with torch.no_grad():
        for batch in loader:
            for camera_key in dataset.meta.camera_keys:
                if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                    batch[camera_key] = batch[camera_key].float() / 255.0
            raw_target = batch["action"].clone()
            raw_is_pad = batch["action_is_pad"].clone()
            processed = preprocessor(batch)
            predicted = policy.predict_action_chunk(processed)
            predicted_action = postprocessor(predicted)
            target = raw_target.to(predicted_action.device)
            valid_mask = ~raw_is_pad.to(predicted_action.device).unsqueeze(-1)
            valid_targets = target[valid_mask.expand_as(target)]
            if not bool(torch.isfinite(valid_targets).all().item()):
                raise ValueError("validation action labels contain non-finite values")
            if bool(((valid_targets < -1.000001) | (valid_targets > 1.000001)).any().item()):
                raise ValueError("validation action labels exceed [-1, 1]")
            absolute_error_sum += float(
                (torch.abs(predicted_action - target) * valid_mask).sum().item()
            )
            valid_value_count += int(valid_mask.sum().item()) * predicted_action.shape[-1]

            action_chunk = predicted_action
            finite_by_sample = torch.isfinite(action_chunk).flatten(1).all(dim=1)
            all_finite = all_finite and bool(finite_by_sample.all().item())
            if bool(finite_by_sample.any().item()):
                finite_values = action_chunk[finite_by_sample]
                action_min = min(action_min, float(finite_values.min().item()))
                action_max = max(action_max, float(finite_values.max().item()))
            unsafe = (~finite_by_sample) | (
                (action_chunk < -1.000001) | (action_chunk > 1.000001)
            ).flatten(1).any(dim=1)
            out_of_range_sample_count += int(unsafe.sum().item())

    if valid_value_count == 0:
        raise ValueError("validation dataset contains no valid ACT action labels")
    if _checkpoint_file_hashes(checkpoint) != initial_hashes:
        raise ValueError(f"checkpoint changed during evaluation: {checkpoint}")
    return CheckpointValidationMetric(
        checkpoint_path=checkpoint,
        checkpoint_files_sha256=tuple(initial_hashes.items()),
        validation_frame_count=dataset.num_frames,
        deployment_prior_l1=absolute_error_sum / valid_value_count,
        action_min=action_min,
        action_max=action_max,
        all_finite=all_finite,
        out_of_range_sample_count=out_of_range_sample_count,
    )


def evaluate_act_checkpoints(
    *,
    checkpoint_paths: list[str | Path],
    split_root: str | Path,
    device: str = "cpu",
    batch_size: int = 4,
    num_workers: int = 0,
) -> CheckpointEvaluationResult:
    """Evaluate checkpoints on held-out Episodes and select the lowest safe L1."""
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    root = Path(split_root).resolve()
    provenance_path = root / "split_provenance.json"
    if not provenance_path.is_file():
        raise ValueError("materialized training split provenance is missing")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("materialized training split provenance is invalid") from exc
    if provenance.get("schema_version") != MATERIALIZED_SPLIT_SCHEMA_VERSION:
        raise ValueError("materialized training split provenance schema is invalid")
    train_root = root / "train"
    validation_root = root / "validation"
    if (validation_root / "pipeline_validation.json").exists():
        raise ValueError("pipeline-validation dataset cannot select checkpoints")
    if _dataset_fingerprint(train_root) != provenance.get("train_dataset_sha256"):
        raise ValueError("materialized train dataset fingerprint mismatch")
    if _dataset_fingerprint(validation_root) != provenance.get("validation_dataset_sha256"):
        raise ValueError("materialized validation dataset fingerprint mismatch")
    train_repo_id = provenance.get("train_repo_id")
    validation_repo_id = provenance.get("validation_repo_id")
    if not isinstance(train_repo_id, str) or not isinstance(validation_repo_id, str):
        raise ValueError("materialized training split repo IDs are invalid")
    checkpoints = [Path(path) for path in checkpoint_paths]
    missing = [path for path in checkpoints if not path.is_dir()]
    if missing:
        raise ValueError(f"checkpoint does not exist: {missing[0]}")
    for checkpoint in checkpoints:
        config_path = checkpoint / "train_config.json"
        if not config_path.is_file():
            raise ValueError(f"checkpoint training provenance is missing: {checkpoint}")
        try:
            dataset_config = json.loads(config_path.read_text(encoding="utf-8"))["dataset"]
            checkpoint_root = Path(dataset_config["root"]).resolve()
            checkpoint_repo_id = dataset_config["repo_id"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"checkpoint training provenance is invalid: {checkpoint}") from exc
        if checkpoint_root != train_root or checkpoint_repo_id != train_repo_id:
            raise ValueError(
                f"checkpoint was not trained on the materialized train split: {checkpoint}"
            )

    metrics = tuple(
        _evaluate_checkpoint(
            checkpoint=checkpoint,
            dataset_root=validation_root,
            repo_id=validation_repo_id,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for checkpoint in checkpoints
    )
    if (
        _dataset_fingerprint(train_root) != provenance.get("train_dataset_sha256")
        or _dataset_fingerprint(validation_root)
        != provenance.get("validation_dataset_sha256")
    ):
        raise ValueError("materialized datasets changed during checkpoint evaluation")
    safe = [
        metric
        for metric in metrics
        if metric.all_finite
        and metric.out_of_range_sample_count == 0
        and math.isfinite(metric.deployment_prior_l1)
        and math.isfinite(metric.action_min)
        and math.isfinite(metric.action_max)
    ]
    selected = min(safe, key=lambda metric: metric.deployment_prior_l1) if safe else None
    return CheckpointEvaluationResult(
        selected_checkpoint=None if selected is None else selected.checkpoint_path,
        selection_reason=(
            "lowest safe validation deployment-prior L1"
            if selected is not None
            else "no checkpoint passed finite-action and normalized-range gates"
        ),
        checkpoints=metrics,
        split_root=root,
        split_provenance=tuple(
            (field, str(provenance.get(field)))
            for field in (
                "source_dataset_sha256",
                "train_repo_id",
                "validation_repo_id",
                "train_dataset_sha256",
                "validation_dataset_sha256",
            )
        ),
    )

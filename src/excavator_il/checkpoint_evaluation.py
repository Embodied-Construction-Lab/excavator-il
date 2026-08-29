"""Offline validation and fail-closed selection for LeRobot ACT checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import packaging.version  # noqa: F401 - safetensors accesses packaging.version lazily
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors

from .act_smoke import _validate_excavator_act_contract
from .dig_policy import MAX_TOLERATED_NORMALIZED_MAGNITUDE
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
    gross_out_of_range_sample_count: int = 0
    saturated_value_count: int = 0


@dataclass(frozen=True)
class CheckpointEvaluationResult:
    selected_checkpoint: Path | None
    selection_reason: str
    checkpoints: tuple[CheckpointValidationMetric, ...]
    split_root: Path
    split_provenance: tuple[tuple[str, str], ...]


DEPLOYMENT_MANIFEST_SCHEMA_VERSION = "excavator_act_deployment.v2"
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


def _policy_input_batch(
    row: Mapping[str, Any], input_features: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    batch = {
        key: (
            value.float().div(255.0)
            if key.startswith("observation.images.") and value.dtype == torch.uint8
            else value
        ).unsqueeze(0)
        for key, value in row.items()
        if key in input_features and isinstance(value, torch.Tensor)
    }
    missing = set(input_features) - set(batch)
    if missing:
        raise ValueError(
            f"validation sample is missing ACT input features: {', '.join(sorted(missing))}"
        )
    return batch


def _score_runtime_selected_actions(
    *,
    policy: Any,
    dataset: Any,
    preprocessor: Any,
    postprocessor: Any,
) -> dict[str, Any]:
    """Replay validation frames with LeRobot's select_action queue semantics."""

    episode_indices = dataset.hf_dataset["episode_index"]
    if len(episode_indices) != dataset.num_frames:
        raise ValueError("validation dataset episode indices are incomplete")
    policy.eval()
    previous_episode_index: int | None = None
    reset_count = 0
    absolute_error_sum = 0.0
    valid_value_count = 0
    action_min = float("inf")
    action_max = float("-inf")
    out_of_range_sample_count = 0
    gross_out_of_range_sample_count = 0
    saturated_value_count = 0
    all_finite = True
    with torch.no_grad():
        for frame_index in range(dataset.num_frames):
            episode_index = int(episode_indices[frame_index])
            if previous_episode_index != episode_index:
                policy.reset()
                reset_count += 1
                previous_episode_index = episode_index
            row = dataset[frame_index]
            batch = _policy_input_batch(row, policy.config.input_features)
            processed = preprocessor(batch)
            predicted = postprocessor(policy.select_action(processed))
            target = _selected_runtime_target(row).to(predicted.device).reshape_as(
                predicted
            )
            if not bool(torch.isfinite(target).all().item()):
                raise ValueError("validation action labels contain non-finite values")
            if bool(((target < -1.000001) | (target > 1.000001)).any().item()):
                raise ValueError("validation action labels exceed [-1, 1]")
            absolute_error_sum += float(torch.abs(predicted - target).sum().item())
            valid_value_count += int(target.numel())
            finite = bool(torch.isfinite(predicted).all().item())
            in_range = bool(
                ((predicted >= -1.000001) & (predicted <= 1.000001)).all().item()
            )
            within_tolerance = bool(
                (
                    torch.abs(predicted)
                    <= MAX_TOLERATED_NORMALIZED_MAGNITUDE
                ).all().item()
            )
            if finite:
                action_min = min(action_min, float(predicted.min().item()))
                action_max = max(action_max, float(predicted.max().item()))
            all_finite = all_finite and finite
            if not finite or not in_range:
                out_of_range_sample_count += 1
            if finite:
                saturated_value_count += int(
                    (torch.abs(predicted) > 1.000001).sum().item()
                )
                if not within_tolerance:
                    gross_out_of_range_sample_count += 1
    if valid_value_count == 0:
        raise ValueError("validation dataset contains no valid ACT action labels")
    return {
        "validation_frame_count": dataset.num_frames,
        "deployment_prior_l1": absolute_error_sum / valid_value_count,
        "action_min": action_min,
        "action_max": action_max,
        "all_finite": all_finite,
        "out_of_range_sample_count": out_of_range_sample_count,
        "gross_out_of_range_sample_count": gross_out_of_range_sample_count,
        "saturated_value_count": saturated_value_count,
        "reset_count": reset_count,
    }


def _selected_runtime_target(row: Mapping[str, Any]) -> torch.Tensor:
    """Extract the causal step-0 label that matches runtime select_action."""

    action = row.get("action")
    if not isinstance(action, torch.Tensor):
        raise ValueError("validation action labels must be tensors")
    if action.ndim == 1:
        return action
    if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] <= 0:
        raise ValueError(
            "validation action labels must be shaped [action_dim] or [chunk, action_dim]"
        )
    action_is_pad = row.get("action_is_pad")
    if action_is_pad is not None:
        if not isinstance(action_is_pad, torch.Tensor):
            raise ValueError("validation action_is_pad must be a tensor")
        flattened_pad = action_is_pad.reshape(-1)
        if flattened_pad.shape[0] != action.shape[0]:
            raise ValueError(
                "validation action_is_pad does not align with action labels"
            )
        if bool(flattened_pad[0].item()):
            raise ValueError("validation selected action is padded")
    return action[0]


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
        or not _metric_is_deployable(metric)
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
    input_features = policy_config.get("input_features", {})
    contract = {
        "action_order": list(ACT_ACTION_ORDER),
        "action_fields": list(ACTION_FIELDS),
        "state_fields": list(STATE_FIELDS),
        "state_dim": len(STATE_FIELDS),
        "action_dim": len(ACTION_FIELDS),
        "front_rgb_chw": input_features["observation.images.front"]["shape"],
        "chunk_size": policy_config.get("chunk_size"),
        "n_action_steps": policy_config.get("n_action_steps"),
        "input_feature_keys": sorted(input_features),
        "temporal_ensemble_coeff": policy_config.get("temporal_ensemble_coeff"),
    }
    if "observation.images.dump" in input_features:
        contract["dump_rgb_chw"] = input_features["observation.images.dump"]["shape"]
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
            "gross_out_of_range_sample_count": (
                metric.gross_out_of_range_sample_count
            ),
            "saturated_value_count": metric.saturated_value_count,
            "max_tolerated_normalized_magnitude": (
                MAX_TOLERATED_NORMALIZED_MAGNITUDE
            ),
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
        "contract": contract,
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
    replay_metrics = _score_runtime_selected_actions(
        policy=policy,
        dataset=dataset,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    if _checkpoint_file_hashes(checkpoint) != initial_hashes:
        raise ValueError(f"checkpoint changed during evaluation: {checkpoint}")
    return CheckpointValidationMetric(
        checkpoint_path=checkpoint,
        checkpoint_files_sha256=tuple(initial_hashes.items()),
        validation_frame_count=int(replay_metrics["validation_frame_count"]),
        deployment_prior_l1=float(replay_metrics["deployment_prior_l1"]),
        action_min=float(replay_metrics["action_min"]),
        action_max=float(replay_metrics["action_max"]),
        all_finite=bool(replay_metrics["all_finite"]),
        out_of_range_sample_count=int(replay_metrics["out_of_range_sample_count"]),
        gross_out_of_range_sample_count=int(
            replay_metrics["gross_out_of_range_sample_count"]
        ),
        saturated_value_count=int(replay_metrics["saturated_value_count"]),
    )


def _metric_is_deployable(metric: CheckpointValidationMetric) -> bool:
    return (
        metric.all_finite
        and metric.gross_out_of_range_sample_count == 0
        and math.isfinite(metric.deployment_prior_l1)
        and math.isfinite(metric.action_min)
        and math.isfinite(metric.action_max)
        and metric.action_min >= -MAX_TOLERATED_NORMALIZED_MAGNITUDE
        and metric.action_max <= MAX_TOLERATED_NORMALIZED_MAGNITUDE
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
        if _metric_is_deployable(metric)
    ]
    selected = min(safe, key=lambda metric: metric.deployment_prior_l1) if safe else None
    return CheckpointEvaluationResult(
        selected_checkpoint=None if selected is None else selected.checkpoint_path,
        selection_reason=(
            "lowest safe validation deployment-prior L1"
            if selected is not None
            else "no checkpoint passed finite-action and bounded-saturation gates"
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

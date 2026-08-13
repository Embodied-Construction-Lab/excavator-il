"""Offline ACT compatibility checks for the excavator feature contract."""

from dataclasses import dataclass
import math
from pathlib import Path
import time

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from .lerobot_conversion import STATE_FIELDS
from .raw_episode import ACTION_FIELDS


@dataclass(frozen=True)
class ActSmokeResult:
    loss: float
    predicted_chunk_shape: tuple[int, ...]


@dataclass(frozen=True)
class ActCheckpointInferenceResult:
    checkpoint_path: Path
    dataset_root: Path
    sample_index: int
    predicted_chunk_shape: tuple[int, ...]
    action_dim: int
    action_min: float
    action_max: float
    all_finite: bool
    inference_ms: float
    inference_min_ms: float
    inference_max_ms: float
    timed_runs: int
    peak_cuda_memory_mb: float | None


def _validate_excavator_act_contract(config: object, dataset: LeRobotDataset) -> None:
    if config.chunk_size != 20 or config.n_action_steps != 10:
        raise ValueError("ACT checkpoint must use chunk_size=20 and n_action_steps=10")
    if getattr(config, "temporal_ensemble_coeff", None) is not None:
        raise ValueError("ACT checkpoint must disable temporal ensemble")
    required_inputs = {
        "observation.state": (len(STATE_FIELDS),),
        "observation.images.front": None,
    }
    if set(config.input_features) != set(required_inputs):
        raise ValueError("ACT checkpoint must use exactly one front RGB camera")
    for key, expected_shape in required_inputs.items():
        if key not in config.input_features:
            raise ValueError(f"ACT checkpoint is missing required input feature: {key}")
        shape = tuple(config.input_features[key].shape)
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(
                f"ACT checkpoint state shape {shape} does not match {expected_shape}"
            )
        if expected_shape is None and (len(shape) != 3 or shape[0] != 3):
            raise ValueError(f"ACT checkpoint front RGB shape is invalid: {shape}")
    output = config.output_features.get("action")
    if output is None or tuple(output.shape) != (len(ACTION_FIELDS),):
        raise ValueError("ACT checkpoint must use the authoritative four-axis action")

    features = dataset.features
    required_dataset_features = {*required_inputs, "action"}
    missing_dataset_features = required_dataset_features - set(features)
    if missing_dataset_features:
        raise ValueError(
            "dataset is missing required feature: "
            + ", ".join(sorted(missing_dataset_features))
        )
    if tuple(features["observation.state"]["shape"]) != (len(STATE_FIELDS),):
        raise ValueError("dataset state shape does not match the 11-dimensional contract")
    if tuple(features["observation.state"].get("names") or ()) != STATE_FIELDS:
        raise ValueError("dataset state names do not match the authoritative contract")
    if tuple(features["action"]["shape"]) != (len(ACTION_FIELDS),):
        raise ValueError("dataset action shape does not match the four-axis contract")
    if tuple(features["action"].get("names") or ()) != ACTION_FIELDS:
        raise ValueError("dataset action names do not match [boom, stick, bucket, swing]")
    image_shape = tuple(features["observation.images.front"]["shape"])
    policy_image_shape = tuple(config.input_features["observation.images.front"].shape)
    if len(image_shape) != 3 or image_shape[2] != 3:
        raise ValueError(f"dataset front RGB shape is invalid: {image_shape}")
    if policy_image_shape != (3, image_shape[0], image_shape[1]):
        raise ValueError("checkpoint and dataset front RGB shapes do not match")


def _summarize_action_chunks(action_chunks: list[torch.Tensor]) -> tuple[float, float]:
    if not action_chunks or not all(
        bool(torch.isfinite(chunk).all().item()) for chunk in action_chunks
    ):
        raise ValueError("ACT checkpoint inference produced non-finite actions")
    action_range = (
        min(float(chunk.min().item()) for chunk in action_chunks),
        max(float(chunk.max().item()) for chunk in action_chunks),
    )
    if action_range[0] < -1.000001 or action_range[1] > 1.000001:
        raise ValueError(f"ACT checkpoint inference action range {action_range} exceeds [-1, 1]")
    return action_range


def _enforce_inference_budget(
    inference_max_ms: float, max_inference_ms: float | None
) -> None:
    if max_inference_ms is not None:
        if not math.isfinite(max_inference_ms) or max_inference_ms <= 0:
            raise ValueError("maximum inference time must be positive")
        if inference_max_ms > max_inference_ms:
            raise ValueError(
                f"ACT inference {inference_max_ms:.3f} ms exceeds "
                f"{max_inference_ms:.3f} ms budget"
            )


def run_act_smoke_train_step(
    *,
    image_shape: tuple[int, int, int],
    state_dim: int,
    action_dim: int,
    chunk_size: int,
) -> ActSmokeResult:
    """Run one optimizer step and one inference pass using the production feature shapes."""
    torch.manual_seed(0)
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        device="cpu",
        push_to_hub=False,
        chunk_size=chunk_size,
        n_action_steps=1,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=16,
        n_vae_encoder_layers=1,
    )
    policy = ACTPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    training_batch = {
        "observation.state": torch.randn(2, state_dim),
        "observation.images.front": torch.rand(2, *image_shape),
        "action": torch.empty(2, chunk_size, action_dim).uniform_(-1.0, 1.0),
        "action_is_pad": torch.zeros(2, chunk_size, dtype=torch.bool),
    }
    policy.train()
    loss, _ = policy(training_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    inference_batch = {
        "observation.state": torch.randn(1, state_dim),
        "observation.images.front": torch.rand(1, *image_shape),
    }
    with torch.no_grad():
        predicted_chunk = policy.predict_action_chunk(inference_batch)

    return ActSmokeResult(
        loss=float(loss.detach()),
        predicted_chunk_shape=tuple(predicted_chunk.shape),
    )


def run_act_checkpoint_inference(
    *,
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    repo_id: str,
    sample_index: int = 0,
    device: str = "cpu",
    warmup_runs: int = 0,
    timed_runs: int = 1,
    max_inference_ms: float | None = None,
) -> ActCheckpointInferenceResult:
    """Reload one ACT checkpoint and infer on one real LeRobotDataset sample."""
    checkpoint = Path(checkpoint_path)
    root = Path(dataset_root)
    if not checkpoint.is_dir():
        raise ValueError(f"ACT checkpoint does not exist: {checkpoint}")
    if not root.is_dir():
        raise ValueError(f"LeRobotDataset root does not exist: {root}")
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("warmup runs must be non-negative and timed runs must be positive")
    _enforce_inference_budget(0.0, max_inference_ms)
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    if sample_index < 0 or sample_index >= dataset.num_frames:
        raise ValueError(
            f"sample index {sample_index} is outside dataset with {dataset.num_frames} frames"
        )

    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(checkpoint)
    _validate_excavator_act_contract(policy.config, dataset)
    policy.to(device)
    policy.config.device = device
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    row = dataset[sample_index]
    batch = {
        key: value.unsqueeze(0)
        for key, value in row.items()
        if key in policy.config.input_features and isinstance(value, torch.Tensor)
    }
    missing = set(policy.config.input_features) - set(batch)
    if missing:
        raise ValueError(
            f"dataset sample is missing ACT input features: {', '.join(sorted(missing))}"
        )
    policy.reset()
    policy.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            processed = preprocessor(batch)
            policy.predict_action_chunk(processed)
            policy.reset()
    policy.reset()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations_ms: list[float] = []
    action_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(timed_runs):
            started = time.perf_counter()
            processed = preprocessor(batch)
            normalized_chunk = policy.predict_action_chunk(processed)
            action_chunk = postprocessor(normalized_chunk)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            durations_ms.append((time.perf_counter() - started) * 1000.0)
            action_chunks.append(action_chunk)
            policy.reset()
    peak_cuda_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.startswith("cuda")
        else None
    )
    action_dim = policy.config.output_features["action"].shape[0]
    expected_shape = (1, policy.config.chunk_size, action_dim)
    if tuple(action_chunk.shape) != expected_shape:
        raise ValueError(
            f"ACT action chunk shape {tuple(action_chunk.shape)} does not match {expected_shape}"
        )
    action_min, action_max = _summarize_action_chunks(action_chunks)
    _enforce_inference_budget(max(durations_ms), max_inference_ms)
    return ActCheckpointInferenceResult(
        checkpoint_path=checkpoint,
        dataset_root=root,
        sample_index=sample_index,
        predicted_chunk_shape=tuple(action_chunk.shape),
        action_dim=action_dim,
        action_min=action_min,
        action_max=action_max,
        all_finite=True,
        inference_ms=sum(durations_ms) / len(durations_ms),
        inference_min_ms=min(durations_ms),
        inference_max_ms=max(durations_ms),
        timed_runs=timed_runs,
        peak_cuda_memory_mb=peak_cuda_memory_mb,
    )

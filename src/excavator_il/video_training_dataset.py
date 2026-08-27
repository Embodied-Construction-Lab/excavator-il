"""Derive a provenance-bound video representation for ACT training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from .training_split import (
    MATERIALIZED_SPLIT_SCHEMA_VERSION,
    _atomic_write_json,
    _dataset_fingerprint,
)


VIDEO_TRAINING_DERIVATION_SCHEMA_VERSION = (
    "excavator_video_training_derivation.v1"
)
_PARTITIONS = ("train", "validation")
_CODEC = {
    "vcodec": "h264",
    "pix_fmt": "yuv420p",
    "g": 2,
    "crf": 18,
    "fast_decode": 1,
}


@dataclass(frozen=True)
class VideoTrainingDerivation:
    output_root: Path
    train_dataset_sha256: str
    validation_dataset_sha256: str


def _load_dataset(repo_id: str, *, root: Path) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=root)


def _convert_dataset(**kwargs: Any) -> Any:
    from lerobot.datasets.dataset_tools import convert_image_to_video_dataset

    return convert_image_to_video_dataset(**kwargs)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _validate_source(
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    provenance = _read_json(
        source_root / "split_provenance.json", "split provenance"
    )
    if provenance.get("schema_version") != MATERIALIZED_SPLIT_SCHEMA_VERSION:
        raise ValueError("split provenance schema is invalid")
    fingerprints: dict[str, str] = {}
    for partition in _PARTITIONS:
        partition_root = source_root / partition
        current = _dataset_fingerprint(partition_root)
        if current != provenance.get(f"{partition}_dataset_sha256"):
            raise ValueError(f"{partition} dataset fingerprint mismatch")
        fingerprints[partition] = current
    return provenance, fingerprints


def _validate_video_partition(
    root: Path,
    *,
    expected_episodes: int,
    expected_frames: int,
) -> None:
    info = _read_json(root / "meta" / "info.json", "video dataset metadata")
    if info.get("total_episodes") != expected_episodes:
        raise ValueError("video dataset Episode count changed")
    if info.get("total_frames") != expected_frames:
        raise ValueError("video dataset frame count changed")
    if not isinstance(info.get("video_path"), str) or not info["video_path"]:
        raise ValueError("video dataset has no video path")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("video dataset features are invalid")
    camera_features = {
        name: feature
        for name, feature in features.items()
        if name.startswith("observation.images.")
    }
    if not camera_features or any(
        not isinstance(feature, dict) or feature.get("dtype") != "video"
        for feature in camera_features.values()
    ):
        raise ValueError("derived camera features must all be videos")
    stats = _read_json(root / "meta" / "stats.json", "video dataset stats")
    if not camera_features.keys() <= stats.keys():
        raise ValueError("video dataset camera stats are incomplete")


def _restore_camera_stats(source_root: Path, output_root: Path) -> None:
    """Restore stats dropped by LeRobot's image-to-video converter."""

    source_info = _read_json(
        source_root / "meta" / "info.json", "source dataset metadata"
    )
    source_stats = _read_json(
        source_root / "meta" / "stats.json", "source dataset stats"
    )
    output_stats = _read_json(
        output_root / "meta" / "stats.json", "converted dataset stats"
    )
    features = source_info.get("features")
    if not isinstance(features, dict):
        raise ValueError("source dataset features are invalid")
    camera_keys = {
        name
        for name in features
        if name.startswith("observation.images.")
    }
    if not camera_keys or not camera_keys <= source_stats.keys():
        raise ValueError("source dataset camera stats are incomplete")
    merged_stats = {
        **output_stats,
        **{name: source_stats[name] for name in sorted(camera_keys)},
    }
    _atomic_write_json(output_root / "meta" / "stats.json", merged_stats)


def derive_video_training_split(
    source_split_root: str | Path,
    output_split_root: str | Path,
    *,
    dataset_loader: Callable[..., Any] = _load_dataset,
    converter: Callable[..., Any] = _convert_dataset,
    num_workers: int = 2,
    max_frames_per_batch: int = 3000,
) -> VideoTrainingDerivation:
    """Convert both split partitions without changing Episode membership."""

    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise ValueError("num_workers must be a positive integer")
    if num_workers <= 0:
        raise ValueError("num_workers must be a positive integer")
    if (
        isinstance(max_frames_per_batch, bool)
        or not isinstance(max_frames_per_batch, int)
        or max_frames_per_batch <= 0
    ):
        raise ValueError("max_frames_per_batch must be a positive integer")
    source_root = Path(source_split_root).resolve()
    output_root = Path(output_split_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"output split already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    provenance, source_fingerprints = _validate_source(source_root)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    output_fingerprints: dict[str, str] = {}
    repo_ids: dict[str, str] = {}
    try:
        for partition in _PARTITIONS:
            source_partition = source_root / partition
            source_repo_id = provenance.get(f"{partition}_repo_id")
            if not isinstance(source_repo_id, str) or not source_repo_id:
                raise ValueError(f"{partition} repo ID is invalid")
            dataset = dataset_loader(source_repo_id, root=source_partition)
            if getattr(dataset.meta, "video_keys", None):
                raise ValueError("source split is already video-backed")
            output_repo_id = f"{source_repo_id}_video"
            output_partition = staging / partition
            converter(
                dataset=dataset,
                output_dir=output_partition,
                repo_id=output_repo_id,
                **_CODEC,
                num_workers=num_workers,
                max_frames_per_batch=max_frames_per_batch,
            )
            _restore_camera_stats(source_partition, output_partition)
            _validate_video_partition(
                output_partition,
                expected_episodes=int(dataset.meta.total_episodes),
                expected_frames=int(dataset.meta.total_frames),
            )
            repo_ids[partition] = output_repo_id
            output_fingerprints[partition] = _dataset_fingerprint(
                output_partition
            )

        output_provenance = {
            **provenance,
            "train_repo_id": repo_ids["train"],
            "validation_repo_id": repo_ids["validation"],
            "train_dataset_sha256": output_fingerprints["train"],
            "validation_dataset_sha256": output_fingerprints["validation"],
        }
        _atomic_write_json(staging / "split_provenance.json", output_provenance)
        transform_path = source_root / "action_transform_provenance.json"
        if transform_path.is_file() and not transform_path.is_symlink():
            shutil.copyfile(
                transform_path, staging / "action_transform_provenance.json"
            )
        derivation = {
            "schema_version": VIDEO_TRAINING_DERIVATION_SCHEMA_VERSION,
            "source_split_root": str(source_root),
            "source_partition_sha256": source_fingerprints,
            "output_partition_sha256": output_fingerprints,
            "codec": dict(_CODEC),
            "camera_stats": {
                "method": "inherited_source_image_stats",
                "training_override": "imagenet_mean_std",
            },
            "num_workers": num_workers,
            "max_frames_per_batch": max_frames_per_batch,
        }
        _atomic_write_json(
            staging / "video_training_derivation.json", derivation
        )
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return VideoTrainingDerivation(
        output_root=output_root,
        train_dataset_sha256=output_fingerprints["train"],
        validation_dataset_sha256=output_fingerprints["validation"],
    )

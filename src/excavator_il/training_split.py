"""Stable parent-Episode splits for LeRobot ACT training and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


SPLIT_SCHEMA_VERSION = "excavator_training_split.v1"
MATERIALIZED_SPLIT_SCHEMA_VERSION = "excavator_materialized_training_split.v1"


@dataclass(frozen=True)
class TrainingSplit:
    schema_version: str
    dataset_root: str
    repo_id: str
    seed: int
    train_ratio: float
    train_source_episode_ids: tuple[str, ...]
    validation_source_episode_ids: tuple[str, ...]
    train_lerobot_episode_indices: tuple[int, ...]
    validation_lerobot_episode_indices: tuple[int, ...]
    lerobot_episode_to_source: dict[int, str]
    source_dataset_sha256: str


@dataclass(frozen=True)
class MaterializedTrainingSplit:
    train_root: Path
    validation_root: Path
    train_repo_id: str
    validation_repo_id: str
    provenance_path: Path


def _dataset_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "split_provenance.json"
    )
    if not files:
        raise ValueError("dataset contains no files to fingerprint")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _episode_source_mapping(dataset: LeRobotDataset) -> dict[int, str]:
    rows = dataset.hf_dataset.select_columns(
        ["episode_index", "source.episode_id"]
    )
    mapping: dict[int, str] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        source_id = str(row["source.episode_id"])
        if not source_id.strip():
            raise ValueError("source.episode_id must be non-empty")
        previous = mapping.get(episode_index)
        if previous is not None and previous != source_id:
            raise ValueError(
                f"LeRobot Episode {episode_index} contains multiple source Episode IDs"
            )
        mapping[episode_index] = source_id
    if len(mapping) != dataset.num_episodes:
        raise ValueError("dataset Episode metadata is incomplete")
    return mapping


def prepare_training_split(
    *,
    dataset_root: str | Path,
    repo_id: str,
    output_path: str | Path,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> TrainingSplit:
    """Split by source Episode while returning LeRobot Episode indices."""
    ratio = float(train_ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("train_ratio must be finite and between 0 and 1")

    root = Path(dataset_root).resolve()
    pipeline_marker = root / "pipeline_validation.json"
    if pipeline_marker.exists():
        raise ValueError("pipeline-validation dataset is not eligible for training")
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    mapping = _episode_source_mapping(dataset)
    source_ids = sorted(set(mapping.values()))
    if len(source_ids) < 2:
        raise ValueError("at least two source Episodes are required for train/validation split")

    shuffled = list(np.random.default_rng(seed).permutation(source_ids))
    train_count = round(ratio * len(shuffled))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    train_sources = tuple(sorted(str(value) for value in shuffled[:train_count]))
    validation_sources = tuple(sorted(str(value) for value in shuffled[train_count:]))
    train_source_set = set(train_sources)
    train_indices = tuple(
        sorted(index for index, source in mapping.items() if source in train_source_set)
    )
    validation_indices = tuple(
        sorted(index for index, source in mapping.items() if source not in train_source_set)
    )

    result = TrainingSplit(
        schema_version=SPLIT_SCHEMA_VERSION,
        dataset_root=str(root),
        repo_id=repo_id,
        seed=int(seed),
        train_ratio=ratio,
        train_source_episode_ids=train_sources,
        validation_source_episode_ids=validation_sources,
        train_lerobot_episode_indices=train_indices,
        validation_lerobot_episode_indices=validation_indices,
        lerobot_episode_to_source=dict(sorted(mapping.items())),
        source_dataset_sha256=_dataset_fingerprint(root),
    )
    destination = Path(output_path)
    serialized = asdict(result)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(serialized)):
            raise ValueError("existing manifest does not match requested split")
        return result
    _atomic_write_json(destination, serialized)
    return result


def _load_training_split(path: Path) -> TrainingSplit:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected = set(TrainingSplit.__dataclass_fields__)
        if not isinstance(raw, dict) or set(raw) != expected:
            raise TypeError("training split manifest fields are invalid")
        if raw.get("schema_version") != SPLIT_SCHEMA_VERSION:
            raise TypeError("unsupported training split schema")
        if not isinstance(raw["dataset_root"], str) or not raw["dataset_root"].strip():
            raise TypeError("dataset_root must be a non-empty string")
        if not isinstance(raw["repo_id"], str) or not raw["repo_id"].strip():
            raise TypeError("repo_id must be a non-empty string")
        if isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int):
            raise TypeError("seed must be an integer")
        ratio = raw["train_ratio"]
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not np.isfinite(ratio)
            or not 0.0 < ratio < 1.0
        ):
            raise TypeError("train_ratio must be a finite number between 0 and 1")

        def string_tuple(field: str) -> tuple[str, ...]:
            values = raw[field]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)
                or len(values) != len(set(values))
            ):
                raise TypeError(f"{field} must be a unique non-empty string list")
            return tuple(values)

        def index_tuple(field: str) -> tuple[int, ...]:
            values = raw[field]
            if (
                not isinstance(values, list)
                or not values
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in values
                )
                or len(values) != len(set(values))
            ):
                raise TypeError(f"{field} must be a unique non-negative integer list")
            return tuple(values)

        raw_mapping = raw["lerobot_episode_to_source"]
        if not isinstance(raw_mapping, dict) or not raw_mapping:
            raise TypeError("lerobot_episode_to_source must be a non-empty object")
        mapping: dict[int, str] = {}
        for key, value in raw_mapping.items():
            if (
                not isinstance(key, str)
                or not key.isdecimal()
                or not isinstance(value, str)
                or not value.strip()
            ):
                raise TypeError("lerobot_episode_to_source entries are invalid")
            index = int(key)
            if index in mapping:
                raise TypeError("lerobot_episode_to_source indices must be unique")
            mapping[index] = value
        source_sha256 = raw["source_dataset_sha256"]
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise TypeError("source_dataset_sha256 must be a lowercase SHA-256")
        result = TrainingSplit(
            schema_version=SPLIT_SCHEMA_VERSION,
            dataset_root=raw["dataset_root"],
            repo_id=raw["repo_id"],
            seed=raw["seed"],
            train_ratio=float(ratio),
            train_source_episode_ids=string_tuple("train_source_episode_ids"),
            validation_source_episode_ids=string_tuple(
                "validation_source_episode_ids"
            ),
            train_lerobot_episode_indices=index_tuple(
                "train_lerobot_episode_indices"
            ),
            validation_lerobot_episode_indices=index_tuple(
                "validation_lerobot_episode_indices"
            ),
            lerobot_episode_to_source=mapping,
            source_dataset_sha256=source_sha256,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid training split manifest") from exc
    return result


def materialize_training_split(
    *, manifest_path: str | Path, output_root: str | Path
) -> MaterializedTrainingSplit:
    """Create isolated train/validation datasets with subset-specific stats."""
    manifest = _load_training_split(Path(manifest_path))
    root = Path(output_root)
    if root.exists():
        raise ValueError(f"split output root already exists: {root}")

    source_root = Path(manifest.dataset_root)
    if (source_root / "pipeline_validation.json").exists():
        raise ValueError("pipeline-validation dataset is not eligible for training")

    dataset = LeRobotDataset(
        repo_id=manifest.repo_id,
        root=source_root,
    )
    if _episode_source_mapping(dataset) != manifest.lerobot_episode_to_source:
        raise ValueError("training split mapping no longer matches source dataset")
    if _dataset_fingerprint(source_root) != manifest.source_dataset_sha256:
        raise ValueError("source dataset content no longer matches training split manifest")
    train_indices = set(manifest.train_lerobot_episode_indices)
    validation_indices = set(manifest.validation_lerobot_episode_indices)
    mapping = manifest.lerobot_episode_to_source
    if train_indices & validation_indices or train_indices | validation_indices != set(mapping):
        raise ValueError("training split indices must be a disjoint complete partition")
    actual_train_sources = {mapping[index] for index in train_indices}
    actual_validation_sources = {mapping[index] for index in validation_indices}
    if (
        actual_train_sources != set(manifest.train_source_episode_ids)
        or actual_validation_sources != set(manifest.validation_source_episode_ids)
        or actual_train_sources & actual_validation_sources
    ):
        raise ValueError("training split parent Episode partition is inconsistent")
    from lerobot.datasets import dataset_tools

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        split_datasets = dataset_tools.split_dataset(
            dataset,
            {
                "train": list(manifest.train_lerobot_episode_indices),
                "validation": list(manifest.validation_lerobot_episode_indices),
            },
            output_dir=staging,
        )
        train_repo_id = split_datasets["train"].repo_id
        validation_repo_id = split_datasets["validation"].repo_id
        if _dataset_fingerprint(source_root) != manifest.source_dataset_sha256:
            raise ValueError("source dataset changed while materializing training split")
        train_sha256 = _dataset_fingerprint(staging / "train")
        validation_sha256 = _dataset_fingerprint(staging / "validation")
        provenance_path = staging / "split_provenance.json"
        _atomic_write_json(
            provenance_path,
            {
                "schema_version": MATERIALIZED_SPLIT_SCHEMA_VERSION,
                "source_dataset_sha256": manifest.source_dataset_sha256,
                "train_repo_id": train_repo_id,
                "validation_repo_id": validation_repo_id,
                "train_root": str((root.resolve() / "train")),
                "validation_root": str((root.resolve() / "validation")),
                "train_dataset_sha256": train_sha256,
                "validation_dataset_sha256": validation_sha256,
                "train_source_episode_ids": list(manifest.train_source_episode_ids),
                "validation_source_episode_ids": list(manifest.validation_source_episode_ids),
            },
        )
        staging.rename(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return MaterializedTrainingSplit(
        train_root=root / "train",
        validation_root=root / "validation",
        train_repo_id=train_repo_id,
        validation_repo_id=validation_repo_id,
        provenance_path=root / "split_provenance.json",
    )

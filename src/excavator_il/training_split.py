"""Leakage-safe grouped splits for LeRobot ACT training and validation."""

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


SPLIT_SCHEMA_VERSION = "excavator_training_split.v2"
LEGACY_SPLIT_SCHEMA_VERSION = "excavator_training_split.v1"
MATERIALIZED_SPLIT_SCHEMA_VERSION = "excavator_materialized_training_split.v1"

_SOURCE_EPISODE_GROUPING_KEY = "source.episode_id"
_SOIL_BLOCK_GROUPING_KEY = "source.soil_reset_block_id"
_UNKNOWN = "unknown"


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
    grouping_key: str
    train_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    source_episode_to_group: dict[str, str]
    group_to_task_variant: dict[str, str]


@dataclass(frozen=True)
class MaterializedTrainingSplit:
    train_root: Path
    validation_root: Path
    train_repo_id: str
    validation_repo_id: str
    provenance_path: Path


@dataclass(frozen=True)
class _DatasetGrouping:
    lerobot_episode_to_source: dict[int, str]
    grouping_key: str
    source_episode_to_group: dict[str, str]
    group_to_task_variant: dict[str, str]


def _dataset_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
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


def _dataset_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _episode_source_mapping(dataset: LeRobotDataset) -> dict[int, str]:
    rows = dataset.hf_dataset.select_columns(
        ["episode_index", "source.episode_id"]
    )
    mapping: dict[int, str] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        source_id = _dataset_text(row["source.episode_id"], "source.episode_id")
        previous = mapping.get(episode_index)
        if previous is not None and previous != source_id:
            raise ValueError(
                f"LeRobot Episode {episode_index} contains multiple source Episode IDs"
            )
        mapping[episode_index] = source_id
    if len(mapping) != dataset.num_episodes:
        raise ValueError("dataset Episode metadata is incomplete")
    return mapping


def _is_unknown(value: str) -> bool:
    return value.strip().lower() == _UNKNOWN


def _dataset_grouping(dataset: LeRobotDataset) -> _DatasetGrouping:
    """Derive the strongest safe grouping supported by dataset metadata."""
    mapping = _episode_source_mapping(dataset)
    metadata_columns = {
        "source.task_variant",
        "source.soil_reset_block_id",
    }
    available_metadata = metadata_columns & set(dataset.hf_dataset.column_names)
    if available_metadata and available_metadata != metadata_columns:
        raise ValueError("dataset source split metadata columns are incomplete")
    if not available_metadata:
        source_ids = sorted(set(mapping.values()))
        return _DatasetGrouping(
            lerobot_episode_to_source=mapping,
            grouping_key=_SOURCE_EPISODE_GROUPING_KEY,
            source_episode_to_group={source_id: source_id for source_id in source_ids},
            group_to_task_variant={source_id: _UNKNOWN for source_id in source_ids},
        )

    required = {"episode_index", "source.episode_id", *metadata_columns}
    rows = dataset.hf_dataset.select_columns(sorted(required))
    episode_metadata: dict[int, tuple[str, str, str]] = {}
    source_metadata: dict[str, tuple[str, str]] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        source_id = _dataset_text(row["source.episode_id"], "source.episode_id")
        task_variant = _dataset_text(
            row["source.task_variant"], "source.task_variant"
        )
        block_id = _dataset_text(
            row["source.soil_reset_block_id"], "source.soil_reset_block_id"
        )
        metadata = (source_id, task_variant, block_id)
        previous_episode = episode_metadata.get(episode_index)
        if previous_episode is not None and previous_episode != metadata:
            raise ValueError(
                f"LeRobot Episode {episode_index} contains inconsistent "
                "source split metadata"
            )
        episode_metadata[episode_index] = metadata
        previous_source = source_metadata.get(source_id)
        source_value = (task_variant, block_id)
        if previous_source is not None and previous_source != source_value:
            raise ValueError(
                f"source Episode {source_id!r} maps to multiple task variants "
                "or soil blocks"
            )
        source_metadata[source_id] = source_value

    if set(episode_metadata) != set(mapping):
        raise ValueError("dataset source split metadata is incomplete")
    if any(
        episode_metadata[index][0] != source_id
        for index, source_id in mapping.items()
    ):
        raise ValueError("source Episode mapping disagrees with split metadata")

    known_block_flags = [
        not _is_unknown(block_id) for _, block_id in source_metadata.values()
    ]
    if any(known_block_flags) and not all(known_block_flags):
        raise ValueError(
            "source.soil_reset_block_id is only partially populated; "
            "refusing a leaky fallback"
        )
    if not all(known_block_flags):
        source_ids = sorted(source_metadata)
        return _DatasetGrouping(
            lerobot_episode_to_source=mapping,
            grouping_key=_SOURCE_EPISODE_GROUPING_KEY,
            source_episode_to_group={source_id: source_id for source_id in source_ids},
            group_to_task_variant={
                source_id: source_metadata[source_id][0] for source_id in source_ids
            },
        )

    source_to_group = {
        source_id: block_id
        for source_id, (_, block_id) in sorted(source_metadata.items())
    }
    group_to_variant: dict[str, str] = {}
    for source_id, (task_variant, block_id) in sorted(source_metadata.items()):
        previous = group_to_variant.get(block_id)
        if previous is not None and previous != task_variant:
            raise ValueError(
                f"soil reset block {block_id!r} contains multiple task variants: "
                f"{previous!r} and {task_variant!r}"
            )
        group_to_variant[block_id] = task_variant
    return _DatasetGrouping(
        lerobot_episode_to_source=mapping,
        grouping_key=_SOIL_BLOCK_GROUPING_KEY,
        source_episode_to_group=source_to_group,
        group_to_task_variant=dict(sorted(group_to_variant.items())),
    )


def _episode_grouping(grouping: _DatasetGrouping) -> _DatasetGrouping:
    source_ids = sorted(grouping.source_episode_to_group)
    return _DatasetGrouping(
        lerobot_episode_to_source=dict(grouping.lerobot_episode_to_source),
        grouping_key=_SOURCE_EPISODE_GROUPING_KEY,
        source_episode_to_group={source_id: source_id for source_id in source_ids},
        group_to_task_variant={
            source_id: grouping.group_to_task_variant[
                grouping.source_episode_to_group[source_id]
            ]
            for source_id in source_ids
        },
    )


def _bounded_train_count(count: int, ratio: float) -> int:
    return min(max(round(ratio * count), 1), count - 1)


def _partition_group_ids(
    grouping: _DatasetGrouping,
    *,
    train_ratio: float,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    group_ids = sorted(grouping.group_to_task_variant)
    if len(group_ids) < 2:
        raise ValueError(
            "at least two split groups are required for train/validation split"
        )
    rng = np.random.default_rng(seed)
    if grouping.grouping_key == _SOURCE_EPISODE_GROUPING_KEY:
        shuffled = [str(value) for value in rng.permutation(group_ids)]
        train_count = _bounded_train_count(len(shuffled), train_ratio)
        return tuple(sorted(shuffled[:train_count])), tuple(
            sorted(shuffled[train_count:])
        )

    groups_by_variant: dict[str, list[str]] = {}
    for group_id, task_variant in grouping.group_to_task_variant.items():
        groups_by_variant.setdefault(task_variant, []).append(group_id)

    train: list[str] = []
    validation: list[str] = []
    singletons: list[str] = []
    for task_variant in sorted(groups_by_variant):
        variant_groups = sorted(groups_by_variant[task_variant])
        shuffled = [str(value) for value in rng.permutation(variant_groups)]
        if len(shuffled) == 1:
            singletons.extend(shuffled)
            continue
        train_count = _bounded_train_count(len(shuffled), train_ratio)
        train.extend(shuffled[:train_count])
        validation.extend(shuffled[train_count:])

    shuffled_singletons = [str(value) for value in rng.permutation(singletons)]
    target_train_count = _bounded_train_count(len(group_ids), train_ratio)
    singleton_train_count = min(
        max(target_train_count - len(train), 0), len(shuffled_singletons)
    )
    train.extend(shuffled_singletons[:singleton_train_count])
    validation.extend(shuffled_singletons[singleton_train_count:])
    if not train or not validation:
        raise ValueError(
            "task-stratified grouping could not create two non-empty splits"
        )
    return tuple(sorted(train)), tuple(sorted(validation))


def prepare_training_split(
    *,
    dataset_root: str | Path,
    repo_id: str,
    output_path: str | Path,
    train_ratio: float = 0.8,
    seed: int = 0,
    grouping: str = "auto",
) -> TrainingSplit:
    """Create a deterministic split, preferring whole soil reset blocks."""
    ratio = float(train_ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("train_ratio must be finite and between 0 and 1")
    if grouping not in {"auto", "episode"}:
        raise ValueError("grouping must be auto or episode")

    root = Path(dataset_root).resolve()
    pipeline_marker = root / "pipeline_validation.json"
    if pipeline_marker.exists():
        raise ValueError("pipeline-validation dataset is not eligible for training")
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    dataset_grouping = _dataset_grouping(dataset)
    if grouping == "episode":
        dataset_grouping = _episode_grouping(dataset_grouping)
    mapping = dataset_grouping.lerobot_episode_to_source
    source_ids = sorted(set(mapping.values()))
    if len(source_ids) < 2:
        raise ValueError(
            "at least two source Episodes are required for train/validation split"
        )

    train_groups, validation_groups = _partition_group_ids(
        dataset_grouping,
        train_ratio=ratio,
        seed=seed,
    )
    train_group_set = set(train_groups)
    train_sources = tuple(
        sorted(
            source_id
            for source_id, group_id in (
                dataset_grouping.source_episode_to_group.items()
            )
            if group_id in train_group_set
        )
    )
    validation_sources = tuple(
        sorted(set(source_ids) - set(train_sources))
    )
    train_source_set = set(train_sources)
    train_indices = tuple(
        sorted(index for index, source in mapping.items() if source in train_source_set)
    )
    validation_indices = tuple(
        sorted(
            index
            for index, source in mapping.items()
            if source not in train_source_set
        )
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
        grouping_key=dataset_grouping.grouping_key,
        train_group_ids=train_groups,
        validation_group_ids=validation_groups,
        source_episode_to_group=dict(
            sorted(dataset_grouping.source_episode_to_group.items())
        ),
        group_to_task_variant=dict(
            sorted(dataset_grouping.group_to_task_variant.items())
        ),
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


def _string_tuple(raw: dict, field: str) -> tuple[str, ...]:
    values = raw[field]
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(values) != len(set(values))
    ):
        raise TypeError(f"{field} must be a unique non-empty string list")
    return tuple(values)


def _index_tuple(raw: dict, field: str) -> tuple[int, ...]:
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


def _string_mapping(raw: dict, field: str) -> dict[str, str]:
    values = raw[field]
    if not isinstance(values, dict) or not values:
        raise TypeError(f"{field} must be a non-empty object")
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in values.items()
    ):
        raise TypeError(f"{field} entries are invalid")
    return dict(values)


def _validate_loaded_split(result: TrainingSplit) -> None:
    train_sources = set(result.train_source_episode_ids)
    validation_sources = set(result.validation_source_episode_ids)
    train_indices = set(result.train_lerobot_episode_indices)
    validation_indices = set(result.validation_lerobot_episode_indices)
    if train_sources & validation_sources:
        raise TypeError("source Episode partitions overlap")
    if train_indices & validation_indices:
        raise TypeError("LeRobot Episode partitions overlap")
    if train_indices | validation_indices != set(result.lerobot_episode_to_source):
        raise TypeError("LeRobot Episode partition is incomplete")
    all_sources = train_sources | validation_sources
    if set(result.lerobot_episode_to_source.values()) != all_sources:
        raise TypeError("source Episode partition is incomplete")
    if set(result.source_episode_to_group) != all_sources:
        raise TypeError("source_episode_to_group keys are incomplete")

    train_groups = set(result.train_group_ids)
    validation_groups = set(result.validation_group_ids)
    if train_groups & validation_groups:
        raise TypeError("split group partitions overlap")
    all_groups = train_groups | validation_groups
    if set(result.source_episode_to_group.values()) != all_groups:
        raise TypeError("source_episode_to_group values are incomplete")
    if set(result.group_to_task_variant) != all_groups:
        raise TypeError("group_to_task_variant keys are incomplete")
    if {
        result.source_episode_to_group[source_id] for source_id in train_sources
    } != train_groups or {
        result.source_episode_to_group[source_id] for source_id in validation_sources
    } != validation_groups:
        raise TypeError("group partition does not match source Episode partition")
    if result.grouping_key == _SOURCE_EPISODE_GROUPING_KEY:
        if any(
            source_id != group_id
            for source_id, group_id in result.source_episode_to_group.items()
        ):
            raise TypeError("source Episode grouping must map each Episode to itself")
    elif result.grouping_key == _SOIL_BLOCK_GROUPING_KEY:
        if any(_is_unknown(group_id) for group_id in all_groups):
            raise TypeError("soil reset block groups must be known")
    else:
        raise TypeError("unsupported grouping_key")


def _load_training_split(path: Path) -> TrainingSplit:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("training split manifest must be an object")
        version = raw.get("schema_version")
        legacy_fields = {
            "schema_version",
            "dataset_root",
            "repo_id",
            "seed",
            "train_ratio",
            "train_source_episode_ids",
            "validation_source_episode_ids",
            "train_lerobot_episode_indices",
            "validation_lerobot_episode_indices",
            "lerobot_episode_to_source",
            "source_dataset_sha256",
        }
        v2_fields = set(TrainingSplit.__dataclass_fields__)
        expected_fields = (
            legacy_fields if version == LEGACY_SPLIT_SCHEMA_VERSION else v2_fields
        )
        if version not in {LEGACY_SPLIT_SCHEMA_VERSION, SPLIT_SCHEMA_VERSION}:
            raise TypeError("unsupported training split schema")
        if set(raw) != expected_fields:
            raise TypeError("training split manifest fields are invalid")
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

        train_sources = _string_tuple(raw, "train_source_episode_ids")
        validation_sources = _string_tuple(raw, "validation_source_episode_ids")
        if version == LEGACY_SPLIT_SCHEMA_VERSION:
            all_sources = train_sources + validation_sources
            grouping_key = _SOURCE_EPISODE_GROUPING_KEY
            train_groups = train_sources
            validation_groups = validation_sources
            source_to_group = {source_id: source_id for source_id in all_sources}
            group_to_variant = {source_id: _UNKNOWN for source_id in all_sources}
        else:
            grouping_key = raw["grouping_key"]
            if not isinstance(grouping_key, str):
                raise TypeError("grouping_key must be a string")
            train_groups = _string_tuple(raw, "train_group_ids")
            validation_groups = _string_tuple(raw, "validation_group_ids")
            source_to_group = _string_mapping(raw, "source_episode_to_group")
            group_to_variant = _string_mapping(raw, "group_to_task_variant")

        result = TrainingSplit(
            schema_version=version,
            dataset_root=raw["dataset_root"],
            repo_id=raw["repo_id"],
            seed=raw["seed"],
            train_ratio=float(ratio),
            train_source_episode_ids=train_sources,
            validation_source_episode_ids=validation_sources,
            train_lerobot_episode_indices=_index_tuple(
                raw, "train_lerobot_episode_indices"
            ),
            validation_lerobot_episode_indices=_index_tuple(
                raw, "validation_lerobot_episode_indices"
            ),
            lerobot_episode_to_source=mapping,
            source_dataset_sha256=source_sha256,
            grouping_key=grouping_key,
            train_group_ids=train_groups,
            validation_group_ids=validation_groups,
            source_episode_to_group=source_to_group,
            group_to_task_variant=group_to_variant,
        )
        _validate_loaded_split(result)
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

    dataset = LeRobotDataset(repo_id=manifest.repo_id, root=source_root)
    current_mapping = _episode_source_mapping(dataset)
    if current_mapping != manifest.lerobot_episode_to_source:
        raise ValueError("training split mapping no longer matches source dataset")
    if _dataset_fingerprint(source_root) != manifest.source_dataset_sha256:
        raise ValueError(
            "source dataset content no longer matches training split manifest"
        )

    train_indices = set(manifest.train_lerobot_episode_indices)
    validation_indices = set(manifest.validation_lerobot_episode_indices)
    mapping = manifest.lerobot_episode_to_source
    if train_indices & validation_indices or train_indices | validation_indices != set(
        mapping
    ):
        raise ValueError("training split indices must be a disjoint complete partition")
    actual_train_sources = {mapping[index] for index in train_indices}
    actual_validation_sources = {mapping[index] for index in validation_indices}
    if (
        actual_train_sources != set(manifest.train_source_episode_ids)
        or actual_validation_sources != set(manifest.validation_source_episode_ids)
        or actual_train_sources & actual_validation_sources
    ):
        raise ValueError("training split parent Episode partition is inconsistent")

    if manifest.schema_version == SPLIT_SCHEMA_VERSION:
        current_grouping = _dataset_grouping(dataset)
        if manifest.grouping_key == _SOURCE_EPISODE_GROUPING_KEY:
            current_grouping = _episode_grouping(current_grouping)
        if (
            current_grouping.grouping_key != manifest.grouping_key
            or current_grouping.source_episode_to_group
            != manifest.source_episode_to_group
            or current_grouping.group_to_task_variant
            != manifest.group_to_task_variant
        ):
            raise ValueError("training split grouping no longer matches source dataset")
    actual_train_groups = {
        manifest.source_episode_to_group[source_id]
        for source_id in actual_train_sources
    }
    actual_validation_groups = {
        manifest.source_episode_to_group[source_id]
        for source_id in actual_validation_sources
    }
    if (
        actual_train_groups != set(manifest.train_group_ids)
        or actual_validation_groups != set(manifest.validation_group_ids)
        or actual_train_groups & actual_validation_groups
    ):
        raise ValueError("training split group partition is inconsistent")

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
            raise ValueError(
                "source dataset changed while materializing training split"
            )
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
                "train_root": str(root.resolve() / "train"),
                "validation_root": str(root.resolve() / "validation"),
                "train_dataset_sha256": train_sha256,
                "validation_dataset_sha256": validation_sha256,
                "train_source_episode_ids": list(manifest.train_source_episode_ids),
                "validation_source_episode_ids": list(
                    manifest.validation_source_episode_ids
                ),
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

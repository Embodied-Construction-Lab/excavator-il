"""Derive an immutable three-phase ACT dataset from a frozen video split."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .act_phase_conditioning import (
    PHASE_CONDITIONED_DATASET_SCHEMA_VERSION,
    PHASE_FEATURE_NAMES,
    PHASE_ORDER,
)
from .training_split import (
    MATERIALIZED_SPLIT_SCHEMA_VERSION,
    _atomic_write_json,
    _dataset_fingerprint,
)


PHASE_ANNOTATION_SCHEMA_VERSION = "excavator_act_three_phase_annotations.v1"
_PARTITIONS = ("train", "validation")
_DATA_FILE = re.compile(r"file-(\d{3})\.parquet")
_STAT_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


@dataclass(frozen=True)
class PhaseConditionedSplit:
    root: Path
    train_root: Path
    validation_root: Path
    provenance_path: Path
    train_dataset_sha256: str
    validation_dataset_sha256: str


@dataclass(frozen=True)
class _Annotation:
    partition: str
    data_file_index: int
    source_episode_id: str
    frame_count: int
    dig_to_transport_frame: int | None
    transport_to_dump_frame: int | None
    disposition: str


def derive_phase_conditioned_video_split(
    *,
    source_root: str | Path,
    annotation_path: str | Path,
    output_root: str | Path,
) -> PhaseConditionedSplit:
    """Append a three-way phase one-hot while preserving parent train/val isolation."""

    source = Path(source_root).expanduser().resolve()
    annotations_file = Path(annotation_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    _validate_paths(source, annotations_file, output)
    source_provenance = _load_source_provenance(source)
    source_fingerprints = _verify_source(source, source_provenance)
    annotations = _load_annotations(annotations_file)
    annotation_sha256 = _sha256_file(annotations_file)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(source, staging, copy_function=os.link, dirs_exist_ok=True)
        source_video_derivation_sha256 = _archive_video_derivation(staging)
        phase_counts: dict[str, dict[str, int]] = {}
        excluded: list[dict[str, Any]] = []
        for partition in _PARTITIONS:
            counts, removed = _transform_partition(
                staging / partition,
                annotations={
                    key[1]: value
                    for key, value in annotations.items()
                    if key[0] == partition
                },
            )
            phase_counts[partition] = counts
            excluded.extend(removed)

        output_fingerprints = {
            partition: _dataset_fingerprint(staging / partition)
            for partition in _PARTITIONS
        }
        output_provenance = {
            **source_provenance,
            "train_repo_id": f"{source_provenance['train_repo_id']}_three_phase",
            "validation_repo_id": (
                f"{source_provenance['validation_repo_id']}_three_phase"
            ),
            "train_root": str(output / "train"),
            "validation_root": str(output / "validation"),
            "train_dataset_sha256": output_fingerprints["train"],
            "validation_dataset_sha256": output_fingerprints["validation"],
        }
        _atomic_write_json(staging / "split_provenance.json", output_provenance)
        phase_provenance = {
            "schema_version": PHASE_CONDITIONED_DATASET_SCHEMA_VERSION,
            "source_split_root": str(source),
            "source_partition_sha256": source_fingerprints,
            "source_video_training_derivation_sha256": (
                source_video_derivation_sha256
            ),
            "annotation_path": str(annotations_file),
            "annotation_sha256": annotation_sha256,
            "phase_order": list(PHASE_ORDER),
            "phase_feature_names": list(PHASE_FEATURE_NAMES),
            "state_contract": {
                "source_dimension": 11,
                "derived_dimension": 14,
                "append": "three_phase_one_hot",
            },
            "phase_frame_counts": phase_counts,
            "excluded_sequences": excluded,
            "output_partition_sha256": output_fingerprints,
        }
        _atomic_write_json(
            staging / "phase_conditioning_provenance.json", phase_provenance
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return PhaseConditionedSplit(
        root=output,
        train_root=output / "train",
        validation_root=output / "validation",
        provenance_path=output / "phase_conditioning_provenance.json",
        train_dataset_sha256=output_fingerprints["train"],
        validation_dataset_sha256=output_fingerprints["validation"],
    )


def _validate_paths(source: Path, annotations: Path, output: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"source split is unavailable: {source}")
    if not annotations.is_file() or annotations.is_symlink():
        raise ValueError(f"phase annotations are unavailable: {annotations}")
    if output.exists() or output.is_symlink():
        raise ValueError(f"phase-conditioned output already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError("phase-conditioned output must not be inside source split")


def _load_source_provenance(source: Path) -> dict[str, Any]:
    value = _read_json(source / "split_provenance.json", "split provenance")
    required = {
        "schema_version",
        "source_dataset_sha256",
        "train_repo_id",
        "validation_repo_id",
        "train_root",
        "validation_root",
        "train_dataset_sha256",
        "validation_dataset_sha256",
        "train_source_episode_ids",
        "validation_source_episode_ids",
    }
    if set(value) != required:
        raise ValueError("split provenance fields are invalid")
    if value["schema_version"] != MATERIALIZED_SPLIT_SCHEMA_VERSION:
        raise ValueError("split provenance schema is invalid")
    return value


def _verify_source(
    source: Path, provenance: dict[str, Any]
) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for partition in _PARTITIONS:
        root = source / partition
        info = _read_json(root / "meta/info.json", f"{partition} metadata")
        state = info.get("features", {}).get("observation.state")
        if not isinstance(state, dict) or state.get("shape") != [11]:
            raise ValueError(f"{partition} source state contract must be 11D")
        camera_types = {
            feature.get("dtype")
            for name, feature in info.get("features", {}).items()
            if name.startswith("observation.images.") and isinstance(feature, dict)
        }
        if camera_types != {"video"}:
            raise ValueError(f"{partition} source cameras must be video-backed")
        current = _dataset_fingerprint(root)
        if current != provenance.get(f"{partition}_dataset_sha256"):
            raise ValueError(f"{partition} source fingerprint mismatch")
        fingerprints[partition] = current
    return fingerprints


def _load_annotations(path: Path) -> dict[tuple[str, int], _Annotation]:
    value = _read_json(path, "phase annotations")
    if set(value) != {
        "schema_version",
        "phase_order",
        "phase_feature_names",
        "entries",
    }:
        raise ValueError("phase annotation fields are invalid")
    if value["schema_version"] != PHASE_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("phase annotation schema is invalid")
    if value["phase_order"] != list(PHASE_ORDER):
        raise ValueError("phase annotation order is invalid")
    if value["phase_feature_names"] != list(PHASE_FEATURE_NAMES):
        raise ValueError("phase feature names are invalid")
    if not isinstance(value["entries"], list) or not value["entries"]:
        raise ValueError("phase annotation entries are invalid")
    result: dict[tuple[str, int], _Annotation] = {}
    expected_fields = {
        "partition",
        "data_file_index",
        "source_episode_id",
        "frame_count",
        "dig_to_transport_frame",
        "transport_to_dump_frame",
        "disposition",
    }
    for raw in value["entries"]:
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ValueError("phase annotation entry fields are invalid")
        entry = _parse_annotation(raw)
        key = (entry.partition, entry.data_file_index)
        if key in result:
            raise ValueError("phase annotation entry is duplicated")
        result[key] = entry
    return result


def _parse_annotation(raw: dict[str, Any]) -> _Annotation:
    partition = raw["partition"]
    file_index = raw["data_file_index"]
    source_id = raw["source_episode_id"]
    frame_count = raw["frame_count"]
    disposition = raw["disposition"]
    if partition not in _PARTITIONS:
        raise ValueError("phase annotation partition is invalid")
    if isinstance(file_index, bool) or not isinstance(file_index, int) or file_index < 0:
        raise ValueError("phase annotation file index is invalid")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("phase annotation source Episode is invalid")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("phase annotation frame count is invalid")
    first = raw["dig_to_transport_frame"]
    second = raw["transport_to_dump_frame"]
    if disposition == "include":
        if (
            isinstance(first, bool)
            or not isinstance(first, int)
            or isinstance(second, bool)
            or not isinstance(second, int)
            or not 0 < first < second < frame_count
        ):
            raise ValueError("phase boundaries must be strictly ordered inside sequence")
    elif disposition == "exclude_short_fragment":
        if first is not None or second is not None:
            raise ValueError("excluded sequence must not define phase boundaries")
    else:
        raise ValueError("phase annotation disposition is invalid")
    return _Annotation(
        partition=partition,
        data_file_index=file_index,
        source_episode_id=source_id,
        frame_count=frame_count,
        dig_to_transport_frame=first,
        transport_to_dump_frame=second,
        disposition=disposition,
    )


def _archive_video_derivation(root: Path) -> str | None:
    source = root / "video_training_derivation.json"
    if not source.is_file() or source.is_symlink():
        return None
    digest = _sha256_file(source)
    source.replace(root / "source_video_training_derivation.json")
    return digest


def _transform_partition(
    root: Path, *, annotations: dict[int, _Annotation]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    paths = sorted((root / "data").rglob("*.parquet"))
    indexed_paths = {_data_file_index(path): path for path in paths}
    if set(indexed_paths) != set(annotations):
        raise ValueError("phase annotations do not exactly cover source sequences")
    episode_meta_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episode_meta = pq.read_table(episode_meta_path)
    excluded_indices: set[int] = set()
    counts = {phase: 0 for phase in PHASE_ORDER}
    excluded: list[dict[str, Any]] = []
    for file_index, path in indexed_paths.items():
        entry = annotations[file_index]
        table = pq.read_table(path)
        episode_index = _validate_sequence(table, entry)
        if entry.disposition == "exclude_short_fragment":
            excluded_indices.add(episode_index)
            path.unlink()
            excluded.append(
                {
                    "partition": entry.partition,
                    "data_file_index": entry.data_file_index,
                    "source_episode_id": entry.source_episode_id,
                    "frame_count": entry.frame_count,
                    "reason": entry.disposition,
                }
            )
            continue
        assert entry.dig_to_transport_frame is not None
        assert entry.transport_to_dump_frame is not None
        counts["DIG"] += entry.dig_to_transport_frame
        counts["TRANSPORT"] += (
            entry.transport_to_dump_frame - entry.dig_to_transport_frame
        )
        counts["DUMP"] += entry.frame_count - entry.transport_to_dump_frame
        _rewrite_state(path, table=table, entry=entry)

    _drop_terminal_episodes(episode_meta_path, episode_meta, excluded_indices)
    _update_partition_metadata(root, counts=counts)
    return counts, excluded


def _data_file_index(path: Path) -> int:
    match = _DATA_FILE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected Parquet filename: {path}")
    return int(match.group(1))


def _validate_sequence(table: pa.Table, entry: _Annotation) -> int:
    required = {
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "source.episode_id",
    }
    if not required <= set(table.column_names) or table.num_rows != entry.frame_count:
        raise ValueError("phase annotation does not match sequence frame count")
    source_ids = set(table["source.episode_id"].to_pylist())
    episode_indices = set(table["episode_index"].to_pylist())
    if source_ids != {entry.source_episode_id} or len(episode_indices) != 1:
        raise ValueError("phase annotation does not match sequence identity")
    if table["frame_index"].to_pylist() != list(range(entry.frame_count)):
        raise ValueError("source sequence frame indices are not contiguous")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if states.shape != (entry.frame_count, 11) or not np.isfinite(states).all():
        raise ValueError("source sequence state contract is invalid")
    return int(next(iter(episode_indices)))


def _rewrite_state(path: Path, *, table: pa.Table, entry: _Annotation) -> None:
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    phase = np.zeros((entry.frame_count, 3), dtype=np.float32)
    phase[: entry.dig_to_transport_frame, 0] = 1.0
    phase[entry.dig_to_transport_frame : entry.transport_to_dump_frame, 1] = 1.0
    phase[entry.transport_to_dump_frame :, 2] = 1.0
    combined = np.concatenate((states, phase), axis=1)
    values = pa.array(combined.reshape(-1), type=pa.float32())
    replacement = pa.FixedSizeListArray.from_arrays(values, 14)
    index = table.schema.get_field_index("observation.state")
    transformed = table.set_column(index, "observation.state", replacement)
    _atomic_write_parquet(path, transformed)


def _drop_terminal_episodes(
    path: Path, table: pa.Table, excluded_indices: set[int]
) -> None:
    if not excluded_indices:
        return
    episode_indices = [int(value) for value in table["episode_index"].to_pylist()]
    expected = set(range(min(excluded_indices), len(episode_indices)))
    if excluded_indices != expected or episode_indices != list(range(len(episode_indices))):
        raise ValueError("only terminal short-fragment Episodes may be excluded")
    keep_count = len(episode_indices) - len(excluded_indices)
    _atomic_write_parquet(path, table.slice(0, keep_count))


def _update_partition_metadata(root: Path, *, counts: dict[str, int]) -> None:
    info_path = root / "meta/info.json"
    info = _read_json(info_path, "partition metadata")
    total_frames = sum(counts.values())
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet")
    state_feature = info["features"]["observation.state"]
    names = state_feature.get("names")
    if not isinstance(names, list) or len(names) != 11:
        raise ValueError("source state feature names are invalid")
    updated_features = {
        **info["features"],
        "observation.state": {
            **state_feature,
            "shape": [14],
            "names": [*names, *PHASE_FEATURE_NAMES],
        },
    }
    updated_info = {
        **info,
        "total_episodes": episodes.num_rows,
        "total_frames": total_frames,
        "splits": {"train": f"0:{episodes.num_rows}"},
        "features": updated_features,
    }
    _atomic_write_json(info_path, updated_info)
    _recompute_state_stats(root, total_frames=total_frames)


def _recompute_state_stats(root: Path, *, total_frames: int) -> None:
    arrays = []
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["observation.state"])
        arrays.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float64))
    if not arrays:
        raise ValueError("derived partition contains no training sequences")
    states = np.concatenate(arrays, axis=0)
    if states.shape != (total_frames, 14) or not np.isfinite(states).all():
        raise ValueError("derived state matrix is invalid")
    quantiles = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
    computed = {
        "min": states.min(axis=0).tolist(),
        "max": states.max(axis=0).tolist(),
        "mean": states.mean(axis=0).tolist(),
        "std": states.std(axis=0).tolist(),
        "count": [total_frames],
        **{key: np.quantile(states, value, axis=0).tolist() for key, value in quantiles.items()},
    }
    stats_path = root / "meta/stats.json"
    stats = _read_json(stats_path, "partition stats")
    _atomic_write_json(stats_path, {**stats, "observation.state": computed})


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="snappy")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

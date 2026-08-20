"""Reproducible derived LeRobot split with the expert swing label fixed at zero."""

from __future__ import annotations

from dataclasses import dataclass
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

from .raw_episode import ACTION_FIELDS
from .training_split import (
    MATERIALIZED_SPLIT_SCHEMA_VERSION,
    _atomic_write_json,
    _dataset_fingerprint,
)


ACTION_TRANSFORM_SCHEMA_VERSION = "excavator_action_transform.v1"
_SAFE_SUFFIX = re.compile(r"[a-z0-9][a-z0-9_]*")
_STAT_VECTOR_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


@dataclass(frozen=True)
class DerivedActionSplit:
    root: Path
    train_root: Path
    validation_root: Path
    train_repo_id: str
    validation_repo_id: str
    provenance_path: Path


def derive_zero_swing_split(
    *,
    source_root: str | Path,
    output_root: str | Path,
    repo_suffix: str = "swing_zero",
) -> DerivedActionSplit:
    """Copy a materialized split and atomically force action_swing to zero."""

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not _SAFE_SUFFIX.fullmatch(repo_suffix):
        raise ValueError("repo_suffix must contain lowercase letters, digits, or underscores")
    if output.exists():
        raise ValueError(f"derived split output already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError("derived split output must not be inside the source split")
    provenance = _load_source_provenance(source)
    _verify_action_contract(source)
    _verify_source_fingerprints(source, provenance)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(source, staging, copy_function=os.link, dirs_exist_ok=True)
        rewritten_files = 0
        rewritten_rows = 0
        for partition in ("train", "validation"):
            parquet_paths = sorted((staging / partition / "data").rglob("*.parquet"))
            if not parquet_paths:
                raise ValueError(f"{partition} contains no Parquet data")
            for parquet_path in parquet_paths:
                rewritten_rows += _rewrite_action_swing(parquet_path)
                rewritten_files += 1
            _zero_action_swing_stats(staging / partition / "meta" / "stats.json")

        train_sha256 = _dataset_fingerprint(staging / "train")
        validation_sha256 = _dataset_fingerprint(staging / "validation")
        train_repo_id = f"{provenance['train_repo_id']}_{repo_suffix}"
        validation_repo_id = f"{provenance['validation_repo_id']}_{repo_suffix}"
        derived_split_provenance = {
            **provenance,
            "train_repo_id": train_repo_id,
            "validation_repo_id": validation_repo_id,
            "train_root": str(output / "train"),
            "validation_root": str(output / "validation"),
            "train_dataset_sha256": train_sha256,
            "validation_dataset_sha256": validation_sha256,
        }
        _atomic_write_json(staging / "split_provenance.json", derived_split_provenance)
        transform_path = staging / "action_transform_provenance.json"
        _atomic_write_json(
            transform_path,
            {
                "schema_version": ACTION_TRANSFORM_SCHEMA_VERSION,
                "source_split_root": str(source),
                "source_train_dataset_sha256": provenance["train_dataset_sha256"],
                "source_validation_dataset_sha256": provenance[
                    "validation_dataset_sha256"
                ],
                "derived_train_dataset_sha256": train_sha256,
                "derived_validation_dataset_sha256": validation_sha256,
                "rewritten_parquet_file_count": rewritten_files,
                "rewritten_frame_count": rewritten_rows,
                "action_order": list(ACTION_FIELDS),
                "transform": {
                    "feature": "action",
                    "field": "action_swing",
                    "index": 3,
                    "value": 0.0,
                },
            },
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DerivedActionSplit(
        root=output,
        train_root=output / "train",
        validation_root=output / "validation",
        train_repo_id=train_repo_id,
        validation_repo_id=validation_repo_id,
        provenance_path=output / "action_transform_provenance.json",
    )


def _load_source_provenance(source: Path) -> dict[str, Any]:
    try:
        value = json.loads((source / "split_provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("source split provenance is unavailable") from exc
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
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("source split provenance fields are invalid")
    if value.get("schema_version") != MATERIALIZED_SPLIT_SCHEMA_VERSION:
        raise ValueError("source split provenance schema is invalid")
    for key in ("train_repo_id", "validation_repo_id"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"source split {key} is invalid")
    return value


def _verify_source_fingerprints(source: Path, provenance: dict[str, Any]) -> None:
    if (
        _dataset_fingerprint(source / "train") != provenance["train_dataset_sha256"]
        or _dataset_fingerprint(source / "validation")
        != provenance["validation_dataset_sha256"]
    ):
        raise ValueError("source split content does not match its provenance")


def _verify_action_contract(source: Path) -> None:
    expected = list(ACTION_FIELDS)
    for partition in ("train", "validation"):
        try:
            info = json.loads(
                (source / partition / "meta" / "info.json").read_text(encoding="utf-8")
            )
            action = info["features"]["action"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{partition} action metadata is invalid") from exc
        if action.get("shape") != [4] or action.get("names") != expected:
            raise ValueError(f"{partition} action contract is not authoritative")


def _rewrite_action_swing(path: Path) -> int:
    table = pq.read_table(path)
    index = table.schema.get_field_index("action")
    if index < 0:
        raise ValueError(f"Parquet file has no action column: {path}")
    field = table.schema.field(index)
    if not pa.types.is_fixed_size_list(field.type) or field.type.list_size != 4:
        raise ValueError(f"Parquet action column is not float32[4]: {path}")
    actions = np.asarray(table.column(index).combine_chunks().to_pylist(), dtype=np.float32)
    if actions.shape != (table.num_rows, 4) or not np.isfinite(actions).all():
        raise ValueError(f"Parquet action values are invalid: {path}")
    actions[:, 3] = 0.0
    flattened = pa.array(actions.reshape(-1), type=pa.float32())
    replacement = pa.FixedSizeListArray.from_arrays(flattened, 4)
    transformed = table.set_column(index, field, replacement)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(transformed, temporary, compression="snappy")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return table.num_rows


def _zero_action_swing_stats(path: Path) -> None:
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
        action = stats["action"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"action stats are invalid: {path}") from exc
    for key in _STAT_VECTOR_KEYS:
        vector = action.get(key)
        if not isinstance(vector, list) or len(vector) != 4:
            raise ValueError(f"action stats {key} must contain four values")
        action[key] = [*vector[:3], 0.0]
    _atomic_write_json(path, stats)

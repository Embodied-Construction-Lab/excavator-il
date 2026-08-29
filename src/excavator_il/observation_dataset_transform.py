"""Derive an ACT state-only ablation by replacing RGB observations with a constant.

The visual feature is intentionally retained so the ACT architecture and parameter count
stay identical to the visual baseline.  The derived dataset therefore removes scene
information without silently changing the policy class.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
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
from PIL import Image

from .training_split import (
    MATERIALIZED_SPLIT_SCHEMA_VERSION,
    _atomic_write_json,
    _dataset_fingerprint,
)


OBSERVATION_TRANSFORM_SCHEMA_VERSION = "excavator_observation_transform.v1"
DEFAULT_IMAGE_FEATURE = "observation.images.front"
DEFAULT_REPO_SUFFIX = "state_only_constant_image"
_SAFE_SUFFIX = re.compile(r"[a-z0-9][a-z0-9_]*")
_IMAGE_STAT_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


@dataclass(frozen=True)
class DerivedObservationSplit:
    root: Path
    train_root: Path
    validation_root: Path
    train_repo_id: str
    validation_repo_id: str
    provenance_path: Path


def derive_constant_image_split(
    *,
    source_root: str | Path,
    output_root: str | Path,
    image_feature: str = DEFAULT_IMAGE_FEATURE,
    repo_suffix: str = DEFAULT_REPO_SUFFIX,
) -> DerivedObservationSplit:
    """Publish an immutable constant-image copy of a materialized train/validation split."""

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    _validate_request(source, output, image_feature, repo_suffix)
    provenance = _load_source_provenance(source)
    height, width = _verify_image_contract(source, image_feature)
    _verify_source_fingerprints(source, provenance)
    constant_bytes = _constant_black_png(height=height, width=width)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(source, staging, copy_function=os.link, dirs_exist_ok=True)
        rewritten_files = 0
        rewritten_frames = 0
        for partition in ("train", "validation"):
            parquet_paths = sorted((staging / partition / "data").rglob("*.parquet"))
            if not parquet_paths:
                raise ValueError(f"{partition} contains no Parquet data")
            for parquet_path in parquet_paths:
                rewritten_frames += _rewrite_images(
                    parquet_path,
                    image_feature=image_feature,
                    constant_bytes=constant_bytes,
                    constant_path=f"constant_black_{width}x{height}.png",
                )
                rewritten_files += 1
            _zero_image_stats(
                staging / partition / "meta" / "stats.json",
                image_feature=image_feature,
            )

        train_sha256 = _dataset_fingerprint(staging / "train")
        validation_sha256 = _dataset_fingerprint(staging / "validation")
        train_repo_id = f"{provenance['train_repo_id']}_{repo_suffix}"
        validation_repo_id = f"{provenance['validation_repo_id']}_{repo_suffix}"
        _atomic_write_json(
            staging / "split_provenance.json",
            {
                **provenance,
                "train_repo_id": train_repo_id,
                "validation_repo_id": validation_repo_id,
                "train_root": str(output / "train"),
                "validation_root": str(output / "validation"),
                "train_dataset_sha256": train_sha256,
                "validation_dataset_sha256": validation_sha256,
            },
        )
        _atomic_write_json(
            staging / "observation_transform_provenance.json",
            {
                "schema_version": OBSERVATION_TRANSFORM_SCHEMA_VERSION,
                "ablation": "state_only_constant_image",
                "source_split_root": str(source),
                "source_train_dataset_sha256": provenance["train_dataset_sha256"],
                "source_validation_dataset_sha256": provenance[
                    "validation_dataset_sha256"
                ],
                "derived_train_dataset_sha256": train_sha256,
                "derived_validation_dataset_sha256": validation_sha256,
                "rewritten_parquet_file_count": rewritten_files,
                "rewritten_frame_count": rewritten_frames,
                "preserved_features": ["observation.state", "action"],
                "constant_image": {
                    "feature": image_feature,
                    "shape": [height, width, 3],
                    "format": "PNG",
                    "rgb_value": [0, 0, 0],
                    "sha256": hashlib.sha256(constant_bytes).hexdigest(),
                },
            },
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return DerivedObservationSplit(
        root=output,
        train_root=output / "train",
        validation_root=output / "validation",
        train_repo_id=train_repo_id,
        validation_repo_id=validation_repo_id,
        provenance_path=output / "observation_transform_provenance.json",
    )


def _validate_request(
    source: Path, output: Path, image_feature: str, repo_suffix: str
) -> None:
    if not source.is_dir():
        raise ValueError(f"source split does not exist: {source}")
    if output.exists():
        raise ValueError(f"derived split output already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError("derived split output must not be inside the source split")
    if not isinstance(image_feature, str) or not image_feature.startswith(
        "observation.images."
    ):
        raise ValueError("image_feature must be an observation.images.* feature")
    if not _SAFE_SUFFIX.fullmatch(repo_suffix):
        raise ValueError("repo_suffix must contain lowercase letters, digits, or underscores")


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
    if value["schema_version"] != MATERIALIZED_SPLIT_SCHEMA_VERSION:
        raise ValueError("source split provenance schema is invalid")
    return value


def _verify_source_fingerprints(source: Path, provenance: dict[str, Any]) -> None:
    for partition in ("train", "validation"):
        expected = provenance[f"{partition}_dataset_sha256"]
        if _dataset_fingerprint(source / partition) != expected:
            raise ValueError("source split content does not match its provenance")


def _verify_image_contract(source: Path, image_feature: str) -> tuple[int, int]:
    expected_shape: list[int] | None = None
    for partition in ("train", "validation"):
        try:
            info = json.loads(
                (source / partition / "meta" / "info.json").read_text(encoding="utf-8")
            )
            feature = info["features"][image_feature]
            shape = feature["shape"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{partition} image metadata is invalid") from exc
        if feature.get("dtype") != "image" or not (
            isinstance(shape, list)
            and len(shape) == 3
            and all(isinstance(value, int) and value > 0 for value in shape)
            and shape[2] == 3
        ):
            raise ValueError(f"{partition} must provide a positive RGB image shape")
        if expected_shape is not None and shape != expected_shape:
            raise ValueError("train and validation image shapes differ")
        expected_shape = shape
    assert expected_shape is not None
    return expected_shape[0], expected_shape[1]


def _constant_black_png(*, height: int, width: int) -> bytes:
    image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8), mode="RGB")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _rewrite_images(
    path: Path,
    *,
    image_feature: str,
    constant_bytes: bytes,
    constant_path: str,
) -> int:
    table = pq.read_table(path)
    index = table.schema.get_field_index(image_feature)
    if index < 0:
        raise ValueError(f"Parquet file has no {image_feature} column: {path}")
    field = table.schema.field(index)
    if not pa.types.is_struct(field.type) or set(field.type.names) != {"bytes", "path"}:
        raise ValueError(f"Parquet image column has an invalid type: {path}")
    replacement = pa.array(
        [
            {"bytes": constant_bytes, "path": constant_path}
            for _ in range(table.num_rows)
        ],
        type=field.type,
    )
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


def _zero_image_stats(path: Path, *, image_feature: str) -> None:
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
        image_stats = stats[image_feature]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"image stats are invalid: {path}") from exc
    zero_channels = [[[0.0]], [[0.0]], [[0.0]]]
    for key in _IMAGE_STAT_KEYS:
        if key not in image_stats:
            raise ValueError(f"image stats are missing {key}")
        image_stats[key] = zero_channels
    _atomic_write_json(path, stats)

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from excavator_il.action_dataset_transform import derive_zero_swing_split
from excavator_il.training_split import (
    MATERIALIZED_SPLIT_SCHEMA_VERSION,
    _dataset_fingerprint,
)


def _write_partition(root: Path, actions: list[list[float]]) -> None:
    data = root / "data" / "chunk-000"
    meta = root / "meta"
    data.mkdir(parents=True)
    meta.mkdir(parents=True)
    values = pa.array(
        [value for row in actions for value in row], type=pa.float32()
    )
    table = pa.table(
        {
            "action": pa.FixedSizeListArray.from_arrays(values, 4),
            "episode_index": pa.array([0] * len(actions), type=pa.int64()),
        }
    )
    pq.write_table(table, data / "file-000.parquet", compression="snappy")
    (meta / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "action": {
                        "dtype": "float32",
                        "shape": [4],
                        "names": [
                            "action_boom",
                            "action_stick",
                            "action_bucket",
                            "action_swing",
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    action_stats = {
        key: [1.0, 2.0, 3.0, 4.0]
        for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    action_stats["count"] = [len(actions)]
    (meta / "stats.json").write_text(
        json.dumps({"action": action_stats}), encoding="utf-8"
    )


def test_derive_zero_swing_split_preserves_source_and_publishes_valid_fingerprints(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _write_partition(source / "train", [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]])
    _write_partition(source / "validation", [[0.5, 0.6, 0.7, 0.8]])
    source_provenance = {
        "schema_version": MATERIALIZED_SPLIT_SCHEMA_VERSION,
        "source_dataset_sha256": "a" * 64,
        "train_repo_id": "local/source_train",
        "validation_repo_id": "local/source_validation",
        "train_root": str(source / "train"),
        "validation_root": str(source / "validation"),
        "train_dataset_sha256": _dataset_fingerprint(source / "train"),
        "validation_dataset_sha256": _dataset_fingerprint(source / "validation"),
        "train_source_episode_ids": ["episode_0001"],
        "validation_source_episode_ids": ["episode_0002"],
    }
    (source / "split_provenance.json").write_text(
        json.dumps(source_provenance), encoding="utf-8"
    )
    original_actions = pq.read_table(
        source / "train/data/chunk-000/file-000.parquet", columns=["action"]
    )["action"].to_pylist()

    output = tmp_path / "derived"
    result = derive_zero_swing_split(
        source_root=source,
        output_root=output,
        repo_suffix="swing_zero",
    )

    np.testing.assert_allclose(
        pq.read_table(
            output / "train/data/chunk-000/file-000.parquet", columns=["action"]
        )["action"].to_pylist(),
        [[0.1, 0.2, 0.3, 0.0], [-0.1, -0.2, -0.3, 0.0]],
    )
    assert pq.read_table(
        source / "train/data/chunk-000/file-000.parquet", columns=["action"]
    )["action"].to_pylist() == original_actions
    stats = json.loads((output / "train/meta/stats.json").read_text())["action"]
    assert all(stats[key][3] == 0.0 for key in stats if key != "count")
    provenance = json.loads((output / "split_provenance.json").read_text())
    assert provenance["train_repo_id"] == "local/source_train_swing_zero"
    assert provenance["validation_repo_id"] == "local/source_validation_swing_zero"
    assert provenance["train_dataset_sha256"] == _dataset_fingerprint(output / "train")
    assert provenance["validation_dataset_sha256"] == _dataset_fingerprint(
        output / "validation"
    )
    transform = json.loads(result.provenance_path.read_text())
    assert transform["transform"] == {
        "feature": "action",
        "field": "action_swing",
        "index": 3,
        "value": 0.0,
    }


def test_derive_zero_swing_split_rejects_empty_partition_without_publishing(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _write_partition(source / "train", [[0.1, 0.2, 0.3, 0.4]])
    _write_partition(source / "validation", [[0.5, 0.6, 0.7, 0.8]])
    (source / "validation/data/chunk-000/file-000.parquet").unlink()
    source_provenance = {
        "schema_version": MATERIALIZED_SPLIT_SCHEMA_VERSION,
        "source_dataset_sha256": "a" * 64,
        "train_repo_id": "local/source_train",
        "validation_repo_id": "local/source_validation",
        "train_root": str(source / "train"),
        "validation_root": str(source / "validation"),
        "train_dataset_sha256": _dataset_fingerprint(source / "train"),
        "validation_dataset_sha256": _dataset_fingerprint(source / "validation"),
        "train_source_episode_ids": ["episode_0001"],
        "validation_source_episode_ids": ["episode_0002"],
    }
    (source / "split_provenance.json").write_text(
        json.dumps(source_provenance), encoding="utf-8"
    )
    output = tmp_path / "derived"

    with pytest.raises(ValueError, match="validation contains no Parquet data"):
        derive_zero_swing_split(source_root=source, output_root=output)

    assert not output.exists()
    assert not list(tmp_path.glob(".derived.*"))


def test_derive_zero_swing_split_rejects_parquet_without_action_column(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _write_partition(source / "train", [[0.1, 0.2, 0.3, 0.4]])
    _write_partition(source / "validation", [[0.5, 0.6, 0.7, 0.8]])
    train_path = source / "train/data/chunk-000/file-000.parquet"
    values = pa.array([0.1, 0.2, 0.3, 0.4], type=pa.float32())
    pq.write_table(
        pa.table({"other": pa.FixedSizeListArray.from_arrays(values, 4)}),
        train_path,
    )
    source_provenance = {
        "schema_version": MATERIALIZED_SPLIT_SCHEMA_VERSION,
        "source_dataset_sha256": "a" * 64,
        "train_repo_id": "local/source_train",
        "validation_repo_id": "local/source_validation",
        "train_root": str(source / "train"),
        "validation_root": str(source / "validation"),
        "train_dataset_sha256": _dataset_fingerprint(source / "train"),
        "validation_dataset_sha256": _dataset_fingerprint(source / "validation"),
        "train_source_episode_ids": ["episode_0001"],
        "validation_source_episode_ids": ["episode_0002"],
    }
    (source / "split_provenance.json").write_text(
        json.dumps(source_provenance), encoding="utf-8"
    )
    output = tmp_path / "derived"

    with pytest.raises(ValueError, match="has no action column"):
        derive_zero_swing_split(source_root=source, output_root=output)

    assert not output.exists()

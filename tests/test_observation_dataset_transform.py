import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from excavator_il.observation_dataset_transform import (
    derive_constant_image_split,
)
from excavator_il.training_split import (
    MATERIALIZED_SPLIT_SCHEMA_VERSION,
    _dataset_fingerprint,
)


IMAGE_FEATURE = "observation.images.front"


def _png(value: int, *, height: int = 4, width: int = 6) -> bytes:
    image = Image.fromarray(
        np.full((height, width, 3), value, dtype=np.uint8), mode="RGB"
    )
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _write_partition(root: Path, *, image_shape: list[int] | None = None) -> None:
    shape = image_shape or [4, 6, 3]
    data = root / "data" / "chunk-000"
    meta = root / "meta"
    data.mkdir(parents=True)
    meta.mkdir(parents=True)
    states = pa.FixedSizeListArray.from_arrays(
        pa.array([float(value) for value in range(22)], type=pa.float32()), 11
    )
    actions = pa.FixedSizeListArray.from_arrays(
        pa.array([0.1, 0.2, 0.3, 0.0, -0.1, -0.2, -0.3, 0.0], type=pa.float32()),
        4,
    )
    images = pa.array(
        [
            {"bytes": _png(30), "path": "frame-000000.png"},
            {"bytes": _png(220), "path": "frame-000001.png"},
        ],
        type=pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
    )
    pq.write_table(
        pa.table(
            {
                "observation.state": states,
                "action": actions,
                IMAGE_FEATURE: images,
                "episode_index": pa.array([0, 0], type=pa.int64()),
            }
        ),
        data / "file-000.parquet",
        compression="snappy",
    )
    (meta / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [11],
                        "names": [f"state_{index}" for index in range(11)],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [4],
                        "names": [
                            "action_boom",
                            "action_stick",
                            "action_bucket",
                            "action_swing",
                        ],
                    },
                    IMAGE_FEATURE: {
                        "dtype": "image",
                        "shape": shape,
                        "names": ["height", "width", "channel"],
                    },
                },
                "total_frames": 2,
            }
        ),
        encoding="utf-8",
    )
    image_stats = {
        key: [[[0.2]], [[0.3]], [[0.4]]]
        for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    image_stats["count"] = [2]
    (meta / "stats.json").write_text(
        json.dumps({IMAGE_FEATURE: image_stats}), encoding="utf-8"
    )


def _write_split(source: Path, *, image_shape: list[int] | None = None) -> None:
    _write_partition(source / "train", image_shape=image_shape)
    _write_partition(source / "validation", image_shape=image_shape)
    provenance = {
        "schema_version": MATERIALIZED_SPLIT_SCHEMA_VERSION,
        "source_dataset_sha256": "a" * 64,
        "train_repo_id": "local/source_train_swing_zero",
        "validation_repo_id": "local/source_validation_swing_zero",
        "train_root": str(source / "train"),
        "validation_root": str(source / "validation"),
        "train_dataset_sha256": _dataset_fingerprint(source / "train"),
        "validation_dataset_sha256": _dataset_fingerprint(source / "validation"),
        "train_source_episode_ids": ["episode_0001"],
        "validation_source_episode_ids": ["episode_0002"],
    }
    (source / "split_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def test_derive_constant_image_split_preserves_state_action_and_source(tmp_path: Path):
    source = tmp_path / "source"
    _write_split(source)
    source_parquet = source / "train/data/chunk-000/file-000.parquet"
    source_table = pq.read_table(source_parquet)
    source_hash = _dataset_fingerprint(source / "train")

    output = tmp_path / "state_only"
    result = derive_constant_image_split(source_root=source, output_root=output)

    derived = pq.read_table(output / "train/data/chunk-000/file-000.parquet")
    assert derived["observation.state"].to_pylist() == source_table[
        "observation.state"
    ].to_pylist()
    assert derived["action"].to_pylist() == source_table["action"].to_pylist()
    images = derived[IMAGE_FEATURE].to_pylist()
    assert len({row["bytes"] for row in images}) == 1
    constant_bytes = images[0]["bytes"]
    assert all(row["path"] == "constant_black_6x4.png" for row in images)
    with Image.open(io.BytesIO(constant_bytes)) as image:
        assert image.mode == "RGB"
        assert image.size == (6, 4)
        assert np.asarray(image).max() == 0
    assert _dataset_fingerprint(source / "train") == source_hash

    stats = json.loads((output / "train/meta/stats.json").read_text())[IMAGE_FEATURE]
    for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
        assert stats[key] == [[[0.0]], [[0.0]], [[0.0]]]
    assert stats["count"] == [2]

    split = json.loads((output / "split_provenance.json").read_text())
    assert split["train_repo_id"].endswith("_state_only_constant_image")
    assert split["train_dataset_sha256"] == _dataset_fingerprint(output / "train")
    transform = json.loads(result.provenance_path.read_text())
    assert transform["ablation"] == "state_only_constant_image"
    assert transform["constant_image"]["sha256"] == hashlib.sha256(
        constant_bytes
    ).hexdigest()
    assert transform["rewritten_frame_count"] == 4


def test_derive_constant_image_split_rejects_non_rgb_shape_without_output(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _write_split(source, image_shape=[4, 6, 1])
    output = tmp_path / "state_only"

    with pytest.raises(ValueError, match="RGB image shape"):
        derive_constant_image_split(source_root=source, output_root=output)

    assert not output.exists()
    assert not list(tmp_path.glob(".state_only.*"))


def test_derive_constant_image_split_rejects_existing_output(tmp_path: Path):
    source = tmp_path / "source"
    _write_split(source)
    output = tmp_path / "state_only"
    output.mkdir()

    with pytest.raises(ValueError, match="output already exists"):
        derive_constant_image_split(source_root=source, output_root=output)

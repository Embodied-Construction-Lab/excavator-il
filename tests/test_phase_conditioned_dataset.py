import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from excavator_il.phase_conditioned_dataset import (
    PHASE_CONDITIONED_DATASET_SCHEMA_VERSION,
    derive_phase_conditioned_video_split,
)
from excavator_il.training_split import (
    MATERIALIZED_SPLIT_SCHEMA_VERSION,
    _dataset_fingerprint,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_sequence(
    root: Path,
    *,
    file_index: int,
    episode_index: int,
    source_episode_id: str,
    frame_count: int,
) -> None:
    values = [float(index) / 10.0 for index in range(frame_count * 11)]
    states = pa.FixedSizeListArray.from_arrays(pa.array(values, pa.float32()), 11)
    actions = pa.FixedSizeListArray.from_arrays(
        pa.array([0.1, 0.2, 0.3, 0.4] * frame_count, pa.float32()), 4
    )
    table = pa.table(
        {
            "observation.state": states,
            "action": actions,
            "episode_index": pa.array([episode_index] * frame_count, pa.int64()),
            "frame_index": pa.array(range(frame_count), pa.int64()),
            "index": pa.array(range(frame_count), pa.int64()),
            "source.episode_id": pa.array([source_episode_id] * frame_count),
        }
    )
    path = root / "data/chunk-000" / f"file-{file_index:03d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_partition(
    root: Path,
    *,
    sequences: list[tuple[int, int, str, int]],
) -> None:
    for args in sequences:
        _write_sequence(
            root,
            file_index=args[0],
            episode_index=args[1],
            source_episode_id=args[2],
            frame_count=args[3],
        )
    total_frames = sum(item[3] for item in sequences)
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "total_episodes": len(sequences),
            "total_frames": total_frames,
            "total_videos": 2,
            "fps": 10,
            "splits": {"train": f"0:{len(sequences)}"},
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [11],
                    "names": [f"state_{index}" for index in range(11)],
                },
                "action": {"dtype": "float32", "shape": [4]},
                "observation.images.front": {
                    "dtype": "video",
                    "shape": [4, 6, 3],
                },
                "observation.images.dump": {
                    "dtype": "video",
                    "shape": [4, 6, 3],
                },
            },
        },
    )
    _write_json(
        root / "meta/stats.json",
        {
            "observation.state": {
                "min": [0.0] * 11,
                "max": [1.0] * 11,
                "mean": [0.5] * 11,
                "std": [0.2] * 11,
                "count": [total_frames],
                "q01": [0.0] * 11,
                "q10": [0.0] * 11,
                "q50": [0.5] * 11,
                "q90": [1.0] * 11,
                "q99": [1.0] * 11,
            }
        },
    )
    episode_rows = []
    offset = 0
    for file_index, episode_index, _source, frame_count in sequences:
        episode_rows.append(
            {
                "episode_index": episode_index,
                "length": frame_count,
                "dataset_from_index": offset,
                "dataset_to_index": offset + frame_count,
                "data/chunk_index": 0,
                "data/file_index": file_index,
            }
        )
        offset += frame_count
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), path)
    for camera in ("front", "dump"):
        video = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")


def _write_source(root: Path) -> None:
    _write_partition(root / "train", sequences=[(0, 0, "episode_0001", 6)])
    _write_partition(
        root / "validation",
        sequences=[
            (1, 0, "episode_0002", 6),
            (2, 1, "episode_0002", 1),
        ],
    )
    _write_json(
        root / "split_provenance.json",
        {
            "schema_version": MATERIALIZED_SPLIT_SCHEMA_VERSION,
            "source_dataset_sha256": "a" * 64,
            "train_repo_id": "local/train_video",
            "validation_repo_id": "local/validation_video",
            "train_root": str(root / "train"),
            "validation_root": str(root / "validation"),
            "train_dataset_sha256": _dataset_fingerprint(root / "train"),
            "validation_dataset_sha256": _dataset_fingerprint(root / "validation"),
            "train_source_episode_ids": ["episode_0001"],
            "validation_source_episode_ids": ["episode_0002"],
        },
    )


def _write_annotations(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": "excavator_act_three_phase_annotations.v1",
            "phase_order": ["DIG", "TRANSPORT", "DUMP"],
            "phase_feature_names": [
                "phase_is_dig",
                "phase_is_transport",
                "phase_is_dump",
            ],
            "entries": [
                {
                    "partition": "train",
                    "data_file_index": 0,
                    "source_episode_id": "episode_0001",
                    "frame_count": 6,
                    "dig_to_transport_frame": 2,
                    "transport_to_dump_frame": 4,
                    "disposition": "include",
                },
                {
                    "partition": "validation",
                    "data_file_index": 1,
                    "source_episode_id": "episode_0002",
                    "frame_count": 6,
                    "dig_to_transport_frame": 2,
                    "transport_to_dump_frame": 4,
                    "disposition": "include",
                },
                {
                    "partition": "validation",
                    "data_file_index": 2,
                    "source_episode_id": "episode_0002",
                    "frame_count": 1,
                    "dig_to_transport_frame": None,
                    "transport_to_dump_frame": None,
                    "disposition": "exclude_short_fragment",
                },
            ],
        },
    )


def test_derivation_appends_three_phase_one_hot_without_splitting_episode(tmp_path: Path):
    source = tmp_path / "source"
    annotations = tmp_path / "annotations.json"
    output = tmp_path / "derived"
    _write_source(source)
    _write_annotations(annotations)
    source_hash = _dataset_fingerprint(source / "train")

    result = derive_phase_conditioned_video_split(
        source_root=source,
        annotation_path=annotations,
        output_root=output,
    )

    table = pq.read_table(output / "train/data/chunk-000/file-000.parquet")
    states = table["observation.state"].to_pylist()
    assert [row[-3:] for row in states] == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    assert np.allclose(
        np.asarray(table["action"].to_pylist()),
        np.asarray([[0.1, 0.2, 0.3, 0.4]] * 6),
    )
    assert table["source.episode_id"].to_pylist() == ["episode_0001"] * 6
    info = json.loads((output / "train/meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [14]
    assert info["features"]["observation.state"]["names"][-3:] == [
        "phase_is_dig",
        "phase_is_transport",
        "phase_is_dump",
    ]
    assert _dataset_fingerprint(source / "train") == source_hash
    assert result.provenance_path.is_file()


def test_derivation_excludes_declared_terminal_fragment_and_updates_metadata(
    tmp_path: Path,
):
    source = tmp_path / "source"
    annotations = tmp_path / "annotations.json"
    output = tmp_path / "derived"
    _write_source(source)
    _write_annotations(annotations)

    derive_phase_conditioned_video_split(
        source_root=source,
        annotation_path=annotations,
        output_root=output,
    )

    assert not (output / "validation/data/chunk-000/file-002.parquet").exists()
    info = json.loads((output / "validation/meta/info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 6
    assert info["splits"] == {"train": "0:1"}
    episodes = pq.read_table(
        output / "validation/meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    assert len(episodes) == 1
    provenance = json.loads(
        (output / "phase_conditioning_provenance.json").read_text()
    )
    assert provenance["schema_version"] == PHASE_CONDITIONED_DATASET_SCHEMA_VERSION
    assert provenance["excluded_sequences"] == [
        {
            "partition": "validation",
            "data_file_index": 2,
            "source_episode_id": "episode_0002",
            "frame_count": 1,
            "reason": "exclude_short_fragment",
        }
    ]
    assert provenance["phase_frame_counts"] == {
        "train": {"DIG": 2, "TRANSPORT": 2, "DUMP": 2},
        "validation": {"DIG": 2, "TRANSPORT": 2, "DUMP": 2},
    }


def test_derivation_rejects_annotation_boundary_drift_without_output(tmp_path: Path):
    source = tmp_path / "source"
    annotations = tmp_path / "annotations.json"
    output = tmp_path / "derived"
    _write_source(source)
    _write_annotations(annotations)
    value = json.loads(annotations.read_text())
    value["entries"][0]["transport_to_dump_frame"] = 2
    _write_json(annotations, value)

    with pytest.raises(ValueError, match="phase boundaries"):
        derive_phase_conditioned_video_split(
            source_root=source,
            annotation_path=annotations,
            output_root=output,
        )

    assert not output.exists()


def test_derivation_rejects_source_fingerprint_drift_without_output(tmp_path: Path):
    source = tmp_path / "source"
    annotations = tmp_path / "annotations.json"
    output = tmp_path / "derived"
    _write_source(source)
    _write_annotations(annotations)
    (source / "train/videos/observation.images.front/chunk-000/file-000.mp4").write_bytes(
        b"changed"
    )

    with pytest.raises(ValueError, match="fingerprint"):
        derive_phase_conditioned_video_split(
            source_root=source,
            annotation_path=annotations,
            output_root=output,
        )

    assert not output.exists()


def test_derivation_rejects_incomplete_annotation_coverage_and_cleans_staging(
    tmp_path: Path,
):
    source = tmp_path / "source"
    annotations = tmp_path / "annotations.json"
    output = tmp_path / "derived"
    _write_source(source)
    _write_annotations(annotations)
    value = json.loads(annotations.read_text())
    value["entries"] = value["entries"][:-1]
    _write_json(annotations, value)

    with pytest.raises(ValueError, match="exactly cover"):
        derive_phase_conditioned_video_split(
            source_root=source,
            annotation_path=annotations,
            output_root=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".derived.*"))


def test_derivation_refuses_existing_output(tmp_path: Path):
    source = tmp_path / "source"
    annotations = tmp_path / "annotations.json"
    output = tmp_path / "derived"
    _write_source(source)
    _write_annotations(annotations)
    output.mkdir()

    with pytest.raises(ValueError, match="output already exists"):
        derive_phase_conditioned_video_split(
            source_root=source,
            annotation_path=annotations,
            output_root=output,
        )

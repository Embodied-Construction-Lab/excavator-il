import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from excavator_il.training_split import (
    MATERIALIZED_SPLIT_SCHEMA_VERSION,
    _dataset_fingerprint,
)
from excavator_il.video_training_dataset import (
    VIDEO_TRAINING_DERIVATION_SCHEMA_VERSION,
    derive_video_training_split,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source_split(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for partition in ("train", "validation"):
        partition_root = root / partition
        _write_json(
            partition_root / "meta" / "info.json",
            {
                "total_episodes": 2,
                "total_frames": 4,
                "video_path": None,
                "features": {
                    "observation.state": {"dtype": "float32"},
                    "observation.images.front": {"dtype": "image"},
                },
            },
        )
        (partition_root / "data").mkdir()
        (partition_root / "data" / "rows.bin").write_bytes(partition.encode())
        _write_json(
            partition_root / "meta" / "stats.json",
            {
                "observation.state": {"mean": [0.0]},
                "observation.images.front": {
                    "mean": [0.4, 0.5, 0.6],
                    "std": [0.1, 0.1, 0.1],
                },
            },
        )
    provenance = {
        "schema_version": MATERIALIZED_SPLIT_SCHEMA_VERSION,
        "source_dataset_sha256": "a" * 64,
        "train_repo_id": "local/source_train",
        "validation_repo_id": "local/source_validation",
        "train_dataset_sha256": _dataset_fingerprint(root / "train"),
        "validation_dataset_sha256": _dataset_fingerprint(root / "validation"),
        "train_source_episode_ids": ["episode_1"],
        "validation_source_episode_ids": ["episode_2"],
    }
    _write_json(root / "split_provenance.json", provenance)
    _write_json(root / "action_transform_provenance.json", {"transform": "test"})
    return root


def _loader(repo_id: str, *, root: Path):
    del repo_id
    info = json.loads((root / "meta" / "info.json").read_text())
    return SimpleNamespace(
        root=root,
        repo_id="local/source",
        meta=SimpleNamespace(
            total_episodes=info["total_episodes"],
            total_frames=info["total_frames"],
            video_keys=[],
        ),
    )


def _converter(*, dataset, output_dir: Path, repo_id: str, **kwargs):
    assert dataset.root.name in {"train", "validation"}
    assert kwargs == {
        "vcodec": "h264",
        "pix_fmt": "yuv420p",
        "g": 2,
        "crf": 18,
        "fast_decode": 1,
        "num_workers": 2,
        "max_frames_per_batch": 3000,
    }
    _write_json(
        output_dir / "meta" / "info.json",
        {
            "total_episodes": 2,
            "total_frames": 4,
            "video_path": (
                "videos/{video_key}/chunk-{chunk_index}/"
                "file-{file_index}.mp4"
            ),
            "features": {
                "observation.state": {"dtype": "float32"},
                "observation.images.front": {"dtype": "video"},
            },
        },
    )
    (output_dir / "data").mkdir()
    (output_dir / "data" / "rows.bin").write_bytes(repo_id.encode())
    _write_json(
        output_dir / "meta" / "stats.json",
        {"observation.state": {"mean": [0.0]}},
    )
    return SimpleNamespace(root=output_dir)


def test_derives_atomic_video_split_with_source_binding(tmp_path: Path):
    source = _source_split(tmp_path)
    output = tmp_path / "video"

    result = derive_video_training_split(
        source,
        output,
        dataset_loader=_loader,
        converter=_converter,
    )

    assert result.output_root == output.resolve()
    provenance = json.loads((output / "split_provenance.json").read_text())
    assert provenance["train_repo_id"] == "local/source_train_video"
    assert provenance["validation_repo_id"] == "local/source_validation_video"
    assert provenance["train_dataset_sha256"] == _dataset_fingerprint(output / "train")
    assert (output / "action_transform_provenance.json").is_file()
    output_stats = json.loads(
        (output / "train" / "meta" / "stats.json").read_text()
    )
    assert "observation.images.front" in output_stats
    derivation = json.loads((output / "video_training_derivation.json").read_text())
    assert derivation["schema_version"] == VIDEO_TRAINING_DERIVATION_SCHEMA_VERSION
    assert derivation["codec"] == {
        "vcodec": "h264",
        "pix_fmt": "yuv420p",
        "g": 2,
        "crf": 18,
        "fast_decode": 1,
    }
    assert derivation["camera_stats"] == {
        "method": "inherited_source_image_stats",
        "training_override": "imagenet_mean_std",
    }


def test_refuses_source_drift_before_conversion(tmp_path: Path):
    source = _source_split(tmp_path)
    (source / "train" / "data" / "rows.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        derive_video_training_split(
            source,
            tmp_path / "video",
            dataset_loader=_loader,
            converter=_converter,
        )


def test_refuses_to_overwrite_existing_output(tmp_path: Path):
    source = _source_split(tmp_path)
    output = tmp_path / "video"
    output.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        derive_video_training_split(
            source,
            output,
            dataset_loader=_loader,
            converter=_converter,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("num_workers", 0), ("num_workers", True), ("max_frames_per_batch", 0)),
)
def test_rejects_invalid_conversion_limits(
    tmp_path: Path, field: str, value: object
):
    kwargs = {"num_workers": 2, "max_frames_per_batch": 3000}
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        derive_video_training_split(
            _source_split(tmp_path),
            tmp_path / "video",
            **kwargs,
        )


def test_refuses_an_already_video_backed_source(tmp_path: Path):
    source = _source_split(tmp_path)

    def video_loader(repo_id: str, *, root: Path):
        dataset = _loader(repo_id, root=root)
        dataset.meta.video_keys = ["observation.images.front"]
        return dataset

    with pytest.raises(ValueError, match="already video-backed"):
        derive_video_training_split(
            source,
            tmp_path / "video",
            dataset_loader=video_loader,
            converter=_converter,
        )
    assert not (tmp_path / "video").exists()


def test_refuses_missing_source_camera_stats(tmp_path: Path):
    source = _source_split(tmp_path)
    stats_path = source / "train" / "meta" / "stats.json"
    _write_json(stats_path, {"observation.state": {"mean": [0.0]}})
    provenance_path = source / "split_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["train_dataset_sha256"] = _dataset_fingerprint(source / "train")
    _write_json(provenance_path, provenance)

    with pytest.raises(ValueError, match="camera stats are incomplete"):
        derive_video_training_split(
            source,
            tmp_path / "video",
            dataset_loader=_loader,
            converter=_converter,
        )

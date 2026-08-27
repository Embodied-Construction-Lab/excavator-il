import json
from pathlib import Path

import pytest

pytest.importorskip("lerobot", reason="install excavator-il[training] for split tests")

from excavator_il.lerobot_conversion import convert_episodes
from excavator_il.training_split import (
    materialize_training_split,
    prepare_training_split,
)


def _split_first_episode_into_two_segments(episode):
    (episode / "quality_report.json").write_text(
        '{"joystick_timeout_count":0,"training_segment_count":2}',
        encoding="utf-8",
    )
    (episode / "training_segments.json").write_text(
        """{
  "schema_version": "excavator_training_segments.v1",
  "parent_episode_id": "episode_0001",
  "strategy": "lerobot_episode_boundaries",
  "recovery_joystick_sample_count": 10,
  "fault_events": [],
  "unresolved_safety_event_count": 0,
  "excluded_training_step_count": 0,
  "segments": [
    {
      "segment_id": "episode_0001_segment_0000",
      "start_frame_index": 0,
      "end_frame_index_exclusive": 2,
      "step_count": 2,
      "start_state_receive_monotonic_ns": 1000000000,
      "end_state_receive_monotonic_ns": 1100000000
    },
    {
      "segment_id": "episode_0001_segment_0001",
      "start_frame_index": 2,
      "end_frame_index_exclusive": 4,
      "step_count": 2,
      "start_state_receive_monotonic_ns": 1200000000,
      "end_state_receive_monotonic_ns": 1300000000
    }
  ]
}
""",
        encoding="utf-8",
    )


def _set_collection_protocol(
    episode: Path,
    *,
    task_variant: str,
    soil_reset_block_id: str,
    dig_point_id: str = "dig_01",
) -> None:
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["collection_protocol"] = {
        "task_variant": task_variant,
        "soil_reset_block_id": soil_reset_block_id,
        "dig_point_id": dig_point_id,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_prepare_training_split_groups_by_soil_block_and_stratifies_task_variant(
    tmp_path, rgb_episode_factory
):
    episodes = []
    expected_source_to_block = {}
    expected_block_to_variant = {}
    episode_index = 1
    for variant, blocks in (
        ("dig_only", ("block_01", "block_02")),
        ("dig_transport_dump", ("block_03", "block_04")),
    ):
        for block_id in blocks:
            expected_block_to_variant[block_id] = variant
            for _ in range(2):
                episode_id = f"episode_{episode_index:04d}"
                episode = rgb_episode_factory(
                    episode_id=episode_id,
                    step_count=2,
                    dual_camera=True,
                )
                _set_collection_protocol(
                    episode,
                    task_variant=variant,
                    soil_reset_block_id=block_id,
                )
                episodes.append(episode)
                expected_source_to_block[episode_id] = block_id
                episode_index += 1

    dataset_root = tmp_path / "dataset"
    repo_id = "local/soil_block_split"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"

    split = prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
        train_ratio=0.5,
        seed=17,
    )

    assert split.schema_version == "excavator_training_split.v2"
    assert split.grouping_key == "source.soil_reset_block_id"
    assert split.source_episode_to_group == expected_source_to_block
    assert split.group_to_task_variant == expected_block_to_variant
    assert set(split.train_group_ids).isdisjoint(split.validation_group_ids)
    assert set(split.train_group_ids) | set(split.validation_group_ids) == set(
        expected_block_to_variant
    )
    for variant in ("dig_only", "dig_transport_dump"):
        assert variant in {
            split.group_to_task_variant[group_id]
            for group_id in split.train_group_ids
        }
        assert variant in {
            split.group_to_task_variant[group_id]
            for group_id in split.validation_group_ids
        }
    train_sources = set(split.train_source_episode_ids)
    validation_sources = set(split.validation_source_episode_ids)
    for source_id, block_id in expected_source_to_block.items():
        sources_for_block = {
            candidate
            for candidate, candidate_block in expected_source_to_block.items()
            if candidate_block == block_id
        }
        assert (
            sources_for_block <= train_sources
            or sources_for_block <= validation_sources
        )

    materialized = materialize_training_split(
        manifest_path=manifest_path,
        output_root=tmp_path / "splits",
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    train = LeRobotDataset(
        repo_id=materialized.train_repo_id,
        root=materialized.train_root,
    )
    validation = LeRobotDataset(
        repo_id=materialized.validation_repo_id,
        root=materialized.validation_root,
    )
    train_blocks = set(train.hf_dataset["source.soil_reset_block_id"])
    validation_blocks = set(validation.hf_dataset["source.soil_reset_block_id"])
    assert train_blocks == set(split.train_group_ids)
    assert validation_blocks == set(split.validation_group_ids)
    assert train_blocks.isdisjoint(validation_blocks)


def test_prepare_training_split_rejects_mixed_task_variant_within_soil_block(
    tmp_path, rgb_episode_factory
):
    episodes = []
    for index, (variant, block_id) in enumerate(
        (
            ("dig_only", "block_01"),
            ("dig_transport_dump", "block_01"),
            ("dig_only", "block_02"),
        ),
        start=1,
    ):
        episode = rgb_episode_factory(
            episode_id=f"episode_{index:04d}",
            step_count=2,
            dual_camera=True,
        )
        _set_collection_protocol(
            episode,
            task_variant=variant,
            soil_reset_block_id=block_id,
        )
        episodes.append(episode)
    dataset_root = tmp_path / "dataset"
    repo_id = "local/mixed_block"
    convert_episodes(episodes, dataset_root, repo_id)

    with pytest.raises(ValueError, match="contains multiple task variants"):
        prepare_training_split(
            dataset_root=dataset_root,
            repo_id=repo_id,
            output_path=tmp_path / "training_split.json",
            train_ratio=0.5,
        )


def test_prepare_training_split_rejects_partially_populated_soil_blocks(
    tmp_path, rgb_episode_factory
):
    episodes = []
    for index, block_id in enumerate(("block_01", "unknown"), start=1):
        episode = rgb_episode_factory(
            episode_id=f"episode_{index:04d}",
            step_count=2,
        )
        if block_id != "unknown":
            _set_collection_protocol(
                episode,
                task_variant="dig_only",
                soil_reset_block_id=block_id,
            )
        episodes.append(episode)
    dataset_root = tmp_path / "dataset"
    repo_id = "local/partial_soil_metadata"
    convert_episodes(episodes, dataset_root, repo_id)

    with pytest.raises(ValueError, match="only partially populated"):
        prepare_training_split(
            dataset_root=dataset_root,
            repo_id=repo_id,
            output_path=tmp_path / "training_split.json",
        )


def test_prepare_training_split_can_explicitly_group_by_episode(
    tmp_path, rgb_episode_factory
):
    episodes = []
    for index in range(1, 6):
        episode = rgb_episode_factory(
            episode_id=f"episode_{index:04d}", step_count=2
        )
        _set_collection_protocol(
            episode,
            task_variant="dig_only",
            soil_reset_block_id="block_01",
        )
        episodes.append(episode)
    dataset_root = tmp_path / "dataset"
    repo_id = "local/episode_grouping_override"
    convert_episodes(episodes, dataset_root, repo_id)

    split = prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=tmp_path / "training_split.json",
        train_ratio=0.8,
        seed=2027,
        grouping="episode",
    )

    assert split.grouping_key == "source.episode_id"
    assert len(split.train_source_episode_ids) == 4
    assert len(split.validation_source_episode_ids) == 1
    assert set(split.train_source_episode_ids).isdisjoint(
        split.validation_source_episode_ids
    )

    materialized = materialize_training_split(
        manifest_path=tmp_path / "training_split.json",
        output_root=tmp_path / "splits",
    )
    assert materialized.train_root.is_dir()
    assert materialized.validation_root.is_dir()


def test_prepare_training_split_keeps_parent_episode_segments_together(
    tmp_path, rgb_episode_factory
):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=4)
        for index in range(1, 5)
    ]
    _split_first_episode_into_two_segments(episodes[0])
    dataset_root = tmp_path / "dataset"
    repo_id = "local/parent_episode_split"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"

    split = prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
        train_ratio=0.5,
        seed=7,
    )

    assert set(split.train_source_episode_ids).isdisjoint(
        split.validation_source_episode_ids
    )
    assert set(split.train_source_episode_ids) | set(
        split.validation_source_episode_ids
    ) == {f"episode_{index:04d}" for index in range(1, 5)}
    assert len(split.train_lerobot_episode_indices) + len(
        split.validation_lerobot_episode_indices
    ) == 5

    first_parent_indices = {
        index
        for index, source_id in split.lerobot_episode_to_source.items()
        if source_id == "episode_0001"
    }
    assert len(first_parent_indices) == 2
    assert first_parent_indices <= set(split.train_lerobot_episode_indices) or (
        first_parent_indices <= set(split.validation_lerobot_episode_indices)
    )

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "excavator_training_split.v2"
    assert persisted["grouping_key"] == "source.episode_id"
    assert len(persisted["source_dataset_sha256"]) == 64
    assert persisted["train_lerobot_episode_indices"] == list(
        split.train_lerobot_episode_indices
    )


def test_prepare_training_split_rejects_pipeline_only_dataset(
    tmp_path, rgb_episode_factory
):
    episodes = [
        rgb_episode_factory(episode_id=f"synthetic_episode_{index:04d}", step_count=3)
        for index in range(2)
    ]
    dataset_root = tmp_path / "synthetic_dataset"
    convert_episodes(episodes, dataset_root, "local/synthetic_fixture")
    (dataset_root / "pipeline_validation.json").write_text(
        json.dumps(
            {
                "contains_synthetic_episodes": True,
                "training_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not eligible for training"):
        prepare_training_split(
            dataset_root=dataset_root,
            repo_id="local/synthetic_fixture",
            output_path=tmp_path / "training_split.json",
        )


def test_prepare_training_split_reuses_matching_manifest_and_rejects_drift(
    tmp_path, rgb_episode_factory
):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(4)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/stable_split"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"

    original = prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
        train_ratio=0.75,
        seed=11,
    )
    reused = prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
        train_ratio=0.75,
        seed=11,
    )
    assert reused == original

    with pytest.raises(ValueError, match="does not match requested split"):
        prepare_training_split(
            dataset_root=dataset_root,
            repo_id=repo_id,
            output_path=manifest_path,
            train_ratio=0.5,
            seed=11,
        )


def test_materialize_training_split_recomputes_subset_stats(
    tmp_path, rgb_episode_factory
):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(4)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/materialized_split"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"
    split = prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
        train_ratio=0.75,
        seed=3,
    )

    materialized = materialize_training_split(
        manifest_path=manifest_path,
        output_root=tmp_path / "splits",
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    train = LeRobotDataset(
        repo_id=materialized.train_repo_id,
        root=materialized.train_root,
    )
    validation = LeRobotDataset(
        repo_id=materialized.validation_repo_id,
        root=materialized.validation_root,
    )
    assert train.num_episodes == len(split.train_lerobot_episode_indices)
    assert validation.num_episodes == len(split.validation_lerobot_episode_indices)
    assert set(train.hf_dataset["source.episode_id"]) == set(
        split.train_source_episode_ids
    )
    assert set(validation.hf_dataset["source.episode_id"]) == set(
        split.validation_source_episode_ids
    )
    assert (materialized.train_root / "meta" / "stats.json").is_file()
    assert (materialized.validation_root / "meta" / "stats.json").is_file()
    assert materialized.provenance_path.is_file()


def test_materialize_training_split_accepts_legacy_v1_manifest(
    tmp_path, rgb_episode_factory
):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(4)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/legacy_v1_split"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
        train_ratio=0.5,
        seed=5,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    legacy_manifest = {
        key: value for key, value in manifest.items() if key in legacy_fields
    }
    legacy_manifest["schema_version"] = "excavator_training_split.v1"
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")

    materialized = materialize_training_split(
        manifest_path=manifest_path,
        output_root=tmp_path / "splits",
    )

    provenance = json.loads(materialized.provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == "excavator_materialized_training_split.v1"
    assert set(provenance["train_source_episode_ids"]) == set(
        legacy_manifest["train_source_episode_ids"]
    )


def test_materialize_training_split_cleans_staging_after_failure(
    tmp_path, rgb_episode_factory, monkeypatch
):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(2)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/failed_materialization"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
    )
    output_root = tmp_path / "splits"

    from lerobot.datasets import dataset_tools

    def fail_after_writing(dataset, splits, output_dir):
        (Path(output_dir) / "train").mkdir(parents=True)
        raise RuntimeError("copy failed")

    monkeypatch.setattr(dataset_tools, "split_dataset", fail_after_writing)

    with pytest.raises(RuntimeError, match="copy failed"):
        materialize_training_split(
            manifest_path=manifest_path,
            output_root=output_root,
        )

    assert not output_root.exists()
    assert not list(tmp_path.glob(".splits.*"))


def test_materialize_training_split_rejects_tampered_episode_mapping(
    tmp_path, rgb_episode_factory
):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(2)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/tampered_split"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lerobot_episode_to_source"]["0"] = "episode_tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match="invalid training split manifest|mapping no longer matches"
    ):
        materialize_training_split(
            manifest_path=manifest_path,
            output_root=tmp_path / "splits",
        )


def test_materialize_training_split_rejects_tampered_soil_group_mapping(
    tmp_path, rgb_episode_factory
):
    episodes = []
    for index in range(1, 5):
        episode = rgb_episode_factory(
            episode_id=f"episode_{index:04d}",
            step_count=2,
            dual_camera=True,
        )
        _set_collection_protocol(
            episode,
            task_variant="dig_only",
            soil_reset_block_id=f"block_{index:02d}",
        )
        episodes.append(episode)
    dataset_root = tmp_path / "dataset"
    repo_id = "local/tampered_soil_groups"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
        train_ratio=0.5,
        seed=3,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group_id = next(iter(manifest["group_to_task_variant"]))
    manifest["group_to_task_variant"][group_id] = "tampered_variant"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="grouping no longer matches"):
        materialize_training_split(
            manifest_path=manifest_path,
            output_root=tmp_path / "splits",
        )


def test_materialize_training_split_rejects_source_dataset_content_drift(
    tmp_path, rgb_episode_factory
):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=2)
        for index in range(2)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/source_content_drift"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
    )
    (dataset_root / "unexpected_after_split.txt").write_text(
        "dataset changed",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content no longer matches"):
        materialize_training_split(
            manifest_path=manifest_path,
            output_root=tmp_path / "splits",
        )


def test_materialize_training_split_rejects_parent_leak_in_manifest(
    tmp_path, rgb_episode_factory
):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=4)
        for index in range(1, 3)
    ]
    _split_first_episode_into_two_segments(episodes[0])
    dataset_root = tmp_path / "dataset"
    repo_id = "local/parent_leak"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
        train_ratio=0.5,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_indices = [
        int(index)
        for index, source in manifest["lerobot_episode_to_source"].items()
        if source == "episode_0001"
    ]
    first, second = parent_indices
    manifest["train_lerobot_episode_indices"] = [first]
    manifest["validation_lerobot_episode_indices"] = [
        index
        for index in range(3)
        if index != first
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="parent Episode partition is inconsistent"):
        materialize_training_split(
            manifest_path=manifest_path,
            output_root=tmp_path / "splits",
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("dataset_root", 123),
        ("repo_id", ""),
        ("seed", True),
        ("train_ratio", "0.8"),
        ("train_source_episode_ids", "episode_0001"),
        ("validation_source_episode_ids", [1]),
        ("train_source_episode_ids", ["episode_0001", "episode_0001"]),
        ("train_lerobot_episode_indices", "0"),
        ("validation_lerobot_episode_indices", [0.5]),
        ("lerobot_episode_to_source", []),
        ("source_dataset_sha256", "not-a-sha256"),
        ("grouping_key", 42),
        ("train_group_ids", "episode_0001"),
        ("source_episode_to_group", []),
        ("group_to_task_variant", {"episode_0001": ""}),
    ],
)
def test_materialize_training_split_rejects_malformed_manifest_types(
    tmp_path, rgb_episode_factory, field, invalid_value
):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(2)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/malformed_manifest"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest_path = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_path=manifest_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = invalid_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid training split manifest"):
        materialize_training_split(
            manifest_path=manifest_path,
            output_root=tmp_path / "splits",
        )

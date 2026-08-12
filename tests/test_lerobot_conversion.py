import json
import shutil

import numpy as np
import pytest

pytest.importorskip("lerobot", reason="install excavator-il[training] for conversion tests")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from excavator_il.lerobot_conversion import convert_episodes


def test_convert_requires_explicit_opt_in_for_synthetic_episode(
    tmp_path, rgb_episode_factory
):
    episode = rgb_episode_factory(step_count=3)
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["synthetic_provenance"] = {
        "source_episode_id": metadata["episode_id"],
        "method": "exact_duplicate_for_pipeline_validation",
        "training_eligible": False,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="allow-synthetic"):
        convert_episodes(
            episode_paths=[episode],
            output_root=tmp_path / "rejected",
            repo_id="local/rejected_synthetic",
        )

    summary = convert_episodes(
        episode_paths=[episode],
        output_root=tmp_path / "allowed",
        repo_id="local/allowed_synthetic",
        allow_synthetic=True,
    )
    assert summary.frame_count == 3
    marker = json.loads(
        (tmp_path / "allowed" / "pipeline_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["training_eligible"] is False
    assert marker["contains_synthetic_episodes"] is True
    assert marker["synthetic_episode_ids"] == [metadata["episode_id"]]
    assert marker["source_episode_ids"] == [metadata["episode_id"]]


def test_convert_rejects_mixed_real_and_synthetic_episodes(
    tmp_path, rgb_episode_factory
):
    real = rgb_episode_factory(episode_id="episode_real", step_count=3)
    synthetic = rgb_episode_factory(episode_id="episode_synthetic", step_count=3)
    metadata_path = synthetic / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["synthetic_provenance"] = {
        "source_episode_id": "episode_real",
        "method": "exact_duplicate_for_pipeline_validation",
        "training_eligible": False,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="must not be mixed"):
        convert_episodes(
            episode_paths=[real, synthetic],
            output_root=tmp_path / "mixed",
            repo_id="local/mixed",
            allow_synthetic=True,
        )


def test_convert_rejects_invalid_synthetic_provenance(tmp_path, rgb_episode_factory):
    episode = rgb_episode_factory(step_count=3)
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["synthetic_provenance"] = {
        "source_episode_id": metadata["episode_id"],
        "method": "unknown",
        "training_eligible": True,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid synthetic provenance"):
        convert_episodes(
            episode_paths=[episode],
            output_root=tmp_path / "invalid",
            repo_id="local/invalid_synthetic",
            allow_synthetic=True,
        )


def test_convert_rejects_duplicate_paths_and_episode_ids(tmp_path, rgb_episode_factory):
    episode = rgb_episode_factory(episode_id="episode_0001", step_count=3)

    with pytest.raises(ValueError, match="duplicate Episode path"):
        convert_episodes(
            episode_paths=[episode, episode],
            output_root=tmp_path / "duplicate_path",
            repo_id="local/duplicate_path",
        )

    duplicate_id = tmp_path / "different_path"
    shutil.copytree(episode, duplicate_id)
    with pytest.raises(ValueError, match="duplicate episode_id"):
        convert_episodes(
            episode_paths=[episode, duplicate_id],
            output_root=tmp_path / "duplicate_id",
            repo_id="local/duplicate_id",
        )


def test_convert_episode_builds_rgb_state_action_lerobot_dataset(tmp_path, rgb_episode_factory):
    episode = rgb_episode_factory(step_count=3)
    output = tmp_path / "lerobot_dataset"

    summary = convert_episodes(
        episode_paths=[episode],
        output_root=output,
        repo_id="local/excavator_dig_v1",
    )

    assert summary.episode_count == 1
    assert summary.frame_count == 3
    assert summary.fps == 10

    dataset = LeRobotDataset(repo_id="local/excavator_dig_v1", root=output)
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3
    assert dataset.features["observation.state"]["shape"] == (11,)
    assert dataset.features["action"]["shape"] == (4,)
    assert dataset.features["observation.images.front"]["shape"] == (24, 32, 3)

    row = dataset.hf_dataset[0]
    np.testing.assert_allclose(
        row["observation.state"],
        [0.15, 0.14, 0.13, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 0.0, 0.0],
    )
    np.testing.assert_allclose(row["action"], [0.2, -0.3, 0.4, 0.0])


def test_convert_preserves_all_source_episode_boundaries(tmp_path, rgb_episode_factory):
    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(1, 6)
    ]
    output = tmp_path / "five_episode_dataset"

    summary = convert_episodes(
        episode_paths=episodes,
        output_root=output,
        repo_id="local/five_episode_dataset",
    )

    dataset = LeRobotDataset(repo_id="local/five_episode_dataset", root=output)
    assert summary.source_episode_count == 5
    assert summary.episode_count == 5
    assert summary.frame_count == 15
    assert dataset.num_episodes == 5
    assert dataset.num_frames == 15
    assert set(dataset.hf_dataset["source.episode_id"]) == {
        f"episode_{index:04d}" for index in range(1, 6)
    }


def test_convert_fails_if_reloaded_dataset_is_incomplete(
    tmp_path, rgb_episode_factory, monkeypatch
):
    episode = rgb_episode_factory(step_count=3)

    class _IncompleteDataset:
        num_episodes = 0
        num_frames = 0

    from excavator_il import lerobot_conversion

    real_dataset = lerobot_conversion.LeRobotDataset

    class _DatasetProxy:
        create = real_dataset.create

        def __new__(cls, *args, **kwargs):
            if kwargs.get("root") == tmp_path / "incomplete":
                return _IncompleteDataset()
            return real_dataset(*args, **kwargs)

    monkeypatch.setattr(lerobot_conversion, "LeRobotDataset", _DatasetProxy)

    with pytest.raises(RuntimeError, match="episode count mismatch"):
        convert_episodes(
            episode_paths=[episode],
            output_root=tmp_path / "incomplete",
            repo_id="local/incomplete",
        )


def test_convert_uses_lerobot_episode_boundaries_for_training_segments(
    tmp_path, rgb_episode_factory
):
    episode = rgb_episode_factory(step_count=5)
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
      "end_frame_index_exclusive": 5,
      "step_count": 3,
      "start_state_receive_monotonic_ns": 1200000000,
      "end_state_receive_monotonic_ns": 1400000000
    }
  ]
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "segmented_lerobot_dataset"

    summary = convert_episodes(
        episode_paths=[episode],
        output_root=output,
        repo_id="local/excavator_segmented_v1",
    )

    assert summary.source_episode_count == 1
    assert summary.episode_count == 2
    assert summary.frame_count == 5
    dataset = LeRobotDataset(
        repo_id="local/excavator_segmented_v1",
        root=output,
        delta_timestamps={"action": [0.0, 0.1, 0.2, 0.3]},
    )
    assert dataset.num_episodes == 2
    assert dataset.hf_dataset[0]["source.episode_id"] == "episode_0001"
    assert (
        dataset.hf_dataset[0]["source.segment_id"]
        == "episode_0001_segment_0000"
    )
    assert dataset.hf_dataset[2]["source.frame_index"] == "2"
    assert dataset[1]["action_is_pad"].tolist() == [False, True, True, True]
    assert dataset[2]["action_is_pad"].tolist() == [False, False, False, True]

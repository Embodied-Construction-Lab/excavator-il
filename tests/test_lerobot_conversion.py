import numpy as np
import pytest

pytest.importorskip("lerobot", reason="install excavator-il[training] for conversion tests")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from excavator_il.lerobot_conversion import convert_episodes


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

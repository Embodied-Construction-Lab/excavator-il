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

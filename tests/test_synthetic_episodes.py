import csv
import json
import os

import numpy as np
import pytest
from PIL import Image

from excavator_il.synthetic_episodes import synthesize_episodes


def test_synthesize_episodes_assigns_unique_ids_and_records_provenance(
    tmp_path, rgb_episode_factory
):
    source = rgb_episode_factory(episode_id="episode_0004", step_count=2)
    output = tmp_path / "synthetic"

    result = synthesize_episodes(source, output, count=2)

    assert result.source_episode_id == "episode_0004"
    assert result.episode_ids == (
        "synthetic_episode_0001",
        "synthetic_episode_0002",
    )
    assert result.image_storage == "hardlink"
    assert result.training_eligible is False

    first = output / "synthetic_episode_0001"
    metadata = json.loads((first / "episode.json").read_text(encoding="utf-8"))
    assert metadata["episode_id"] == "synthetic_episode_0001"
    assert metadata["synthetic_provenance"] == {
        "source_episode_id": "episode_0004",
        "method": "exact_duplicate_for_pipeline_validation",
        "training_eligible": False,
    }
    with (first / "steps.csv").open(newline="", encoding="utf-8") as stream:
        assert {row["episode_id"] for row in csv.DictReader(stream)} == {
            "synthetic_episode_0001"
        }
    assert os.stat(source / "camera_front/000000.png").st_ino == os.stat(
        first / "camera_front/000000.png"
    ).st_ino
    np.testing.assert_array_equal(
        np.asarray(Image.open(source / "camera_front/000000.png")),
        np.asarray(Image.open(first / "camera_front/000000.png")),
    )


def test_synthesize_removes_only_new_outputs_after_copy_failure(
    tmp_path, rgb_episode_factory, monkeypatch
):
    import excavator_il.synthetic_episodes as module

    source = rgb_episode_factory(episode_id="episode_0004", step_count=2)
    output = tmp_path / "synthetic"
    original_copy = module._copy_tree
    call_count = 0

    def fail_second_copy(source_path, destination_path):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            destination_path.mkdir(parents=True)
            raise OSError("injected copy failure")
        original_copy(source_path, destination_path)

    monkeypatch.setattr(module, "_copy_tree", fail_second_copy)

    with pytest.raises(OSError, match="injected copy failure"):
        synthesize_episodes(source, output, count=2)

    assert list(output.iterdir()) == []


def test_synthesize_rejects_output_inside_source(tmp_path, rgb_episode_factory):
    source = rgb_episode_factory(episode_id="episode_0004", step_count=2)

    with pytest.raises(ValueError, match="must not be inside"):
        synthesize_episodes(source, source / "synthetic", count=2)

    assert not (source / "synthetic").exists()

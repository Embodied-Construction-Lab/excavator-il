import math
import json
from dataclasses import replace

import pytest

pytest.importorskip("lerobot", reason="install excavator-il[training] for ACT tests")

from excavator_il.checkpoint_evaluation import evaluate_act_checkpoints
import excavator_il.checkpoint_evaluation as checkpoint_evaluation
from excavator_il.lerobot_conversion import convert_episodes
from excavator_il.training_split import (
    materialize_training_split,
    prepare_training_split,
)


def _write_checkpoint(
    checkpoint,
    dataset,
    *,
    train_root,
    train_repo_id,
    action_bias=0.0,
    action_std_override=None,
):
    import packaging.version  # noqa: F401 - required by safetensors loader
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(11,)),
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 24, 32)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        },
        device="cpu",
        push_to_hub=False,
        chunk_size=20,
        n_action_steps=10,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=16,
        n_vae_encoder_layers=1,
    )
    policy = ACTPolicy(config)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.model.action_head.bias.fill_(action_bias)
    preprocessor, postprocessor = make_pre_post_processors(
        config, dataset_stats=dataset.meta.stats
    )
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    (checkpoint / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "root": str(train_root),
                    "repo_id": train_repo_id,
                }
            }
        ),
        encoding="utf-8",
    )
    if action_std_override is not None:
        from safetensors.torch import load_file, save_file

        state_path = (
            checkpoint
            / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
        state = dict(load_file(state_path))
        state["action.std"] = torch.full_like(
            state["action.std"], float(action_std_override)
        )
        save_file(state, state_path)


def test_evaluate_act_checkpoints_selects_safe_lowest_validation_loss(
    tmp_path, rgb_episode_factory
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(2)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/checkpoint_validation"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root, repo_id=repo_id, output_path=manifest
    )
    split = materialize_training_split(
        manifest_path=manifest, output_root=tmp_path / "split"
    )
    dataset = LeRobotDataset(repo_id=split.train_repo_id, root=split.train_root)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint,
        dataset,
        train_root=split.train_root.resolve(),
        train_repo_id=split.train_repo_id,
    )

    result = evaluate_act_checkpoints(
        checkpoint_paths=[checkpoint],
        split_root=tmp_path / "split",
        device="cpu",
        batch_size=2,
        num_workers=0,
    )

    assert result.selected_checkpoint == checkpoint
    assert result.selection_reason == "lowest safe validation deployment-prior L1"
    assert len(result.checkpoints) == 1
    metric = result.checkpoints[0]
    assert metric.validation_frame_count == 3
    assert math.isfinite(metric.deployment_prior_l1)
    assert metric.all_finite is True
    assert metric.out_of_range_sample_count == 0
    assert -1.0 <= metric.action_min <= metric.action_max <= 1.0


def test_evaluate_act_checkpoints_never_selects_out_of_range_checkpoint(
    tmp_path, rgb_episode_factory
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(2)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/unsafe_checkpoint_validation"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root, repo_id=repo_id, output_path=manifest
    )
    split = materialize_training_split(
        manifest_path=manifest, output_root=tmp_path / "split"
    )
    dataset = LeRobotDataset(repo_id=split.train_repo_id, root=split.train_root)
    checkpoint = tmp_path / "unsafe_checkpoint"
    _write_checkpoint(
        checkpoint,
        dataset,
        train_root=split.train_root.resolve(),
        train_repo_id=split.train_repo_id,
        action_bias=100.0,
        action_std_override=1.0,
    )

    result = evaluate_act_checkpoints(
        checkpoint_paths=[checkpoint],
        split_root=tmp_path / "split",
        device="cpu",
        batch_size=2,
        num_workers=0,
    )

    assert result.selected_checkpoint is None
    assert result.selection_reason.startswith("no checkpoint passed")
    assert result.checkpoints[0].out_of_range_sample_count == 3


def test_evaluate_act_checkpoints_rejects_checkpoint_from_other_training_dataset(
    tmp_path, rgb_episode_factory
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(2)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/provenance_validation"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root, repo_id=repo_id, output_path=manifest
    )
    split = materialize_training_split(
        manifest_path=manifest, output_root=tmp_path / "split"
    )
    train = LeRobotDataset(repo_id=split.train_repo_id, root=split.train_root)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint,
        train,
        train_root=tmp_path / "different_train",
        train_repo_id="local/different_train",
    )

    with pytest.raises(ValueError, match="was not trained on"):
        evaluate_act_checkpoints(
            checkpoint_paths=[checkpoint],
            split_root=tmp_path / "split",
        )


def test_checkpoint_ranking_uses_raw_action_space_across_processor_stats(
    tmp_path, rgb_episode_factory
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(2)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/raw_space_ranking"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root, repo_id=repo_id, output_path=manifest
    )
    split = materialize_training_split(
        manifest_path=manifest, output_root=tmp_path / "split"
    )
    train = LeRobotDataset(repo_id=split.train_repo_id, root=split.train_root)
    accurate = tmp_path / "accurate"
    distorted = tmp_path / "distorted"
    _write_checkpoint(
        accurate,
        train,
        train_root=split.train_root.resolve(),
        train_repo_id=split.train_repo_id,
    )
    _write_checkpoint(
        distorted,
        train,
        train_root=split.train_root.resolve(),
        train_repo_id=split.train_repo_id,
        action_bias=0.5,
        action_std_override=0.1,
    )

    result = evaluate_act_checkpoints(
        checkpoint_paths=[distorted, accurate],
        split_root=tmp_path / "split",
        batch_size=2,
    )

    assert result.selected_checkpoint == accurate
    metrics = {metric.checkpoint_path: metric for metric in result.checkpoints}
    assert metrics[accurate].deployment_prior_l1 < metrics[distorted].deployment_prior_l1


def test_evaluate_act_checkpoints_rejects_dataset_change_during_evaluation(
    tmp_path, rgb_episode_factory, monkeypatch
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = [
        rgb_episode_factory(episode_id=f"episode_{index:04d}", step_count=3)
        for index in range(2)
    ]
    dataset_root = tmp_path / "dataset"
    repo_id = "local/evaluation_toctou"
    convert_episodes(episodes, dataset_root, repo_id)
    manifest = tmp_path / "training_split.json"
    prepare_training_split(
        dataset_root=dataset_root, repo_id=repo_id, output_path=manifest
    )
    split_root = tmp_path / "split"
    split = materialize_training_split(
        manifest_path=manifest, output_root=split_root
    )
    train = LeRobotDataset(repo_id=split.train_repo_id, root=split.train_root)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint,
        train,
        train_root=split.train_root.resolve(),
        train_repo_id=split.train_repo_id,
    )
    original = checkpoint_evaluation._evaluate_checkpoint

    def mutate_validation_dataset(**kwargs):
        metric = original(**kwargs)
        (split.validation_root / "changed_during_evaluation").write_text(
            "changed", encoding="utf-8"
        )
        return replace(metric)

    monkeypatch.setattr(
        checkpoint_evaluation, "_evaluate_checkpoint", mutate_validation_dataset
    )

    with pytest.raises(ValueError, match="changed during checkpoint evaluation"):
        evaluate_act_checkpoints(
            checkpoint_paths=[checkpoint],
            split_root=split_root,
            batch_size=2,
        )

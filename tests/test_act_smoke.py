import math

import pytest

pytest.importorskip("lerobot", reason="install excavator-il[training] for ACT tests")

from excavator_il.act_smoke import (
    run_act_checkpoint_inference,
    run_act_smoke_train_step,
)


def test_act_smoke_train_step_accepts_rgb_state_and_four_axis_action():
    result = run_act_smoke_train_step(
        image_shape=(3, 32, 32),
        state_dim=11,
        action_dim=4,
        chunk_size=3,
    )

    assert math.isfinite(result.loss)
    assert result.loss > 0.0
    assert result.predicted_chunk_shape == (1, 3, 4)


def test_act_checkpoint_inference_loads_saved_processors_and_dataset_sample(
    tmp_path, rgb_episode_factory
):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    from excavator_il.lerobot_conversion import convert_episodes

    episode = rgb_episode_factory(step_count=3)
    dataset_root = tmp_path / "dataset"
    repo_id = "local/checkpoint_inference_fixture"
    convert_episodes([episode], dataset_root, repo_id)
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)

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
        chunk_size=3,
        n_action_steps=1,
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
    checkpoint = tmp_path / "checkpoint"
    policy = ACTPolicy(config)
    preprocessor, postprocessor = make_pre_post_processors(
        config, dataset_stats=dataset.meta.stats
    )
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)

    result = run_act_checkpoint_inference(
        checkpoint_path=checkpoint,
        dataset_root=dataset_root,
        repo_id=repo_id,
        sample_index=0,
        device="cpu",
    )

    assert result.predicted_chunk_shape == (1, 3, 4)
    assert result.action_dim == 4
    assert result.all_finite is True
    assert math.isfinite(result.action_min)
    assert math.isfinite(result.action_max)


def test_act_contract_rejects_non_authoritative_action_shape():
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature

    from excavator_il.act_smoke import _validate_excavator_act_contract

    config = SimpleNamespace(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(11,)),
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(5,)),
        },
    )
    dataset = SimpleNamespace(
        features={
            "observation.state": {"shape": (11,), "names": ["state"] * 11},
            "observation.images.front": {"shape": (480, 640, 3)},
            "action": {"shape": (4,), "names": ["boom", "stick", "bucket", "swing"]},
        }
    )

    with pytest.raises(ValueError, match="four-axis action"):
        _validate_excavator_act_contract(config, dataset)


def test_act_contract_reports_missing_dataset_feature_as_value_error():
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature

    from excavator_il.act_smoke import _validate_excavator_act_contract

    config = SimpleNamespace(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(11,)),
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        },
    )

    with pytest.raises(ValueError, match="missing required feature"):
        _validate_excavator_act_contract(config, SimpleNamespace(features={}))

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

    assert result.predicted_chunk_shape == (1, 20, 4)
    assert result.action_dim == 4
    assert result.all_finite is True
    assert math.isfinite(result.action_min)
    assert math.isfinite(result.action_max)
    assert result.inference_ms > 0.0
    assert result.inference_min_ms > 0.0
    assert result.inference_max_ms >= result.inference_min_ms
    assert result.timed_runs == 1
    assert result.peak_cuda_memory_mb is None


@pytest.mark.parametrize(
    ("warmup_runs", "timed_runs"),
    [(-1, 1), (0, 0)],
)
def test_act_checkpoint_inference_rejects_invalid_benchmark_counts(
    tmp_path, warmup_runs, timed_runs
):
    checkpoint = tmp_path / "checkpoint"
    dataset = tmp_path / "dataset"
    checkpoint.mkdir()
    dataset.mkdir()

    with pytest.raises(ValueError, match="warmup runs"):
        run_act_checkpoint_inference(
            checkpoint_path=checkpoint,
            dataset_root=dataset,
            repo_id="local/unused",
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )


def test_summarize_inference_rejects_nonfinite_earlier_run():
    import torch

    from excavator_il.act_smoke import _summarize_action_chunks

    chunks = [
        torch.tensor([[[float("nan"), 0.0, 0.0, 0.0]]]),
        torch.zeros((1, 1, 4)),
    ]

    with pytest.raises(ValueError, match="non-finite"):
        _summarize_action_chunks(chunks)


def test_summarize_inference_uses_all_runs_for_range():
    import torch

    from excavator_il.act_smoke import _summarize_action_chunks

    minimum, maximum = _summarize_action_chunks(
        [torch.tensor([[[-0.8, 0.0, 0.0, 0.0]]]), torch.full((1, 1, 4), 0.7)]
    )

    assert minimum == pytest.approx(-0.8)
    assert maximum == pytest.approx(0.7)


def test_summarize_inference_reports_the_saturated_runtime_range():
    import torch

    from excavator_il.act_smoke import _summarize_action_chunks

    minimum, maximum = _summarize_action_chunks(
        [torch.tensor([[[1.01, -1.024, 0.0, 0.0]]])]
    )

    assert minimum == -1.0
    assert maximum == 1.0


def test_summarize_inference_rejects_gross_raw_action_outlier():
    import torch

    from excavator_il.act_smoke import _summarize_action_chunks

    with pytest.raises(ValueError, match="raw action magnitude"):
        _summarize_action_chunks(
            [torch.tensor([[[100.0, -0.25, 0.0, 0.0]]])]
        )


def test_enforce_inference_budget_fails_closed():
    from excavator_il.act_smoke import _enforce_inference_budget

    with pytest.raises(ValueError, match="exceeds 100.000 ms"):
        _enforce_inference_budget(101.0, 100.0)


def test_enforce_inference_budget_accepts_disabled_or_passing_limit():
    from excavator_il.act_smoke import _enforce_inference_budget

    _enforce_inference_budget(101.0, None)
    _enforce_inference_budget(99.0, 100.0)


@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), float("inf")])
def test_enforce_inference_budget_rejects_invalid_limit(limit):
    from excavator_il.act_smoke import _enforce_inference_budget

    with pytest.raises(ValueError, match="must be positive"):
        _enforce_inference_budget(0.0, limit)


def test_act_contract_rejects_non_authoritative_action_shape():
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature

    from excavator_il.act_smoke import _validate_excavator_act_contract

    config = SimpleNamespace(
        chunk_size=20,
        n_action_steps=10,
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
        chunk_size=20,
        n_action_steps=10,
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


def test_act_contract_rejects_temporal_ensemble_or_extra_camera():
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature
    from excavator_il.act_smoke import _validate_excavator_act_contract
    from excavator_il.lerobot_conversion import STATE_FIELDS
    from excavator_il.raw_episode import ACTION_FIELDS

    inputs = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(11,)),
        "observation.images.front": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        ),
        "observation.images.wrist": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        ),
    }
    config = SimpleNamespace(
        chunk_size=20,
        n_action_steps=10,
        temporal_ensemble_coeff=0.01,
        input_features=inputs,
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))
        },
    )
    dataset = SimpleNamespace(
        features={
            "observation.state": {"shape": (11,), "names": STATE_FIELDS},
            "observation.images.front": {"shape": (480, 640, 3)},
            "action": {"shape": (4,), "names": ACTION_FIELDS},
        }
    )

    with pytest.raises(ValueError, match="temporal ensemble"):
        _validate_excavator_act_contract(config, dataset)


def test_act_contract_accepts_front_and_dump_rgb_cameras():
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature

    from excavator_il.act_smoke import _validate_excavator_act_contract
    from excavator_il.lerobot_conversion import STATE_FIELDS
    from excavator_il.raw_episode import ACTION_FIELDS

    config = SimpleNamespace(
        chunk_size=20,
        n_action_steps=10,
        temporal_ensemble_coeff=None,
        input_features={
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(11,)
            ),
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            ),
            "observation.images.dump": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))
        },
    )
    dataset = SimpleNamespace(
        features={
            "observation.state": {"shape": (11,), "names": STATE_FIELDS},
            "observation.images.front": {"shape": (480, 640, 3)},
            "observation.images.dump": {"shape": (480, 640, 3)},
            "action": {"shape": (4,), "names": ACTION_FIELDS},
        }
    )

    _validate_excavator_act_contract(config, dataset)

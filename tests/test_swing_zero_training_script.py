from pathlib import Path


def test_swing_zero_training_script_preserves_runtime_contract_and_long_run():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_act_swing_zero_200k.sh"
    ).read_text(encoding="utf-8")

    assert "derive-zero-swing-split" in script
    assert "--policy.chunk_size=20" in script
    assert "--policy.n_action_steps=10" in script
    assert "--batch_size=2" in script
    assert "--num_workers=2" in script
    assert "--steps=200000" in script
    assert "--save_freq=20000" in script
    assert "--seed=2026" in script
    assert "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1" in script
    assert "--dataset.image_transforms.enable=false" in script
    assert "pipeline_validation.json" in script
    assert "training output already exists" in script

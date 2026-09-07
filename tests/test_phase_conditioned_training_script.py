from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_phase_conditioned_transport_dump_act.sh"


def test_phase_training_uses_isolated_dataset_output_and_live_log():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "icra2027_transport_dump_dual_rgb_three_phase_video" in script
    assert "icra2027_transport_dump_three_phase_video_b${BATCH_SIZE}_seed${SEED}" in script
    assert "outputs/icra2027_transport_dump_dual_rgb_video_b4_seed2027" not in script
    assert "2>&1 | tee" in script
    assert "another lerobot-train process is already running" in script
    assert "refusing to overwrite" in script


def test_phase_training_preserves_act_architecture_and_sample_exposure():
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'SAMPLE_BUDGET="${ACT_PHASE_SAMPLE_BUDGET:-600000}"' in script
    assert 'BATCH_SIZE="${ACT_PHASE_BATCH_SIZE:-4}"' in script
    assert 'STEPS=$(((SAMPLE_BUDGET + BATCH_SIZE - 1) / BATCH_SIZE))' in script
    assert "--policy.chunk_size=20" in script
    assert "--policy.n_action_steps=10" in script
    assert "--policy.vision_backbone=resnet18" in script
    assert "--policy.dim_model=512" in script
    assert "--policy.n_heads=8" in script
    assert "--policy.dim_feedforward=3200" in script
    assert "--policy.n_encoder_layers=4" in script
    assert "--policy.n_decoder_layers=1" in script
    assert "--policy.latent_dim=32" in script
    assert "--policy.n_vae_encoder_layers=4" in script
    assert "--policy.kl_weight=10.0" in script
    assert "--policy.optimizer_lr=1e-5" in script


def test_phase_training_preflight_requires_14d_one_hot_and_parent_isolation():
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'state_feature.get("shape") != [14]' in script
    assert 'phase.sum(axis=1)' in script
    assert "train_source_episode_ids" in script
    assert "validation_source_episode_ids" in script
    assert "source Episode leakage" in script
    assert "excluded_sequences" in script
    assert "episode_0143" in script
    assert "phase_annotations_sha256" in script
    assert "dataset_deriver_sha256" in script
    assert "training_script_sha256" in script


def test_phase_training_script_has_help_preflight_and_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--preflight-only" in result.stdout

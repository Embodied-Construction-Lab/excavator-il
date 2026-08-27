from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_icra2027_two_act_models.sh"


def test_icra2027_training_script_runs_two_models_in_required_order_with_live_logs():
    script = SCRIPT.read_text(encoding="utf-8")

    dig_job = "icra2027_dig_only_front_swing_zero_seed"
    full_job = "icra2027_transport_dump_dual_rgb_seed"
    assert script.index(dig_job) < script.index(full_job)
    assert "set -Eeuo pipefail" in script
    # Two training calls plus one reusable evaluation pipeline, invoked twice below.
    assert script.count("2>&1 | tee") == 3
    assert "only after the dig-only model completed" in script
    assert script.index("完成：$PHASE\"") < script.index("开始留出集 checkpoint 评估")


def test_icra2027_training_script_uses_frozen_datasets_and_long_run_contract():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "icra2027_dig_only_front_split_swing_zero" in script
    assert "icra2027_transport_dump_dual_rgb_split" in script
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
    assert "--policy.dropout=0.1" in script
    assert "--policy.kl_weight=10.0" in script
    assert "--policy.optimizer_lr=1e-5" in script
    assert "--policy.optimizer_weight_decay=1e-4" in script
    assert "--policy.optimizer_lr_backbone=1e-5" in script
    assert "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1" in script
    assert "--dataset.image_transforms.enable=false" in script
    assert 'DIG_STEPS="${ICRA2027_DIG_TRAIN_STEPS:-400000}"' in script
    assert 'FULL_STEPS="${ICRA2027_FULL_TRAIN_STEPS:-300000}"' in script
    assert 'BATCH_SIZE="${ICRA2027_BATCH_SIZE:-2}"' in script
    assert 'SAVE_FREQ="${ICRA2027_SAVE_FREQ:-10000}"' in script
    assert '--steps="$DIG_STEPS"' in script
    assert '--steps="$FULL_STEPS"' in script
    assert "training output already exists" in script
    assert "another lerobot-train process is already running" in script


def test_icra2027_training_script_ranks_every_numeric_checkpoint_on_held_out_data():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "excavator-il evaluate-checkpoints" in script
    assert '"${checkpoints[@]}"' in script
    assert '--split-root="$split_root"' in script
    assert 'evaluate_checkpoints "$DIG_OUTPUT" "$DIG_SPLIT"' in script
    assert '"$FULL_OUTPUT" "$FULL_SPLIT" "$FULL_EVAL_LOG" "$FULL_EVAL_JSON"' in script
    assert script.index('lerobot-train \\\n  --dataset.repo_id="$FULL_TRAIN_REPO"') < script.index(
        "开始留出集 checkpoint 评估"
    )
    assert "只选择数值目录，避免重复评估 checkpoints/last 符号链接" in script
    assert "checkpoint_evaluation.json" in script
    assert 'print("选中的 checkpoint：", result["selected_checkpoint"])' in script


def test_icra2027_training_script_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_icra2027_training_script_exposes_a_non_mutating_preflight_mode():
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--preflight-only" in result.stdout

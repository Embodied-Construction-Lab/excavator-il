from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_icra2027_two_act_models.sh"


def test_icra2027_training_script_runs_two_models_in_required_order_with_live_logs():
    script = SCRIPT.read_text(encoding="utf-8")

    assert script.index('PHASE="仅挖掘 ACT 训练"') < script.index(
        'PHASE="挖掘—运转—倾倒 ACT 训练"'
    )
    assert "set -Eeuo pipefail" in script
    # Two fresh training calls, one resume call, and one reusable evaluator.
    assert script.count("2>&1 | tee") == 4
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
    assert (
        "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1"
        in script
    )
    assert "--dataset.image_transforms.enable=false" in script
    assert 'DIG_SAMPLE_BUDGET="${ICRA2027_DIG_SAMPLE_BUDGET:-800000}"' in script
    assert 'FULL_SAMPLE_BUDGET="${ICRA2027_FULL_SAMPLE_BUDGET:-600000}"' in script
    assert 'DIG_BATCH_SIZE="${ICRA2027_DIG_BATCH_SIZE:-2}"' in script
    assert 'FULL_BATCH_SIZE="${ICRA2027_FULL_BATCH_SIZE:-4}"' in script
    assert '--steps="$DIG_STEPS"' in script
    assert '--steps="$FULL_STEPS"' in script
    assert "training output or plan already exists" in script
    assert "another lerobot-train process is already running" in script


def test_icra2027_training_script_ranks_every_numeric_checkpoint_on_held_out_data():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "excavator-il evaluate-checkpoints" in script
    assert '"${checkpoints[@]}"' in script
    assert '--split-root="$split_root"' in script
    assert 'evaluate_checkpoints "$DIG_OUTPUT" "$DIG_SPLIT"' in script
    assert '"$FULL_OUTPUT" "$FULL_SPLIT" "$FULL_EVAL_LOG" "$FULL_EVAL_JSON"' in script
    full_training = '--dataset.repo_id="$FULL_TRAIN_REPO"'
    full_evaluation = 'PHASE="挖掘—运转—倾倒 ACT 留出集 checkpoint 评估"'
    assert script.index(full_training) < script.index(full_evaluation)
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


def test_icra2027_training_script_can_run_only_the_full_model():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "--model dig|full|both" in script
    assert 'MODEL_SELECTION="both"' in script
    assert 'model_selected "dig"' in script
    assert 'model_selected "full"' in script
    assert 'for output in "$DIG_OUTPUT" "$FULL_OUTPUT"' not in script


def test_icra2027_training_script_preserves_sample_exposure_per_model():
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'DIG_BATCH_SIZE="${ICRA2027_DIG_BATCH_SIZE:-2}"' in script
    assert 'FULL_BATCH_SIZE="${ICRA2027_FULL_BATCH_SIZE:-4}"' in script
    assert 'DIG_SAMPLE_BUDGET="${ICRA2027_DIG_SAMPLE_BUDGET:-800000}"' in script
    assert 'FULL_SAMPLE_BUDGET="${ICRA2027_FULL_SAMPLE_BUDGET:-600000}"' in script
    assert "steps_for_sample_budget" in script
    assert "save_freq_for_sample_interval" in script


def test_icra2027_training_script_requires_real_video_features_when_requested():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "--dataset-format image|video" in script
    assert 'DATASET_FORMAT="video"' in script
    assert 'expected_camera_dtype, dig_root, full_root = sys.argv[1:]' in script
    assert 'camera_dtypes != {expected_camera_dtype}' in script
    assert 'info.get("video_path")' in script
    assert 'cameras <= stats.keys()' in script
    assert "excavator_video_training_derivation.v1" in script
    assert "video_training_derivation_sha256" in script


def test_icra2027_training_script_writes_per_model_provenance_before_training():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "excavator_act_training_plan.v1" in script
    assert "write_training_plan" in script
    assert "split_provenance_sha256" in script
    assert "sample_budget" in script
    assert 'write_training_plan "dig"' in script
    assert 'write_training_plan "full"' in script


def test_icra2027_training_script_exposes_single_model_resume_contract():
    help_result = subprocess.run(
        [str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--resume" in help_result.stdout
    script = SCRIPT.read_text(encoding="utf-8")
    assert 'RESUME=false' in script
    assert '"$MODEL_SELECTION" == both' in script
    assert 'checkpoints/last/pretrained_model/train_config.json' in script
    assert '--config_path="$resume_config"' in script
    assert '--resume=true' in script
    assert 'tee -a "$training_log"' in script
    assert "resume training Git commit changed" in script
    assert "resume requires the original clean training plan" in script

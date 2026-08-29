#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/zhaoshuai/app/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-excavator-il}"

DIG_SAMPLE_BUDGET="${ICRA2027_DIG_SAMPLE_BUDGET:-800000}"
FULL_SAMPLE_BUDGET="${ICRA2027_FULL_SAMPLE_BUDGET:-600000}"
DIG_BATCH_SIZE="${ICRA2027_DIG_BATCH_SIZE:-2}"
FULL_BATCH_SIZE="${ICRA2027_FULL_BATCH_SIZE:-4}"
NUM_WORKERS="${ICRA2027_NUM_WORKERS:-2}"
SAVE_SAMPLE_INTERVAL="${ICRA2027_SAVE_SAMPLE_INTERVAL:-20000}"
SEED="${ICRA2027_SEED:-2027}"
MODEL_SELECTION="both"
DATASET_FORMAT="video"

DIG_IMAGE_SPLIT="data/lerobot/icra2027_dig_only_front_split_swing_zero"
FULL_IMAGE_SPLIT="data/lerobot/icra2027_transport_dump_dual_rgb_split"
DIG_VIDEO_SPLIT="data/lerobot/icra2027_dig_only_front_split_swing_zero_video"
FULL_VIDEO_SPLIT="data/lerobot/icra2027_transport_dump_dual_rgb_split_video"
DIG_STEPS=""
FULL_STEPS=""
DIG_SAVE_FREQ=""
FULL_SAVE_FREQ=""
DIG_SPLIT=""
FULL_SPLIT=""
DIG_JOB=""
FULL_JOB=""
DIG_OUTPUT=""
FULL_OUTPUT=""
DIG_LOG=""
FULL_LOG=""
DIG_EVAL_LOG=""
FULL_EVAL_LOG=""
DIG_EVAL_JSON=""
FULL_EVAL_JSON=""
BACKBONE_CACHE="/home/zhaoshuai/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
BACKBONE_SHA256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
PREFLIGHT_ONLY=false
RESUME=false

usage() {
  cat <<'EOF'
用法：bash scripts/train_icra2027_two_act_models.sh [选项]

默认依次训练：
  1. 仅挖掘 front-only swing-zero ACT：800000 个样本曝光
  2. 挖掘—运转—倾倒 front+dump ACT：600000 个样本曝光
  3. 分别在严格隔离的 validation Episode 上扫描全部 checkpoint 并选择最低安全 L1

--model dig|full|both           训练一个模型或依次训练两个模型（默认 both）
--dataset-format image|video   输入表示（默认 video；video 必须是真实 video feature）
--resume                       从所选模型 checkpoints/last 安全续训；必须只选一个模型
--preflight-only               只检查数据、CUDA、输出路径与权重，不启动训练

可选环境变量：
  ICRA2027_DIG_SAMPLE_BUDGET    仅挖掘样本曝光预算（默认 800000）
  ICRA2027_FULL_SAMPLE_BUDGET   完整动作样本曝光预算（默认 600000）
  ICRA2027_DIG_BATCH_SIZE       仅挖掘 batch（默认 2）
  ICRA2027_FULL_BATCH_SIZE      双相机 batch（默认 4）
  ICRA2027_NUM_WORKERS          DataLoader workers（默认 2）
  ICRA2027_SAVE_SAMPLE_INTERVAL 每多少个已见样本保存（默认 20000）
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model)
      [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
      MODEL_SELECTION="$2"
      shift 2
      ;;
    --model=*) MODEL_SELECTION="${1#*=}"; shift ;;
    --dataset-format)
      [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
      DATASET_FORMAT="$2"
      shift 2
      ;;
    --dataset-format=*) DATASET_FORMAT="${1#*=}"; shift ;;
    --resume) RESUME=true; shift ;;
    --preflight-only) PREFLIGHT_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ "$MODEL_SELECTION" != dig && "$MODEL_SELECTION" != full \
  && "$MODEL_SELECTION" != both ]]; then
  echo "error: --model must be dig, full, or both" >&2
  exit 2
fi
if [[ "$DATASET_FORMAT" != image && "$DATASET_FORMAT" != video ]]; then
  echo "error: --dataset-format must be image or video" >&2
  exit 2
fi
if [[ "$RESUME" == true && "$MODEL_SELECTION" == both ]]; then
  echo "error: --resume requires --model dig or --model full" >&2
  exit 2
fi

model_selected() {
  [[ "$MODEL_SELECTION" == "$1" || "$MODEL_SELECTION" == both ]]
}

steps_for_sample_budget() {
  local budget="$1"
  local batch_size="$2"
  echo $(((budget + batch_size - 1) / batch_size))
}

save_freq_for_sample_interval() {
  local interval="$1"
  local batch_size="$2"
  echo $(((interval + batch_size - 1) / batch_size))
}

PHASE="训练前检查"
on_error() {
  local status=$?
  echo "错误：${PHASE}失败（exit=${status}），后续模型不会启动。" >&2
  exit "$status"
}
trap on_error ERR

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${name} must be a positive integer: ${value}" >&2
    exit 2
  fi
}

cd "$REPO_ROOT"
mkdir -p logs

for pair in \
  "ICRA2027_DIG_SAMPLE_BUDGET:$DIG_SAMPLE_BUDGET" \
  "ICRA2027_FULL_SAMPLE_BUDGET:$FULL_SAMPLE_BUDGET" \
  "ICRA2027_DIG_BATCH_SIZE:$DIG_BATCH_SIZE" \
  "ICRA2027_FULL_BATCH_SIZE:$FULL_BATCH_SIZE" \
  "ICRA2027_NUM_WORKERS:$NUM_WORKERS" \
  "ICRA2027_SAVE_SAMPLE_INTERVAL:$SAVE_SAMPLE_INTERVAL" \
  "ICRA2027_SEED:$SEED"; do
  require_positive_integer "${pair%%:*}" "${pair#*:}"
done

DIG_STEPS="$(steps_for_sample_budget "$DIG_SAMPLE_BUDGET" "$DIG_BATCH_SIZE")"
FULL_STEPS="$(steps_for_sample_budget "$FULL_SAMPLE_BUDGET" "$FULL_BATCH_SIZE")"
DIG_SAVE_FREQ="$(save_freq_for_sample_interval \
  "$SAVE_SAMPLE_INTERVAL" "$DIG_BATCH_SIZE")"
FULL_SAVE_FREQ="$(save_freq_for_sample_interval \
  "$SAVE_SAMPLE_INTERVAL" "$FULL_BATCH_SIZE")"
if [[ "$DATASET_FORMAT" == video ]]; then
  DIG_SPLIT="$DIG_VIDEO_SPLIT"
  FULL_SPLIT="$FULL_VIDEO_SPLIT"
else
  DIG_SPLIT="$DIG_IMAGE_SPLIT"
  FULL_SPLIT="$FULL_IMAGE_SPLIT"
fi
DIG_JOB="icra2027_dig_only_front_swing_zero_${DATASET_FORMAT}_b${DIG_BATCH_SIZE}_seed${SEED}"
FULL_JOB="icra2027_transport_dump_dual_rgb_${DATASET_FORMAT}_b${FULL_BATCH_SIZE}_seed${SEED}"
DIG_OUTPUT="outputs/${DIG_JOB}"
FULL_OUTPUT="outputs/${FULL_JOB}"
DIG_LOG="logs/${DIG_JOB}.log"
FULL_LOG="logs/${FULL_JOB}.log"
DIG_EVAL_LOG="logs/${DIG_JOB}_checkpoint_evaluation.log"
FULL_EVAL_LOG="logs/${FULL_JOB}_checkpoint_evaluation.log"
DIG_EVAL_JSON="logs/${DIG_JOB}_checkpoint_evaluation.json"
FULL_EVAL_JSON="logs/${FULL_JOB}_checkpoint_evaluation.json"

if [[ ! -r "$CONDA_SH" ]]; then
  echo "error: conda activation script is unavailable: $CONDA_SH" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "error: lerobot-train is unavailable in conda env: $CONDA_ENV" >&2
  exit 2
fi
if pgrep -af '(^|/)lerobot-train( |$)' >/dev/null; then
  echo "error: another lerobot-train process is already running" >&2
  pgrep -af '(^|/)lerobot-train( |$)' >&2 || true
  exit 2
fi
validate_output_state() {
  local output_dir="$1"
  local plan_path="$2"
  if [[ "$RESUME" == true ]]; then
    local resume_config="$output_dir/checkpoints/last/pretrained_model/train_config.json"
    if [[ ! -r "$resume_config" || ! -r "$plan_path" ]]; then
      echo "error: resumable checkpoint or training plan is unavailable: $output_dir" >&2
      return 2
    fi
    python - "$plan_path" "$output_dir" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

plan_path, output_dir = map(Path, sys.argv[1:])
try:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit("resume training plan is unavailable or invalid") from exc
if plan.get("schema_version") != "excavator_act_training_plan.v1":
    raise SystemExit("resume training plan schema changed")
if plan.get("git_dirty") is not False or plan.get("git_status") != []:
    raise SystemExit("resume requires the original clean training plan")
if Path(str(plan.get("output_dir"))).resolve() != output_dir.resolve():
    raise SystemExit("resume training output does not match its plan")
current_head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
if current_head != plan.get("git_head"):
    raise SystemExit("resume training Git commit changed")
if subprocess.check_output(
    ["git", "status", "--porcelain"], text=True
).strip():
    raise SystemExit("resume training working tree is not clean")
PY
  elif [[ -e "$output_dir" || -e "$plan_path" ]]; then
    echo "error: training output or plan already exists; refusing to overwrite: $output_dir" >&2
    return 2
  fi
}
if model_selected "dig"; then
  validate_output_state "$DIG_OUTPUT" "logs/${DIG_JOB}_training_plan.json"
fi
if model_selected "full"; then
  validate_output_state "$FULL_OUTPUT" "logs/${FULL_JOB}_training_plan.json"
fi
if [[ ! -f "$BACKBONE_CACHE" ]]; then
  echo "error: commissioned ResNet18 weights are not cached: $BACKBONE_CACHE" >&2
  exit 2
fi
echo "$BACKBONE_SHA256  $BACKBONE_CACHE" | sha256sum --check --status

PREFLIGHT_OUTPUT="$(python - \
  "$MODEL_SELECTION" "$DATASET_FORMAT" "$DIG_SPLIT" "$FULL_SPLIT" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.dataset as ds

from excavator_il.action_dataset_transform import ACTION_TRANSFORM_SCHEMA_VERSION
from excavator_il.training_split import _dataset_fingerprint


def load_split(
    root_name: str,
    *,
    expected_train_sources: int,
    expected_validation_sources: int,
    expected_train_frames: int,
    expected_validation_frames: int,
    expected_cameras: set[str],
    expected_variant: str,
    require_zero_swing: bool,
    expected_camera_dtype: str,
) -> str:
    root = Path(root_name)
    if (root / "pipeline_validation.json").exists():
        raise SystemExit(f"formal dataset is marked pipeline-only: {root}")
    try:
        provenance = json.loads(
            (root / "split_provenance.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"split provenance is unavailable: {root}") from exc
    derivation = None
    if expected_camera_dtype == "video":
        try:
            derivation = json.loads(
                (root / "video_training_derivation.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"video derivation is unavailable: {root}") from exc
        if derivation.get("schema_version") != (
            "excavator_video_training_derivation.v1"
        ):
            raise SystemExit(f"video derivation schema is invalid: {root}")
    train_sources = set(provenance["train_source_episode_ids"])
    validation_sources = set(provenance["validation_source_episode_ids"])
    if train_sources & validation_sources:
        raise SystemExit(f"source Episode leakage detected: {root}")
    if (len(train_sources), len(validation_sources)) != (
        expected_train_sources,
        expected_validation_sources,
    ):
        raise SystemExit(f"unexpected source Episode counts: {root}")

    for partition, expected_frames, expected_sources in (
        ("train", expected_train_frames, train_sources),
        ("validation", expected_validation_frames, validation_sources),
    ):
        partition_root = root / partition
        info = json.loads(
            (partition_root / "meta" / "info.json").read_text(encoding="utf-8")
        )
        if info["total_frames"] != expected_frames:
            raise SystemExit(f"unexpected {partition} frame count: {root}")
        cameras = {
            name
            for name in info["features"]
            if name.startswith("observation.images.")
        }
        if cameras != expected_cameras:
            raise SystemExit(f"unexpected {partition} camera contract: {root}")
        camera_dtypes = {info["features"][name]["dtype"] for name in cameras}
        if camera_dtypes != {expected_camera_dtype}:
            raise SystemExit(f"unexpected {partition} camera representation: {root}")
        if expected_camera_dtype == "video" and not info.get("video_path"):
            raise SystemExit(f"video dataset has no video_path: {root}")
        stats = json.loads(
            (partition_root / "meta" / "stats.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(stats, dict) or not cameras <= stats.keys():
            raise SystemExit(f"camera stats are incomplete: {root}")
        expected_sha = provenance[f"{partition}_dataset_sha256"]
        if derivation is not None and (
            derivation.get("output_partition_sha256", {}).get(partition)
            != expected_sha
        ):
            raise SystemExit(f"video derivation fingerprint mismatch: {root}")
        if _dataset_fingerprint(partition_root) != expected_sha:
            raise SystemExit(f"{partition} dataset fingerprint mismatch: {root}")
        table = ds.dataset(partition_root / "data", format="parquet").to_table(
            columns=["action", "source.episode_id", "source.task_variant"]
        )
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 4 or not np.isfinite(actions).all():
            raise SystemExit(f"invalid {partition} four-axis actions: {root}")
        if set(table["source.episode_id"].to_pylist()) != expected_sources:
            raise SystemExit(f"unexpected {partition} source Episodes: {root}")
        if set(table["source.task_variant"].to_pylist()) != {expected_variant}:
            raise SystemExit(f"unexpected {partition} task variant: {root}")
        if require_zero_swing and np.count_nonzero(actions[:, 3]):
            raise SystemExit(f"nonzero swing label in {partition}: {root}")
    return provenance["train_repo_id"]


selection, expected_camera_dtype, dig_root, full_root = sys.argv[1:]
if selection in {"dig", "both"}:
    transform = json.loads(
        (Path(dig_root) / "action_transform_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if transform.get("schema_version") != ACTION_TRANSFORM_SCHEMA_VERSION:
        raise SystemExit("dig-only action-transform provenance is invalid")
    if transform.get("transform") != {
        "feature": "action",
        "field": "action_swing",
        "index": 3,
        "value": 0.0,
    }:
        raise SystemExit("dig-only split is not the commissioned swing-zero transform")
    print(
        "dig\t" + load_split(
        dig_root,
        expected_train_sources=83,
        expected_validation_sources=21,
        expected_train_frames=10436,
        expected_validation_frames=2571,
        expected_cameras={"observation.images.front"},
        expected_variant="dig_only",
        require_zero_swing=True,
        expected_camera_dtype=expected_camera_dtype,
    )
    )
if selection in {"full", "both"}:
    print(
        "full\t" + load_split(
        full_root,
        expected_train_sources=28,
        expected_validation_sources=7,
        expected_train_frames=5737,
        expected_validation_frames=1424,
        expected_cameras={
            "observation.images.front",
            "observation.images.dump",
        },
        expected_variant="dig_transport_dump",
        require_zero_swing=False,
        expected_camera_dtype=expected_camera_dtype,
    )
    )
PY
)"
DIG_TRAIN_REPO=""
FULL_TRAIN_REPO=""
while IFS=$'\t' read -r model repo_id; do
  case "$model" in
    dig) DIG_TRAIN_REPO="$repo_id" ;;
    full) FULL_TRAIN_REPO="$repo_id" ;;
    *) echo "error: dataset preflight returned an unknown model" >&2; exit 2 ;;
  esac
done <<<"$PREFLIGHT_OUTPUT"
if model_selected "dig" && [[ -z "$DIG_TRAIN_REPO" ]]; then
  echo "error: dig dataset preflight did not return a repo ID" >&2
  exit 2
fi
if model_selected "full" && [[ -z "$FULL_TRAIN_REPO" ]]; then
  echo "error: full dataset preflight did not return a repo ID" >&2
  exit 2
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to start formal ACT training")
print("CUDA gate passed:", torch.__version__, torch.cuda.get_device_name(0))
PY

echo "git_head=$(git rev-parse HEAD)"
git status --short || true
sha256sum "$BACKBONE_CACHE"
df -h .
echo "训练选择：$MODEL_SELECTION；数据表示：$DATASET_FORMAT"
python - "$DIG_STEPS" "$FULL_STEPS" "$DIG_BATCH_SIZE" "$FULL_BATCH_SIZE" <<'PY'
import sys

dig_steps, full_steps, dig_batch, full_batch = map(int, sys.argv[1:])
print(
    "等效 epoch 预算："
    f"仅挖掘={dig_steps * dig_batch / 10436:.2f}，"
    f"挖掘—运转—倾倒={full_steps * full_batch / 5737:.2f}"
)
PY
printf '%s\n' \
  "训练参数：dig_steps=$DIG_STEPS full_steps=$FULL_STEPS" \
  "dig_batch=$DIG_BATCH_SIZE full_batch=$FULL_BATCH_SIZE workers=$NUM_WORKERS" \
  "dig_save_freq=$DIG_SAVE_FREQ full_save_freq=$FULL_SAVE_FREQ seed=$SEED"
if [[ "$PREFLIGHT_ONLY" == true ]]; then
  echo "训练前检查通过；未启动训练。"
  exit 0
fi

write_training_plan() {
  local model="$1"
  local split_root="$2"
  local repo_id="$3"
  local job_name="$4"
  local output_dir="$5"
  local batch_size="$6"
  local steps="$7"
  local sample_budget="$8"
  local save_freq="$9"
  local plan_path="logs/${job_name}_training_plan.json"
  if [[ -e "$plan_path" ]]; then
    echo "error: training plan already exists; choose a new seed or remove only after audit: $plan_path" >&2
    return 2
  fi
  python - \
    "$model" "$split_root" "$repo_id" "$job_name" "$output_dir" \
    "$batch_size" "$steps" "$sample_budget" "$save_freq" \
    "$NUM_WORKERS" "$SEED" "$DATASET_FORMAT" "$plan_path" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

(
    model,
    split_root,
    repo_id,
    job_name,
    output_dir,
    batch_size,
    steps,
    sample_budget,
    save_freq,
    num_workers,
    seed,
    dataset_format,
    plan_path,
) = sys.argv[1:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


root = Path(split_root).resolve()
provenance_path = root / "split_provenance.json"
video_derivation_path = root / "video_training_derivation.json"
git_head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
git_status = subprocess.check_output(
    ["git", "status", "--porcelain"], text=True
).splitlines()
value = {
    "schema_version": "excavator_act_training_plan.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": model,
    "job_name": job_name,
    "output_dir": str(Path(output_dir).resolve()),
    "git_head": git_head,
    "git_dirty": bool(git_status),
    "git_status": git_status,
    "dataset_root": str(root),
    "dataset_repo_id": repo_id,
    "dataset_format": dataset_format,
    "split_provenance_sha256": sha256(provenance_path),
    "video_training_derivation_sha256": (
        sha256(video_derivation_path) if dataset_format == "video" else None
    ),
    "batch_size": int(batch_size),
    "num_workers": int(num_workers),
    "steps": int(steps),
    "sample_budget": int(sample_budget),
    "save_freq": int(save_freq),
    "seed": int(seed),
    "policy_contract": {
        "type": "act",
        "chunk_size": 20,
        "n_action_steps": 10,
        "vision_backbone": "resnet18",
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "latent_dim": 32,
        "n_vae_encoder_layers": 4,
        "optimizer_lr": 1e-5,
    },
}
path = Path(plan_path)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", dir=path.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, path)
except BaseException:
    Path(temporary_name).unlink(missing_ok=True)
    raise
print("训练计划：", path)
PY
}

resume_training() {
  local output_dir="$1"
  local training_log="$2"
  local resume_config="$output_dir/checkpoints/last/pretrained_model/train_config.json"
  echo "[$(date --iso-8601=seconds)] 从 checkpoint 续训：$resume_config" \
    | tee -a "$training_log"
  lerobot-train \
    --config_path="$resume_config" \
    --resume=true \
    2>&1 | tee -a "$training_log"
}

if model_selected "dig"; then
  if [[ "$RESUME" == false ]]; then
    write_training_plan "dig" "$DIG_SPLIT" "$DIG_TRAIN_REPO" \
      "$DIG_JOB" "$DIG_OUTPUT" "$DIG_BATCH_SIZE" "$DIG_STEPS" \
      "$DIG_SAMPLE_BUDGET" "$DIG_SAVE_FREQ"
  fi
  PHASE="仅挖掘 ACT 训练"
  echo "[$(date --iso-8601=seconds)] 开始：$PHASE"
  if [[ "$RESUME" == true ]]; then
    resume_training "$DIG_OUTPUT" "$DIG_LOG"
  else
    lerobot-train \
      --dataset.repo_id="$DIG_TRAIN_REPO" \
      --dataset.root="$DIG_SPLIT/train" \
      --dataset.video_backend=pyav \
      --policy.type=act \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.chunk_size=20 \
      --policy.n_action_steps=10 \
      --policy.vision_backbone=resnet18 \
      --policy.dim_model=512 \
      --policy.n_heads=8 \
      --policy.dim_feedforward=3200 \
      --policy.n_encoder_layers=4 \
      --policy.n_decoder_layers=1 \
      --policy.latent_dim=32 \
      --policy.n_vae_encoder_layers=4 \
      --policy.dropout=0.1 \
      --policy.kl_weight=10.0 \
      --policy.optimizer_lr=1e-5 \
      --policy.optimizer_weight_decay=1e-4 \
      --policy.optimizer_lr_backbone=1e-5 \
      --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
      --dataset.image_transforms.enable=false \
      --output_dir="$DIG_OUTPUT" \
      --job_name="$DIG_JOB" \
      --batch_size="$DIG_BATCH_SIZE" \
      --num_workers="$NUM_WORKERS" \
      --steps="$DIG_STEPS" \
      --log_freq=100 \
      --save_checkpoint=true \
      --save_freq="$DIG_SAVE_FREQ" \
      --eval_freq=0 \
      --wandb.enable=false \
      --seed="$SEED" \
      2>&1 | tee "$DIG_LOG"
  fi
  echo "[$(date --iso-8601=seconds)] 完成：$PHASE"
fi

# Start the second model only after the dig-only model completed successfully.
if model_selected "full"; then
  if [[ "$RESUME" == false ]]; then
    write_training_plan "full" "$FULL_SPLIT" "$FULL_TRAIN_REPO" \
      "$FULL_JOB" "$FULL_OUTPUT" "$FULL_BATCH_SIZE" "$FULL_STEPS" \
      "$FULL_SAMPLE_BUDGET" "$FULL_SAVE_FREQ"
  fi
  PHASE="挖掘—运转—倾倒 ACT 训练"
  echo "[$(date --iso-8601=seconds)] 开始：$PHASE"
  if [[ "$RESUME" == true ]]; then
    resume_training "$FULL_OUTPUT" "$FULL_LOG"
  else
    lerobot-train \
      --dataset.repo_id="$FULL_TRAIN_REPO" \
      --dataset.root="$FULL_SPLIT/train" \
      --dataset.video_backend=pyav \
      --policy.type=act \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.chunk_size=20 \
      --policy.n_action_steps=10 \
      --policy.vision_backbone=resnet18 \
      --policy.dim_model=512 \
      --policy.n_heads=8 \
      --policy.dim_feedforward=3200 \
      --policy.n_encoder_layers=4 \
      --policy.n_decoder_layers=1 \
      --policy.latent_dim=32 \
      --policy.n_vae_encoder_layers=4 \
      --policy.dropout=0.1 \
      --policy.kl_weight=10.0 \
      --policy.optimizer_lr=1e-5 \
      --policy.optimizer_weight_decay=1e-4 \
      --policy.optimizer_lr_backbone=1e-5 \
      --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
      --dataset.image_transforms.enable=false \
      --output_dir="$FULL_OUTPUT" \
      --job_name="$FULL_JOB" \
      --batch_size="$FULL_BATCH_SIZE" \
      --num_workers="$NUM_WORKERS" \
      --steps="$FULL_STEPS" \
      --log_freq=100 \
      --save_checkpoint=true \
      --save_freq="$FULL_SAVE_FREQ" \
      --eval_freq=0 \
      --wandb.enable=false \
      --seed="$SEED" \
      2>&1 | tee "$FULL_LOG"
  fi
  echo "[$(date --iso-8601=seconds)] 完成：$PHASE"
fi

evaluate_checkpoints() {
  local output_root="$1"
  local split_root="$2"
  local evaluation_log="$3"
  local evaluation_json="$4"
  local checkpoint_dir
  local -a checkpoints=()

  # 只选择数值目录，避免重复评估 checkpoints/last 符号链接。
  for checkpoint_dir in "$output_root"/checkpoints/[0-9]*; do
    if [[ -d "$checkpoint_dir/pretrained_model" \
      && "$(basename "$checkpoint_dir")" =~ ^[0-9]+$ ]]; then
      checkpoints+=("$checkpoint_dir/pretrained_model")
    fi
  done
  if [[ "${#checkpoints[@]}" -lt 2 ]]; then
    echo "error: fewer than two numeric checkpoints are available: $output_root" >&2
    return 2
  fi
  echo "评估 checkpoint 数量：${#checkpoints[@]}；结果日志：$evaluation_log"
  excavator-il evaluate-checkpoints \
    "${checkpoints[@]}" \
    --split-root="$split_root" \
    --device=cuda \
    --batch-size=4 \
    --num-workers=2 \
    2>&1 | tee "$evaluation_log"
  python - "$evaluation_log" "$evaluation_json" <<'PY'
import json
from pathlib import Path
import sys

log_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
text = log_path.read_text(encoding="utf-8")
decoder = json.JSONDecoder()
result = None
for offset, character in enumerate(text):
    if character != "{":
        continue
    try:
        candidate, end = decoder.raw_decode(text, offset)
    except json.JSONDecodeError:
        continue
    if not text[end:].strip():
        result = candidate
        break
if not isinstance(result, dict) or not isinstance(
    result.get("selected_checkpoint"), str
):
    raise SystemExit("checkpoint evaluation did not produce a selected checkpoint")
output_path.write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("选中的 checkpoint：", result["selected_checkpoint"])
print("纯 JSON 评估结果：", output_path)
PY
}

if model_selected "dig"; then
  PHASE="仅挖掘 ACT 留出集 checkpoint 评估"
  echo "[$(date --iso-8601=seconds)] 开始留出集 checkpoint 评估：$PHASE"
  evaluate_checkpoints "$DIG_OUTPUT" "$DIG_SPLIT" "$DIG_EVAL_LOG" "$DIG_EVAL_JSON"
  echo "[$(date --iso-8601=seconds)] 完成：$PHASE"
fi

if model_selected "full"; then
  PHASE="挖掘—运转—倾倒 ACT 留出集 checkpoint 评估"
  echo "[$(date --iso-8601=seconds)] 开始留出集 checkpoint 评估：$PHASE"
  evaluate_checkpoints \
    "$FULL_OUTPUT" "$FULL_SPLIT" "$FULL_EVAL_LOG" "$FULL_EVAL_JSON"
  echo "[$(date --iso-8601=seconds)] 完成：$PHASE"
fi

echo "所选 ACT 模型已完成长训练与留出集 checkpoint 选择：$MODEL_SELECTION"

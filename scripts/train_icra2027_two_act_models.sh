#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/zhaoshuai/app/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-excavator-il}"

DIG_STEPS="${ICRA2027_DIG_TRAIN_STEPS:-400000}"
FULL_STEPS="${ICRA2027_FULL_TRAIN_STEPS:-300000}"
BATCH_SIZE="${ICRA2027_BATCH_SIZE:-2}"
NUM_WORKERS="${ICRA2027_NUM_WORKERS:-2}"
SAVE_FREQ="${ICRA2027_SAVE_FREQ:-10000}"
SEED="${ICRA2027_SEED:-2027}"

DIG_SPLIT="data/lerobot/icra2027_dig_only_front_split_swing_zero"
FULL_SPLIT="data/lerobot/icra2027_transport_dump_dual_rgb_split"
DIG_JOB="icra2027_dig_only_front_swing_zero_seed${SEED}"
FULL_JOB="icra2027_transport_dump_dual_rgb_seed${SEED}"
DIG_OUTPUT="outputs/${DIG_JOB}"
FULL_OUTPUT="outputs/${FULL_JOB}"
DIG_LOG="logs/${DIG_JOB}.log"
FULL_LOG="logs/${FULL_JOB}.log"
DIG_EVAL_LOG="logs/${DIG_JOB}_checkpoint_evaluation.log"
FULL_EVAL_LOG="logs/${FULL_JOB}_checkpoint_evaluation.log"
DIG_EVAL_JSON="logs/${DIG_JOB}_checkpoint_evaluation.json"
FULL_EVAL_JSON="logs/${FULL_JOB}_checkpoint_evaluation.json"
BACKBONE_CACHE="/home/zhaoshuai/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
BACKBONE_SHA256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
PREFLIGHT_ONLY=false

usage() {
  cat <<'EOF'
用法：bash scripts/train_icra2027_two_act_models.sh [--preflight-only]

默认依次训练：
  1. 仅挖掘 front-only swing-zero ACT：400000 steps
  2. 挖掘—运转—倾倒 front+dump ACT：300000 steps
  3. 分别在严格隔离的 validation Episode 上扫描全部 checkpoint 并选择最低安全 L1

--preflight-only  只检查数据、CUDA、输出路径与预训练权重，不启动训练。

可选环境变量：
  ICRA2027_DIG_TRAIN_STEPS   仅挖掘训练步数（默认 400000）
  ICRA2027_FULL_TRAIN_STEPS  挖掘—运转—倾倒训练步数（默认 300000）
  ICRA2027_BATCH_SIZE        batch size（默认 2，已通过双相机显存预检）
  ICRA2027_SAVE_FREQ         checkpoint 间隔（默认 10000）
EOF
}

case "${1:-}" in
  "") ;;
  --preflight-only) PREFLIGHT_ONLY=true ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi

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
  "ICRA2027_DIG_TRAIN_STEPS:$DIG_STEPS" \
  "ICRA2027_FULL_TRAIN_STEPS:$FULL_STEPS" \
  "ICRA2027_BATCH_SIZE:$BATCH_SIZE" \
  "ICRA2027_NUM_WORKERS:$NUM_WORKERS" \
  "ICRA2027_SAVE_FREQ:$SAVE_FREQ" \
  "ICRA2027_SEED:$SEED"; do
  require_positive_integer "${pair%%:*}" "${pair#*:}"
done

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
for output in "$DIG_OUTPUT" "$FULL_OUTPUT"; do
  if [[ -e "$output" ]]; then
    echo "error: training output already exists; refusing to overwrite: $output" >&2
    exit 2
  fi
done
if [[ ! -f "$BACKBONE_CACHE" ]]; then
  echo "error: commissioned ResNet18 weights are not cached: $BACKBONE_CACHE" >&2
  exit 2
fi
echo "$BACKBONE_SHA256  $BACKBONE_CACHE" | sha256sum --check --status

PREFLIGHT_OUTPUT="$(python - <<'PY'
import json
from pathlib import Path

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
        expected_sha = provenance[f"{partition}_dataset_sha256"]
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


dig_root = "data/lerobot/icra2027_dig_only_front_split_swing_zero"
transform = json.loads(
    (Path(dig_root) / "action_transform_provenance.json").read_text(encoding="utf-8")
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
    load_split(
        dig_root,
        expected_train_sources=83,
        expected_validation_sources=21,
        expected_train_frames=10436,
        expected_validation_frames=2571,
        expected_cameras={"observation.images.front"},
        expected_variant="dig_only",
        require_zero_swing=True,
    )
)
print(
    load_split(
        "data/lerobot/icra2027_transport_dump_dual_rgb_split",
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
    )
)
PY
)"
mapfile -t TRAIN_REPOS <<<"$PREFLIGHT_OUTPUT"
if [[ "${#TRAIN_REPOS[@]}" -ne 2 ]]; then
  echo "error: dataset preflight returned the wrong number of repo IDs" >&2
  exit 2
fi
DIG_TRAIN_REPO="${TRAIN_REPOS[0]}"
FULL_TRAIN_REPO="${TRAIN_REPOS[1]}"

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
echo "训练顺序：1) 仅挖掘 front-only swing-zero；2) 挖掘—运转—倾倒 front+dump"
python - "$DIG_STEPS" "$FULL_STEPS" "$BATCH_SIZE" <<'PY'
import sys

dig_steps, full_steps, batch_size = map(int, sys.argv[1:])
print(
    "等效 epoch 预算："
    f"仅挖掘={dig_steps * batch_size / 10436:.2f}，"
    f"挖掘—运转—倾倒={full_steps * batch_size / 5737:.2f}"
)
PY
printf '%s\n' \
  "训练参数：dig_steps=$DIG_STEPS full_steps=$FULL_STEPS" \
  "batch_size=$BATCH_SIZE workers=$NUM_WORKERS save_freq=$SAVE_FREQ seed=$SEED"
if [[ "$PREFLIGHT_ONLY" == true ]]; then
  echo "训练前检查通过；未启动训练。"
  exit 0
fi

PHASE="仅挖掘 ACT 训练"
echo "[$(date --iso-8601=seconds)] 开始：$PHASE"
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
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$DIG_STEPS" \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed="$SEED" \
  2>&1 | tee "$DIG_LOG"
echo "[$(date --iso-8601=seconds)] 完成：$PHASE"

# Start the second model only after the dig-only model completed successfully.
PHASE="挖掘—运转—倾倒 ACT 训练"
echo "[$(date --iso-8601=seconds)] 开始：$PHASE"
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
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$FULL_STEPS" \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed="$SEED" \
  2>&1 | tee "$FULL_LOG"
echo "[$(date --iso-8601=seconds)] 完成：$PHASE"

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

PHASE="仅挖掘 ACT 留出集 checkpoint 评估"
echo "[$(date --iso-8601=seconds)] 开始留出集 checkpoint 评估：$PHASE"
evaluate_checkpoints "$DIG_OUTPUT" "$DIG_SPLIT" "$DIG_EVAL_LOG" "$DIG_EVAL_JSON"
echo "[$(date --iso-8601=seconds)] 完成：$PHASE"

PHASE="挖掘—运转—倾倒 ACT 留出集 checkpoint 评估"
echo "[$(date --iso-8601=seconds)] 开始留出集 checkpoint 评估：$PHASE"
evaluate_checkpoints \
  "$FULL_OUTPUT" "$FULL_SPLIT" "$FULL_EVAL_LOG" "$FULL_EVAL_JSON"
echo "[$(date --iso-8601=seconds)] 完成：$PHASE"

echo "两套 ACT 模型均已完成长训练与留出集 checkpoint 选择。"

#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/zhaoshuai/app/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-excavator-il}"
DATASET_ROOT="data/lerobot/icra2027_transport_dump_dual_rgb_three_phase_video"
SAMPLE_BUDGET="${ACT_PHASE_SAMPLE_BUDGET:-600000}"
BATCH_SIZE="${ACT_PHASE_BATCH_SIZE:-4}"
NUM_WORKERS="${ACT_PHASE_NUM_WORKERS:-2}"
SAVE_SAMPLE_INTERVAL="${ACT_PHASE_SAVE_SAMPLE_INTERVAL:-40000}"
SEED="${ACT_PHASE_SEED:-2027}"
JOB_NAME="icra2027_transport_dump_three_phase_video_b${BATCH_SIZE}_seed${SEED}"
OUTPUT_ROOT="outputs/${JOB_NAME}"
TRAINING_LOG="logs/${JOB_NAME}.log"
PLAN_PATH="logs/${JOB_NAME}_training_plan.json"
EVALUATION_LOG="logs/${JOB_NAME}_checkpoint_evaluation.log"
EVALUATION_JSON="logs/${JOB_NAME}_checkpoint_evaluation.json"
BACKBONE_CACHE="/home/zhaoshuai/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
BACKBONE_SHA256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
PREFLIGHT_ONLY=false

usage() {
  cat <<'EOF'
用法：bash scripts/train_phase_conditioned_transport_dump_act.sh [--preflight-only]

训练一个连续的三阶段 ACT：
  双路 RGB + 原 11D state + DIG/TRANSPORT/DUMP 3D one-hot → 4D 杆量

--preflight-only  只核验数据、GPU、输出隔离和 provenance，不启动训练

可选环境变量：
  ACT_PHASE_SAMPLE_BUDGET       样本曝光预算（默认 600000）
  ACT_PHASE_BATCH_SIZE          batch size（默认 4）
  ACT_PHASE_NUM_WORKERS         DataLoader workers（默认 2）
  ACT_PHASE_SAVE_SAMPLE_INTERVAL checkpoint 样本间隔（默认 40000）
  ACT_PHASE_SEED                随机种子（默认 2027）
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --preflight-only) PREFLIGHT_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${name} must be a positive integer: ${value}" >&2
    exit 2
  fi
}

for pair in \
  "ACT_PHASE_SAMPLE_BUDGET:$SAMPLE_BUDGET" \
  "ACT_PHASE_BATCH_SIZE:$BATCH_SIZE" \
  "ACT_PHASE_NUM_WORKERS:$NUM_WORKERS" \
  "ACT_PHASE_SAVE_SAMPLE_INTERVAL:$SAVE_SAMPLE_INTERVAL" \
  "ACT_PHASE_SEED:$SEED"; do
  require_positive_integer "${pair%%:*}" "${pair#*:}"
done

STEPS=$(((SAMPLE_BUDGET + BATCH_SIZE - 1) / BATCH_SIZE))
SAVE_FREQ=$(((SAVE_SAMPLE_INTERVAL + BATCH_SIZE - 1) / BATCH_SIZE))
cd "$REPO_ROOT"

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
if [[ -e "$OUTPUT_ROOT" || -e "$PLAN_PATH" ]]; then
  echo "error: refusing to overwrite isolated training output: $OUTPUT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$BACKBONE_CACHE" ]]; then
  echo "error: commissioned ResNet18 weights are unavailable: $BACKBONE_CACHE" >&2
  exit 2
fi
echo "$BACKBONE_SHA256  $BACKBONE_CACHE" | sha256sum --check --status

PREFLIGHT_OUTPUT="$(python - "$DATASET_ROOT" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.dataset as ds

from excavator_il.phase_conditioned_dataset import (
    PHASE_CONDITIONED_DATASET_SCHEMA_VERSION,
    PHASE_FEATURE_NAMES,
)
from excavator_il.training_split import _dataset_fingerprint


root = Path(sys.argv[1]).resolve()
try:
    split = json.loads((root / "split_provenance.json").read_text())
    phase_provenance = json.loads(
        (root / "phase_conditioning_provenance.json").read_text()
    )
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit("phase-conditioned dataset provenance is unavailable") from exc
if phase_provenance.get("schema_version") != PHASE_CONDITIONED_DATASET_SCHEMA_VERSION:
    raise SystemExit("phase-conditioned dataset provenance schema is invalid")
if phase_provenance.get("phase_feature_names") != list(PHASE_FEATURE_NAMES):
    raise SystemExit("phase-conditioned feature order is invalid")
train_sources = set(split.get("train_source_episode_ids", []))
validation_sources = set(split.get("validation_source_episode_ids", []))
if not train_sources or not validation_sources or train_sources & validation_sources:
    raise SystemExit("source Episode leakage or empty parent split detected")
if (len(train_sources), len(validation_sources)) != (28, 7):
    raise SystemExit("unexpected parent Episode counts")
excluded = phase_provenance.get("excluded_sequences")
if excluded != [{
    "partition": "validation",
    "data_file_index": 33,
    "source_episode_id": "episode_0143",
    "frame_count": 3,
    "reason": "exclude_short_fragment",
}]:
    raise SystemExit("the known episode_0143 short fragment was not isolated")

expected_frames = {"train": 5737, "validation": 1421}
for partition in ("train", "validation"):
    partition_root = root / partition
    info = json.loads((partition_root / "meta/info.json").read_text())
    state_feature = info.get("features", {}).get("observation.state", {})
    if state_feature.get("shape") != [14]:
        raise SystemExit(f"{partition} state contract is not 14D")
    if state_feature.get("names", [])[-3:] != list(PHASE_FEATURE_NAMES):
        raise SystemExit(f"{partition} phase feature names are invalid")
    if info.get("total_frames") != expected_frames[partition]:
        raise SystemExit(f"{partition} frame count is invalid")
    cameras = {
        name: feature.get("dtype")
        for name, feature in info.get("features", {}).items()
        if name.startswith("observation.images.")
    }
    if cameras != {
        "observation.images.front": "video",
        "observation.images.dump": "video",
    }:
        raise SystemExit(f"{partition} dual-video contract is invalid")
    expected_sha = split.get(f"{partition}_dataset_sha256")
    if expected_sha != phase_provenance.get("output_partition_sha256", {}).get(
        partition
    ) or _dataset_fingerprint(partition_root) != expected_sha:
        raise SystemExit(f"{partition} dataset fingerprint mismatch")
    table = ds.dataset(partition_root / "data", format="parquet").to_table(
        columns=["observation.state", "action", "source.episode_id"]
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if states.shape != (expected_frames[partition], 14):
        raise SystemExit(f"{partition} state matrix is invalid")
    if actions.shape != (expected_frames[partition], 4):
        raise SystemExit(f"{partition} action matrix is invalid")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise SystemExit(f"{partition} contains non-finite values")
    phase = states[:, -3:]
    if not np.all((phase == 0.0) | (phase == 1.0)) or not np.allclose(
        phase.sum(axis=1), 1.0
    ):
        raise SystemExit(f"{partition} phase one-hot is invalid")
    if np.any(phase.sum(axis=0) == 0):
        raise SystemExit(f"{partition} does not contain all three phases")
    expected_sources = train_sources if partition == "train" else validation_sources
    if set(table["source.episode_id"].to_pylist()) != expected_sources:
        raise SystemExit(f"{partition} source Episode membership changed")
print(split["train_repo_id"])
PY
)"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to start ACT training")
print("CUDA gate passed:", torch.__version__, torch.cuda.get_device_name(0))
PY

echo "数据集检查通过：$PREFLIGHT_OUTPUT"
echo "训练：batch=$BATCH_SIZE steps=$STEPS samples=$((STEPS * BATCH_SIZE)) workers=$NUM_WORKERS"
echo "等效 epoch：$(python -c "print(round($STEPS*$BATCH_SIZE/5737, 2))")"
if [[ "$PREFLIGHT_ONLY" == true ]]; then
  echo "三阶段 ACT 训练前检查通过；未启动训练。"
  exit 0
fi

mkdir -p logs
python - \
  "$DATASET_ROOT" "$PREFLIGHT_OUTPUT" "$OUTPUT_ROOT" "$PLAN_PATH" \
  "$BATCH_SIZE" "$NUM_WORKERS" "$STEPS" "$SAMPLE_BUDGET" "$SAVE_FREQ" \
  "$SEED" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

(
    dataset_root,
    repo_id,
    output_root,
    plan_path,
    batch_size,
    num_workers,
    steps,
    sample_budget,
    save_freq,
    seed,
) = sys.argv[1:]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(dataset_root).resolve()
git_status = subprocess.check_output(
    ["git", "status", "--porcelain"], text=True
).splitlines()
value = {
    "schema_version": "excavator_phase_conditioned_act_training_plan.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": "dig_transport_dump_three_phase",
    "phase_order": ["DIG", "TRANSPORT", "DUMP"],
    "state_dimension": 14,
    "action_dimension": 4,
    "dataset_root": str(root),
    "dataset_repo_id": repo_id,
    "split_provenance_sha256": sha256(root / "split_provenance.json"),
    "phase_provenance_sha256": sha256(
        root / "phase_conditioning_provenance.json"
    ),
    "phase_annotations_sha256": sha256(
        Path("config/training/icra2027_transport_dump_three_phase_annotations.v1.json")
    ),
    "dataset_deriver_sha256": sha256(
        Path("src/excavator_il/phase_conditioned_dataset.py")
    ),
    "training_script_sha256": sha256(
        Path("scripts/train_phase_conditioned_transport_dump_act.sh")
    ),
    "output_dir": str(Path(output_root).resolve()),
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_status": git_status,
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
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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
PY

lerobot-train \
  --dataset.repo_id="$PREFLIGHT_OUTPUT" \
  --dataset.root="$DATASET_ROOT/train" \
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
  --output_dir="$OUTPUT_ROOT" \
  --job_name="$JOB_NAME" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed="$SEED" \
  2>&1 | tee "$TRAINING_LOG"

mapfile -t CHECKPOINTS < <(
  find "$OUTPUT_ROOT/checkpoints" -mindepth 2 -maxdepth 2 \
    -type d -name pretrained_model -printf '%h\n' \
    | awk -F/ '$NF ~ /^[0-9]+$/ {print $0 "/pretrained_model"}' \
    | sort -V
)
if [[ "${#CHECKPOINTS[@]}" -lt 2 ]]; then
  echo "error: fewer than two numeric checkpoints are available" >&2
  exit 2
fi
excavator-il evaluate-checkpoints \
  "${CHECKPOINTS[@]}" \
  --split-root="$DATASET_ROOT" \
  --device=cuda \
  --batch-size=4 \
  --num-workers=2 \
  2>&1 | tee "$EVALUATION_LOG"

python - "$EVALUATION_LOG" "$EVALUATION_JSON" <<'PY'
import json
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
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
if not isinstance(result, dict) or not isinstance(result.get("selected_checkpoint"), str):
    raise SystemExit("checkpoint evaluation did not produce a selected checkpoint")
Path(sys.argv[2]).write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("选中的 checkpoint：", result["selected_checkpoint"])
PY

echo "三阶段 ACT 训练与留出集 checkpoint 选择完成。"

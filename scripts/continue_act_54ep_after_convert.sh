#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/zhaoshuai/workspace_uinty/RL_prj/excavator-il"
CONDA_SH="/home/zhaoshuai/app/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="excavator-il"

DATASET="data/lerobot/excavator_dig_20260819_54ep_v1"
DATASET_REPO="local/excavator_dig_20260819_54ep_v1"
MANIFEST="data/lerobot/split_manifests/excavator_dig_20260819_54ep_v1.json"
SPLIT_ROOT="data/lerobot/excavator_dig_20260819_54ep_v1_split"
RUN="outputs/act_excavator_dig_54ep_seed1000_100k"
TRAIN_LOG="logs/act_excavator_dig_54ep_seed1000_100k.log"
PIPELINE_LOG="logs/act_excavator_dig_54ep_seed1000_100k_pipeline.log"

WAIT_PID="${1:-}"
if [[ ! "$WAIT_PID" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 CONVERT_PID" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$MANIFEST")"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "[$(date --iso-8601=seconds)] continuation pipeline started; convert_pid=$WAIT_PID"

if kill -0 "$WAIT_PID" 2>/dev/null; then
  CONVERT_COMMAND="$(tr '\0' ' ' < "/proc/$WAIT_PID/cmdline")"
  if [[ "$CONVERT_COMMAND" != *"excavator-il convert"* ]] \
    || [[ "$CONVERT_COMMAND" != *"--output-root $DATASET"* ]]; then
    echo "error: PID $WAIT_PID is not the expected dataset conversion" >&2
    echo "command: $CONVERT_COMMAND" >&2
    exit 2
  fi
  echo "Waiting for conversion PID $WAIT_PID to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 10
  done
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "[$(date --iso-8601=seconds)] conversion process ended; verifying published dataset"
python - <<'PY'
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("data/lerobot/excavator_dig_20260819_54ep_v1")
repo_id = "local/excavator_dig_20260819_54ep_v1"
if (root / "pipeline_validation.json").exists():
    raise SystemExit("pipeline-validation dataset is not eligible for formal training")

dataset = LeRobotDataset(repo_id=repo_id, root=root)
assert dataset.num_episodes == 54, dataset.num_episodes
assert dataset.num_frames == 6904, dataset.num_frames
assert dataset.features["observation.state"]["shape"] == (11,)
assert dataset.features["action"]["shape"] == (4,)
assert dataset.features["action"]["names"] == [
    "action_boom",
    "action_stick",
    "action_bucket",
    "action_swing",
]
print(f"conversion gate passed: episodes={dataset.num_episodes} frames={dataset.num_frames}")
PY

echo "[$(date --iso-8601=seconds)] preparing deterministic parent-Episode split"
excavator-il prepare-training-split \
  --dataset-root "$DATASET" \
  --repo-id "$DATASET_REPO" \
  --output "$MANIFEST" \
  --train-ratio 0.8 \
  --seed 1000

if [[ ! -e "$SPLIT_ROOT" ]]; then
  echo "[$(date --iso-8601=seconds)] materializing train/validation datasets"
  excavator-il materialize-training-split \
    --manifest "$MANIFEST" \
    --output-root "$SPLIT_ROOT"
else
  echo "Existing split root found; verifying it instead of overwriting: $SPLIT_ROOT"
fi

TRAIN_REPO="$(python - <<'PY'
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("data/lerobot/excavator_dig_20260819_54ep_v1_split")
provenance = json.loads((root / "split_provenance.json").read_text(encoding="utf-8"))
train = LeRobotDataset(repo_id=provenance["train_repo_id"], root=root / "train")
validation = LeRobotDataset(
    repo_id=provenance["validation_repo_id"], root=root / "validation"
)
assert train.num_episodes == 43, train.num_episodes
assert train.num_frames == 5440, train.num_frames
assert validation.num_episodes == 11, validation.num_episodes
assert validation.num_frames == 1464, validation.num_frames
print(provenance["train_repo_id"])
PY
)"

if [[ -e "$RUN" ]]; then
  echo "error: training output already exists; refusing to overwrite: $RUN" >&2
  exit 2
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to start formal ACT training")
print("CUDA gate passed:", torch.__version__, torch.cuda.get_device_name(0))
PY

echo "git_head=$(git rev-parse HEAD)"
git status --short || true
sha256sum /home/zhaoshuai/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

echo "[$(date --iso-8601=seconds)] starting formal ACT training: steps=100000 batch=2"
set -o pipefail
lerobot-train \
  --dataset.repo_id="$TRAIN_REPO" \
  --dataset.root="$SPLIT_ROOT/train" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --output_dir="$RUN" \
  --job_name=act_excavator_dig_54ep_seed1000_100k \
  --batch_size=2 \
  --num_workers=2 \
  --steps=100000 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed=1000 \
  2>&1 | tee "$TRAIN_LOG"

echo "[$(date --iso-8601=seconds)] formal ACT training completed"

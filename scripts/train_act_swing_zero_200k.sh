#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/home/zhaoshuai/app/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="excavator-il"

SOURCE_SPLIT="data/lerobot/excavator_dig_20260819_54ep_v1_split"
DERIVED_SPLIT="data/lerobot/excavator_dig_20260819_54ep_v1_split_swing_zero"
RUN="outputs/act_excavator_dig_54ep_swing_zero_seed2026_200k"
TRAIN_LOG="logs/act_excavator_dig_54ep_swing_zero_seed2026_200k.log"

cd "$REPO_ROOT"
mkdir -p logs
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if pgrep -af '(^|/)lerobot-train( |$)' >/dev/null; then
  echo "error: another lerobot-train process is already running" >&2
  pgrep -af '(^|/)lerobot-train( |$)' >&2 || true
  exit 2
fi
if [[ -e "$RUN" ]]; then
  echo "error: training output already exists; refusing to overwrite: $RUN" >&2
  exit 2
fi
if [[ -e "$SOURCE_SPLIT/pipeline_validation.json" ]]; then
  echo "error: pipeline-validation source is not eligible for formal training" >&2
  exit 2
fi

if [[ ! -e "$DERIVED_SPLIT" ]]; then
  echo "[$(date --iso-8601=seconds)] deriving an isolated swing-zero split"
  excavator-il derive-zero-swing-split \
    --source-root "$SOURCE_SPLIT" \
    --output-root "$DERIVED_SPLIT" \
    --repo-suffix swing_zero
else
  echo "[$(date --iso-8601=seconds)] derived split exists; verifying instead of overwriting"
fi

TRAIN_REPO="$(python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from excavator_il.action_dataset_transform import ACTION_TRANSFORM_SCHEMA_VERSION
from excavator_il.training_split import _dataset_fingerprint

root = Path("data/lerobot/excavator_dig_20260819_54ep_v1_split_swing_zero")
if (root / "pipeline_validation.json").exists():
    raise SystemExit("derived dataset is marked pipeline-only")
provenance = json.loads((root / "split_provenance.json").read_text(encoding="utf-8"))
transform = json.loads(
    (root / "action_transform_provenance.json").read_text(encoding="utf-8")
)
if transform.get("schema_version") != ACTION_TRANSFORM_SCHEMA_VERSION:
    raise SystemExit("derived action-transform provenance is invalid")
if transform.get("transform") != {
    "feature": "action",
    "field": "action_swing",
    "index": 3,
    "value": 0.0,
}:
    raise SystemExit("derived action transform is not the required swing-zero transform")
for partition in ("train", "validation"):
    expected = provenance[f"{partition}_dataset_sha256"]
    actual = _dataset_fingerprint(root / partition)
    if actual != expected:
        raise SystemExit(f"{partition} fingerprint mismatch")
    if (root / partition / "pipeline_validation.json").exists():
        raise SystemExit(f"{partition} is marked pipeline-only")
    for path in sorted((root / partition / "data").rglob("*.parquet")):
        actions = np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist())
        if actions.ndim != 2 or actions.shape[1] != 4:
            raise SystemExit(f"invalid action shape in {path}")
        if not np.isfinite(actions).all() or not np.equal(actions[:, 3], 0.0).all():
            raise SystemExit(f"nonzero or invalid swing label in {path}")

train = LeRobotDataset(repo_id=provenance["train_repo_id"], root=root / "train")
validation = LeRobotDataset(
    repo_id=provenance["validation_repo_id"], root=root / "validation"
)
if (train.num_episodes, train.num_frames) != (43, 5440):
    raise SystemExit("unexpected train dataset size")
if (validation.num_episodes, validation.num_frames) != (11, 1464):
    raise SystemExit("unexpected validation dataset size")
if train.features["observation.state"]["shape"] != (11,):
    raise SystemExit("state contract is not 11-dimensional")
if train.features["action"]["shape"] != (4,):
    raise SystemExit("action contract is not 4-dimensional")
if train.features["action"]["names"] != [
    "action_boom",
    "action_stick",
    "action_bucket",
    "action_swing",
]:
    raise SystemExit("action order is not authoritative")
print(provenance["train_repo_id"])
PY
)"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to start formal ACT training")
print("CUDA gate passed:", torch.__version__, torch.cuda.get_device_name(0))
PY

echo "git_head=$(git rev-parse HEAD)"
git status --short || true
sha256sum /home/zhaoshuai/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
echo "[$(date --iso-8601=seconds)] starting ACT swing-zero training: steps=200000 batch=2"

set -o pipefail
lerobot-train \
  --dataset.repo_id="$TRAIN_REPO" \
  --dataset.root="$DERIVED_SPLIT/train" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --dataset.image_transforms.enable=false \
  --output_dir="$RUN" \
  --job_name=act_excavator_dig_54ep_swing_zero_seed2026_200k \
  --batch_size=2 \
  --num_workers=2 \
  --steps=200000 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=20000 \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed=2026 \
  2>&1 | tee "$TRAIN_LOG"

echo "[$(date --iso-8601=seconds)] ACT swing-zero training completed"

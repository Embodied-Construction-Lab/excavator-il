#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ACT_RUNTIME_IMAGE:-excavator-act-inference:jp72-pytorch261}"
deployment_root="/home/jetson16/workspace_excavator/act_inference"
collection_config="${repo_dir}/config/collection.orin.json"
backbone_cache="${deployment_root}/torch-cache"
backbone_weight="${backbone_cache}/checkpoints/resnet18-f37072fd.pth"
backbone_weight_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
runtime_uid="$(id -u)"
runtime_gid="$(id -g)"
serial_gid="$(stat -c '%g' /dev/ttyTHS1)"
front_camera_device="$(python3 -c '
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
device = config.get("camera_front", {}).get("device")
if not isinstance(device, str) or not device.startswith("/dev/"):
    raise SystemExit("collection camera_front.device must be an absolute /dev path")
print(device)
' "${collection_config}")"
front_camera_device_resolved="$(readlink -e -- "${front_camera_device}")"
camera_gid="$(stat -c '%g' "${front_camera_device_resolved}")"

test -c /dev/ttyTHS1
test -c "${front_camera_device_resolved}"
test -d "${deployment_root}/checkpoint_swing_zero_200000"
test -f "${deployment_root}/deployment/deployment_manifest.json"
test -f "${backbone_weight}"
printf '%s  %s\n' "${backbone_weight_sha256}" "${backbone_weight}" | sha256sum -c - >/dev/null
test -f /home/jetson16/workspace_excavator/shared/machine_profile.json
mkdir -p "${deployment_root}/logs"
test -w "${deployment_root}/logs"

echo "启动 ACT Shadow：不会传入 motion authorization，串口写边界保持禁用。"
echo "看到 'ACT hardware ready: mode=shadow' 后保持运行至少 30 秒，再按 Ctrl+C。"

docker_command=(docker)
if ! docker info >/dev/null 2>&1; then
  docker_command=(sudo docker)
fi

exec "${docker_command[@]}" run --rm \
  --runtime=nvidia --gpus all \
  --network=host \
  --read-only \
  --user "${runtime_uid}:${runtime_gid}" \
  --group-add "${serial_gid}" \
  --group-add "${camera_gid}" \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/ttyTHS1 \
  --device "${front_camera_device_resolved}:/dev/video0" \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/tmp/huggingface -e XDG_CACHE_HOME=/tmp/cache \
  -v "${backbone_cache}:/tmp/cache/torch/hub:ro" \
  -v "${deployment_root}/checkpoint_swing_zero_200000:/opt/act-checkpoint:ro" \
  -v "${deployment_root}/deployment:/opt/act-deployment:ro" \
  -v /home/jetson16/workspace_excavator/shared:/opt/excavator-config:ro \
  -v "${deployment_root}/logs:/opt/act-runtime-logs" \
  -v "${repo_dir}/config/act_runtime.orin.json:/opt/act-runtime.json:ro" \
  -v "${collection_config}:/opt/collection-runtime.json:ro" \
  "${image}" \
  excavator-il act-runtime --config /opt/act-runtime.json \
    --operator-observation-config /opt/collection-runtime.json

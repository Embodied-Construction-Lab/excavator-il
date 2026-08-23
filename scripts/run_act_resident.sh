#!/usr/bin/env bash
set -euo pipefail

authorization=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    "--authorization")
      [[ $# -ge 2 ]] || { echo "--authorization 缺少值" >&2; exit 2; }
      authorization="$2"
      shift 2
      ;;
    *)
      echo "未知参数：$1" >&2
      exit 2
      ;;
  esac
done

if [[ "${authorization}" != "ALLOW_HYBRID_MACHINE_MOTION" ]]; then
  echo "resident ACT 需要精确授权：ALLOW_HYBRID_MACHINE_MOTION" >&2
  exit 1
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ACT_RUNTIME_IMAGE:-excavator-act-inference:jp72-pytorch261}"
deployment_root="/home/jetson16/workspace_excavator/act_inference"
resident_runtime_root="${RESIDENT_RUNTIME_ROOT:-${HOME}/.local/run/excavator-resident}"
resident_act_socket="${RESIDENT_ACT_SOCKET:-${resident_runtime_root}/act.sock}"
operator_observation_config="/opt/collection-runtime.json"
collection_config="${repo_dir}/config/collection.orin.json"
backbone_cache="${deployment_root}/torch-cache"
backbone_weight="${backbone_cache}/checkpoints/resnet18-f37072fd.pth"
backbone_weight_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
runtime_uid="$(id -u)"
runtime_gid="$(id -g)"
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
camera_gid="$(stat -c '%g' "${front_camera_device}")"

test -c "${front_camera_device}"
test -S "${resident_act_socket}"
test -d "${resident_runtime_root}"
test -r "${resident_runtime_root}"
test -d "${deployment_root}/checkpoint_swing_zero_200000"
test -f "${deployment_root}/deployment/deployment_manifest.json"
test -f "${backbone_weight}"
printf '%s  %s\n' "${backbone_weight_sha256}" "${backbone_weight}" | sha256sum -c - >/dev/null
test -f /home/jetson16/workspace_excavator/shared/machine_profile.json
test -f "${repo_dir}/config/act_runtime.orin.json"
test -f "${collection_config}"
mkdir -p "${deployment_root}/logs"
test -w "${deployment_root}/logs"
if pgrep -f '[p]ython3 -m excavator_il.resident_act_runtime' >/dev/null; then
  echo "拒绝启动：resident ACT worker 已在运行。" >&2
  exit 1
fi

docker_command=(docker)
if ! docker info >/dev/null 2>&1; then
  echo "resident ACT 非交互启动需要 direct docker access（加入 docker group 并重新登录）。" >&2
  exit 1
fi

echo "启动 resident ACT worker：只读复用 ${operator_observation_config} 的相机/观测契约。"
echo "operator-observation-config: ${operator_observation_config}"
echo "该 worker 绝不映射 /dev/ttyTHS1；唯一硬件 owner 仍是 resident Mission owner。"

exec "${docker_command[@]}" run --rm \
  --runtime=nvidia --gpus all \
  --network=host \
  --read-only \
  --user "${runtime_uid}:${runtime_gid}" \
  --group-add "${camera_gid}" \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device "${front_camera_device}:/dev/video0" \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/tmp/huggingface -e XDG_CACHE_HOME=/tmp/cache \
  -v "${backbone_cache}:/tmp/cache/torch/hub:ro" \
  -v "${deployment_root}/checkpoint_swing_zero_200000:/opt/act-checkpoint:ro" \
  -v "${deployment_root}/deployment:/opt/act-deployment:ro" \
  -v /home/jetson16/workspace_excavator/shared:/opt/excavator-config:ro \
  -v "${deployment_root}/logs:/opt/act-runtime-logs" \
  -v "${resident_runtime_root}:/opt/excavator-resident" \
  -v "${repo_dir}/config/act_runtime.orin.json:/opt/act-runtime.json:ro" \
  -v "${collection_config}:${operator_observation_config}:ro" \
  "${image}" \
  /bin/sh -lc \
  'test -f /opt/collection-runtime.json && python3 -m excavator_il.resident_act_runtime --config /opt/act-runtime.json --socket-path /opt/excavator-resident/act.sock --operator-observation-config /opt/collection-runtime.json'

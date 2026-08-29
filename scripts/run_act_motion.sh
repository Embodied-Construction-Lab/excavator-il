#!/usr/bin/env bash
set -euo pipefail

authorization=""
max_steps=""
hardware_start_gate=""
noninteractive=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    "--authorization")
      [[ $# -ge 2 ]] || { echo "--authorization 缺少值" >&2; exit 2; }
      authorization="$2"
      noninteractive=true
      shift 2
      ;;
    "--max-steps")
      [[ $# -ge 2 ]] || { echo "--max-steps 缺少值" >&2; exit 2; }
      max_steps="$2"
      shift 2
      ;;
    "--hardware-start-gate")
      [[ $# -ge 2 ]] || { echo "--hardware-start-gate 缺少值" >&2; exit 2; }
      hardware_start_gate="$2"
      shift 2
      ;;
    *)
      echo "未知参数：$1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${max_steps}" ]] && {
  [[ ! "${max_steps}" =~ ^[0-9]+$ ]] ||
  (( max_steps < 1 || max_steps > 2000 ));
}; then
  echo "--max-steps 必须是 [1, 2000] 内的整数。" >&2
  exit 2
fi
if [[ -n "${hardware_start_gate}" ]] && \
  [[ ! "${hardware_start_gate}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "--hardware-start-gate 必须是安全的单个文件名。" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ACT_RUNTIME_IMAGE:-excavator-act-inference:jp72-pytorch261}"
deployment_root="/home/jetson16/workspace_excavator/act_inference"
collection_config="${repo_dir}/config/collection.orin.json"
act_control_root="${deployment_root}/control"
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
if [[ -n "${hardware_start_gate}" ]]; then
  mkdir -p "${act_control_root}"
  test -w "${act_control_root}"
  rm -f -- "${act_control_root}/${hardware_start_gate}"
fi

competing_pattern='excavator-il (collect|act-runtime)|STM32_USART.py'
if [[ -z "${hardware_start_gate}" ]]; then
  competing_pattern="${competing_pattern}|orin_state_sender.py"
fi
if pgrep -f "${competing_pattern}" >/dev/null; then
  echo "拒绝启动：检测到竞争的 Collector、Runtime 或 STM32 串口进程。" >&2
  exit 1
fi
if [[ -z "${hardware_start_gate}" ]] && \
  fuser /dev/ttyTHS1 "${front_camera_device_resolved}" >/dev/null 2>&1; then
  echo "拒绝启动：串口或相机仍被其他进程占用。" >&2
  exit 1
fi

echo "即将启动 ACT motion Runtime；该模式具备 STM32 写权限。"
echo "Runtime 在 Orin 本地独立推理，不需要启动 PC teleop。"
echo "授权后模型可能立即发送非零杆量；首次验收必须保持发动机关闭。"
echo "继续前确认串口、相机和传感器独占，作业区无人且急停可用。"
confirmation="${authorization}"
if [[ -z "${confirmation}" ]]; then
  read -r -p "请输入 ALLOW_ACT_MACHINE_MOTION 继续：" confirmation
fi
if [[ "${confirmation}" != "ALLOW_ACT_MACHINE_MOTION" ]]; then
  echo "授权不匹配，未启动 Runtime。" >&2
  exit 1
fi

docker_command=(docker)
if ! docker info >/dev/null 2>&1; then
  if ${noninteractive}; then
    echo "非交互 ACT 启动需要 jetson16 直接访问 Docker（加入 docker group 并重新登录）。" >&2
    exit 1
  else
    docker_command=(sudo docker)
  fi
fi
if [[ -n "${hardware_start_gate}" ]]; then
  echo "ACT 预热等待模式：CUDA 预热期间不打开串口和相机；收到内部交接门后才接管硬件。"
fi

runtime_args=(
  excavator-il act-runtime --config /opt/act-runtime.json
  --motion-authorization ALLOW_ACT_MACHINE_MOTION
  --operator-observation-config /opt/collection-runtime.json
)
if [[ -n "${max_steps}" ]]; then
  runtime_args+=(--max-steps "${max_steps}")
fi
if [[ -n "${hardware_start_gate}" ]]; then
  runtime_args+=(--hardware-start-gate "/opt/act-control/${hardware_start_gate}")
fi

control_mount=()
if [[ -n "${hardware_start_gate}" ]]; then
  control_mount=(-v "${act_control_root}:/opt/act-control")
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
  "${control_mount[@]}" \
  "${image}" \
  "${runtime_args[@]}"

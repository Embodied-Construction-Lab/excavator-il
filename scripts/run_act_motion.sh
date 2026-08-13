#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ACT_RUNTIME_IMAGE:-excavator-act-inference:jp72-pytorch261}"
deployment_root="/home/jetson16/workspace_excavator/act_inference"
authentication_key="${deployment_root}/act_operator_hmac.key"
runtime_uid="$(id -u)"
runtime_gid="$(id -g)"
serial_gid="$(stat -c '%g' /dev/ttyTHS1)"
camera_gid="$(stat -c '%g' /dev/video0)"

test -c /dev/ttyTHS1
test -c /dev/video0
test -d "${deployment_root}/checkpoint_parent_split_001054"
test -f "${deployment_root}/deployment/deployment_manifest.json"
test -f /home/jetson16/workspace_excavator/shared/machine_profile.json
test -f "${authentication_key}"
test "$(stat -c '%a' "${authentication_key}")" = "600"
test "$(stat -c '%u' "${authentication_key}")" = "${runtime_uid}"
mkdir -p "${deployment_root}/logs"
test -w "${deployment_root}/logs"

if pgrep -f 'excavator-il (collect|act-runtime)|orin_state_sender.py|STM32_USART.py' \
  >/dev/null; then
  echo "拒绝启动：检测到竞争的 Collector、Runtime 或 STM32 串口进程。" >&2
  exit 1
fi
if fuser /dev/ttyTHS1 /dev/video0 >/dev/null 2>&1; then
  echo "拒绝启动：串口或相机仍被其他进程占用。" >&2
  exit 1
fi

echo "即将启动 ACT motion Runtime；该模式具备 STM32 写权限。"
echo "首次验收必须：发动机关闭、双杆居中、deadman 全程释放。"
read -r -p "请输入 ALLOW_ACT_MACHINE_MOTION 继续：" confirmation
if [[ "${confirmation}" != "ALLOW_ACT_MACHINE_MOTION" ]]; then
  echo "授权不匹配，未启动 Runtime。" >&2
  exit 1
fi

exec sudo docker run --rm \
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
  --device /dev/video0 \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/tmp/huggingface -e XDG_CACHE_HOME=/tmp/cache \
  -v "${deployment_root}/checkpoint_parent_split_001054:/opt/act-checkpoint:ro" \
  -v "${deployment_root}/deployment:/opt/act-deployment:ro" \
  -v /home/jetson16/workspace_excavator/shared:/opt/excavator-config:ro \
  -v "${deployment_root}/logs:/opt/act-runtime-logs" \
  -v "${authentication_key}:/run/secrets/act_operator_hmac:ro" \
  -v "${repo_dir}/config/act_runtime.orin.json:/opt/act-runtime.json:ro" \
  "${image}" \
  excavator-il act-runtime --config /opt/act-runtime.json \
  --motion-authorization ALLOW_ACT_MACHINE_MOTION

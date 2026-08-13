#!/usr/bin/env bash
set -u

image="${ACT_RUNTIME_IMAGE:-excavator-act-inference:jp72-pytorch261}"
runtime_uid="$(id -u)"
runtime_gid="$(id -g)"
serial_gid="$(stat -c '%g' /dev/ttyTHS1)"
video_gid="$(stat -c '%g' /dev/video0)"
render_gid="$(stat -c '%g' /dev/dri/renderD128)"
probe="import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); torch.cuda.synchronize()"

run_case() {
  local case_name="$1"
  shift
  echo "===== ${case_name} ====="
  if sudo docker run --rm --runtime=nvidia --gpus all "$@" \
    "${image}" python3 -c "${probe}"; then
    echo "RESULT ${case_name}=PASS"
  else
    echo "RESULT ${case_name}=FAIL"
  fi
}

run_case root_baseline

run_case launcher_groups \
  --user "${runtime_uid}:${runtime_gid}" \
  --group-add "${serial_gid}" \
  --group-add "${video_gid}"

run_case launcher_groups_plus_render \
  --user "${runtime_uid}:${runtime_gid}" \
  --group-add "${serial_gid}" \
  --group-add "${video_gid}" \
  --group-add "${render_gid}"

run_case hardened_plus_render \
  --network=none \
  --read-only \
  --user "${runtime_uid}:${runtime_gid}" \
  --group-add "${serial_gid}" \
  --group-add "${video_gid}" \
  --group-add "${render_gid}" \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --ulimit memlock=-1 --ulimit stack=67108864

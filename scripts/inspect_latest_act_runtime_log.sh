#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  shadow|motion) ;;
  *)
    echo "usage: $0 shadow|motion" >&2
    exit 2
    ;;
esac

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${ACT_RUNTIME_LOG_ROOT:-/home/jetson16/workspace_excavator/act_inference/logs}"

latest_log="$({
  find "$log_root" -maxdepth 1 -type f \
    -name "act_runtime_${mode}_*.jsonl" -printf '%T@ %p\n' 2>/dev/null || true
} | sort -nr | sed -n '1s/^[^ ]* //p')"

if [[ -z "$latest_log" ]]; then
  echo "error: no act_runtime_${mode}_*.jsonl under $log_root" >&2
  exit 2
fi

export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m excavator_il.cli inspect-act-runtime-log "$latest_log" \
  --mode "$mode" \
  --config "${repo_dir}/config/act_runtime.orin.json"

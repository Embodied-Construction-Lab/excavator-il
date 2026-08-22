#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
PC_CONTROL_IP="192.168.50.1"
ORIN_CONTROL_IP="192.168.50.2"
PC_CONTROL_INTERFACE="enx00e04c266130"
ORIN_CONTROL_INTERFACE="enP8p1s0"

usage() {
  echo "usage: sudo bash scripts/configure_lan_time_sync.sh {pc|orin}" >&2
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "error: this command must run through sudo" >&2
    exit 2
  fi
}

require_address() {
  local interface="$1"
  local address="$2"
  if ! ip -4 -o address show dev "${interface}" | grep -Fq " ${address}/"; then
    echo "error: ${interface} does not own ${address}; refusing to change time service" >&2
    exit 2
  fi
}

backup_if_present() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    cp --archive "${path}" "${path}.backup.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
}

configure_pc() {
  require_address "${PC_CONTROL_INTERFACE}" "${PC_CONTROL_IP}"

  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y chrony

  local target="/etc/chrony/conf.d/excavator-lan-server.conf"
  install -d -m 0755 "$(dirname "${target}")"
  backup_if_present "${target}"
  install -m 0644 /dev/null "${target}"
  printf '%s\n' \
    '# Managed by excavator-il/scripts/configure_lan_time_sync.sh' \
    '# Serve one trusted client on the isolated PC-Orin control link.' \
    "allow ${ORIN_CONTROL_IP}/32" \
    '# Reject high-uncertainty WAN sources instead of moving the PC clock by tens of ms.' \
    'maxdistance 0.05' \
    '# Keep runtime corrections slow enough for the Orin client to track continuously.' \
    'maxslewrate 100' \
    '# Keep PC and Orin mutually aligned if the public NTP uplink is absent.' \
    'local stratum 10' \
    >"${target}"

  chronyd -p >/dev/null
  systemctl enable --now chrony.service
  systemctl restart chrony.service

  if command -v ufw >/dev/null && ufw status | grep -q '^Status: active'; then
    ufw allow in on "${PC_CONTROL_INTERFACE}" \
      proto udp from "${ORIN_CONTROL_IP}" to "${PC_CONTROL_IP}" port 123 \
      comment 'Orin LAN NTP'
  fi

  echo "PC LAN time source configured on ${PC_CONTROL_IP}:123/udp"
  chronyc tracking
}

configure_orin() {
  require_address "${ORIN_CONTROL_INTERFACE}" "${ORIN_CONTROL_IP}"

  local target="/etc/systemd/timesyncd.conf.d/excavator-lan.conf"
  install -d -m 0755 "$(dirname "${target}")"
  backup_if_present "${target}"
  install -m 0644 /dev/null "${target}"
  printf '%s\n' \
    '# Managed by excavator-il/scripts/configure_lan_time_sync.sh' \
    '[Time]' \
    "NTP=${PC_CONTROL_IP}" \
    'FallbackNTP=' \
    'PollIntervalMinSec=16' \
    'PollIntervalMaxSec=16' \
    >"${target}"

  timedatectl set-ntp true
  systemctl restart systemd-timesyncd.service

  echo "Orin time client configured to use ${PC_CONTROL_IP} over the control link"
  timedatectl status
}

case "${ROLE}" in
  pc)
    require_root
    configure_pc
    ;;
  orin)
    require_root
    configure_orin
    ;;
  *)
    usage
    exit 2
    ;;
esac

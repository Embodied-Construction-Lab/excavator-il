"""PC-side 20 Hz dual-joystick sender for demonstration collection."""

from __future__ import annotations

import json
import math
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .joystick_protocol import (
    ControllerIdentity,
    JoystickPacket,
    encode_joystick_packet,
)


TELEOP_CONFIG_SCHEMA_VERSION = "excavator_teleop_config.v1"


@dataclass(frozen=True)
class DeviceSnapshot:
    device_id: str
    name: str
    axes: tuple[float, float, float]
    buttons: tuple[bool, ...]


@dataclass(frozen=True)
class TeleopConfig:
    orin_host: str
    orin_port: int
    rate_hz: int
    mapping_id: str
    calibration_id: str
    device_ids: tuple[str, str]
    axis_indices: tuple[tuple[int, int, int], tuple[int, int, int]]
    deadman_slot: int
    deadman_button: int

    @classmethod
    def load(cls, path: str | Path) -> "TeleopConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("teleop config must be an object")
        if value.get("schema_version") != TELEOP_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {TELEOP_CONFIG_SCHEMA_VERSION}"
            )
        devices = value.get("devices")
        if not isinstance(devices, list) or len(devices) != 2:
            raise ValueError("teleop config devices must contain exactly two entries")
        device_ids: list[str] = []
        axis_indices: list[tuple[int, int, int]] = []
        for index, device in enumerate(devices, start=1):
            if not isinstance(device, Mapping):
                raise ValueError(f"devices[{index}] must be an object")
            device_id = device.get("device_id")
            indices = device.get("axis_indices")
            if not isinstance(device_id, str) or not device_id:
                raise ValueError(f"devices[{index}].device_id must be non-empty")
            if (
                not isinstance(indices, list)
                or len(indices) != 3
                or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in indices)
            ):
                raise ValueError(f"devices[{index}].axis_indices must contain three indices")
            device_ids.append(device_id)
            axis_indices.append(tuple(indices))
        deadman = value.get("deadman")
        if not isinstance(deadman, Mapping):
            raise ValueError("teleop config deadman must be an object")
        config = cls(
            orin_host=str(value.get("orin_host", "")),
            orin_port=int(value.get("orin_port", 0)),
            rate_hz=int(value.get("rate_hz", 0)),
            mapping_id=str(value.get("mapping_id", "")),
            calibration_id=str(value.get("calibration_id", "")),
            device_ids=(device_ids[0], device_ids[1]),
            axis_indices=(axis_indices[0], axis_indices[1]),
            deadman_slot=int(deadman.get("controller_slot", 0)),
            deadman_button=int(deadman.get("button_index", -1)),
        )
        if not config.orin_host or not 1 <= config.orin_port <= 65535:
            raise ValueError("orin_host and orin_port must identify a valid UDP endpoint")
        if config.rate_hz != 20:
            raise ValueError("demonstration teleop rate_hz must be 20")
        if not config.mapping_id or not config.calibration_id:
            raise ValueError("mapping_id and calibration_id must be non-empty")
        if config.deadman_slot not in (1, 2) or config.deadman_button < 0:
            raise ValueError("deadman must identify a non-negative button on slot 1 or 2")
        return config


def build_joystick_packet(
    *,
    sample_seq: int,
    session_id: str,
    pc_sample_monotonic_ns: int,
    pc_sample_wall_ns: int,
    devices: tuple[DeviceSnapshot, DeviceSnapshot],
    deadman_pressed: bool,
    mapping_id: str,
    calibration_id: str,
) -> JoystickPacket:
    for device in devices:
        if len(device.axes) != 3:
            raise ValueError("each joystick must provide X/Y/Z axes")
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in device.axes):
            raise ValueError("joystick axes must be finite and within [-1, 1]")
    return JoystickPacket(
        session_id=session_id,
        sample_seq=sample_seq,
        pc_sample_monotonic_ns=pc_sample_monotonic_ns,
        pc_sample_wall_ns=pc_sample_wall_ns,
        axes=devices[0].axes + devices[1].axes,
        controllers=(
            ControllerIdentity(1, devices[0].device_id, devices[0].name, devices[0].buttons),
            ControllerIdentity(2, devices[1].device_id, devices[1].name, devices[1].buttons),
        ),
        deadman_pressed=deadman_pressed,
        mapping_id=mapping_id,
        calibration_id=calibration_id,
    )


def list_pygame_devices() -> list[dict[str, Any]]:
    pygame = _load_pygame()
    pygame.init()
    pygame.joystick.init()
    devices: list[dict[str, Any]] = []
    try:
        for index in range(pygame.joystick.get_count()):
            joystick = pygame.joystick.Joystick(index)
            joystick.init()
            devices.append(
                {
                    "index": index,
                    "device_id": joystick.get_guid(),
                    "name": joystick.get_name(),
                    "axis_count": joystick.get_numaxes(),
                    "button_count": joystick.get_numbuttons(),
                }
            )
    finally:
        pygame.joystick.quit()
        pygame.quit()
    return devices


def _load_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("teleop requires the optional pygame dependency") from exc
    return pygame


def _open_configured_devices(pygame: Any, config: TeleopConfig) -> tuple[Any, Any]:
    available: dict[str, list[Any]] = {}
    for index in range(pygame.joystick.get_count()):
        joystick = pygame.joystick.Joystick(index)
        joystick.init()
        available.setdefault(joystick.get_guid(), []).append(joystick)
    selected: list[Any] = []
    used_instances: set[int] = set()
    for device_id in config.device_ids:
        candidates = available.get(device_id, [])
        joystick = next(
            (item for item in candidates if item.get_instance_id() not in used_instances),
            None,
        )
        if joystick is None:
            raise RuntimeError(f"configured joystick is not connected: {device_id}")
        selected.append(joystick)
        used_instances.add(joystick.get_instance_id())
    return selected[0], selected[1]


def _snapshot(joystick: Any, indices: Sequence[int]) -> DeviceSnapshot:
    if any(index >= joystick.get_numaxes() for index in indices):
        raise RuntimeError(f"joystick {joystick.get_name()} does not expose configured axes")
    return DeviceSnapshot(
        device_id=joystick.get_guid(),
        name=joystick.get_name(),
        axes=tuple(float(joystick.get_axis(index)) for index in indices),
        buttons=tuple(bool(joystick.get_button(index)) for index in range(joystick.get_numbuttons())),
    )


def run_teleop(config: TeleopConfig, *, print_every: int = 20) -> None:
    """Continuously send numeric joystick samples at the configured 20 Hz."""
    pygame = _load_pygame()
    pygame.init()
    pygame.joystick.init()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    destination = (config.orin_host, config.orin_port)
    period_ns = int(1_000_000_000 / config.rate_hz)
    sequence = 0
    session_id = uuid.uuid4().hex
    next_deadline = time.monotonic_ns()
    last_ack = -1
    accepted_ack_count = 0
    rejected_ack_count = 0
    try:
        devices = _open_configured_devices(pygame, config)
        while True:
            pygame.event.pump()
            snapshots = (
                _snapshot(devices[0], config.axis_indices[0]),
                _snapshot(devices[1], config.axis_indices[1]),
            )
            deadman_buttons = snapshots[config.deadman_slot - 1].buttons
            if config.deadman_button >= len(deadman_buttons):
                raise RuntimeError("configured deadman button does not exist")
            packet = build_joystick_packet(
                sample_seq=sequence,
                session_id=session_id,
                pc_sample_monotonic_ns=time.monotonic_ns(),
                pc_sample_wall_ns=time.time_ns(),
                devices=snapshots,
                deadman_pressed=deadman_buttons[config.deadman_button],
                mapping_id=config.mapping_id,
                calibration_id=config.calibration_id,
            )
            sock.sendto(encode_joystick_packet(packet), destination)
            try:
                while True:
                    ack, source = sock.recvfrom(2048)
                    if source != destination:
                        continue
                    value = json.loads(ack.decode("utf-8"))
                    if value.get("schema_version") == "excavator_joystick_ack.v1":
                        if value.get("accepted") and value.get("sample_seq") is not None:
                            last_ack = max(last_ack, int(value["sample_seq"]))
                            accepted_ack_count += 1
                        else:
                            rejected_ack_count += 1
            except BlockingIOError:
                pass
            if print_every > 0 and sequence % print_every == 0:
                print(
                    f"teleop seq={sequence} ack={last_ack} "
                    f"ack_lag={sequence - last_ack if last_ack >= 0 else 'unknown'} "
                    f"accepted_acks={accepted_ack_count} rejected_acks={rejected_ack_count} "
                    f"deadman={packet.deadman_pressed} axes={packet.axes}",
                    flush=True,
                )
            sequence += 1
            next_deadline += period_ns
            remaining_ns = next_deadline - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000.0)
            elif remaining_ns < -period_ns:
                next_deadline = time.monotonic_ns()
    finally:
        sock.close()
        pygame.joystick.quit()
        pygame.quit()

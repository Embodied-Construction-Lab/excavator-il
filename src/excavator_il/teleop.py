"""PC-side 20 Hz dual-joystick sender for demonstration collection."""

from __future__ import annotations

import ctypes
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


TELEOP_CONFIG_SCHEMA_VERSION = "excavator_teleop_config.v4"


@dataclass(frozen=True)
class DeviceSnapshot:
    device_id: str
    name: str
    axes: tuple[float, float, float]
    buttons: tuple[bool, ...]


@dataclass(frozen=True)
class ConnectedJoystick:
    index: int
    device_id: str
    device_path: Path
    instance_id: int


@dataclass(frozen=True)
class TeleopConfig:
    orin_host: str
    orin_port: int
    rate_hz: int
    mapping_id: str
    calibration_id: str
    device_ids: tuple[str, str]
    device_paths: tuple[Path, Path]
    axis_indices: tuple[tuple[int, int, int], tuple[int, int, int]]
    deadman_slot: int
    deadman_button: int
    startup_axis_abs_max: float = 0.15
    startup_stable_samples: int = 10
    startup_timeout_s: float = 5.0

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
        device_paths: list[Path] = []
        axis_indices: list[tuple[int, int, int]] = []
        for index, device in enumerate(devices, start=1):
            if not isinstance(device, Mapping):
                raise ValueError(f"devices[{index}] must be an object")
            device_id = device.get("device_id")
            device_path = device.get("device_path")
            indices = device.get("axis_indices")
            if not isinstance(device_id, str) or not device_id:
                raise ValueError(f"devices[{index}].device_id must be non-empty")
            if not isinstance(device_path, str) or not device_path:
                raise ValueError(f"devices[{index}].device_path must be non-empty")
            parsed_path = Path(device_path).expanduser()
            if not parsed_path.is_absolute():
                raise ValueError(f"devices[{index}].device_path must be absolute")
            if (
                not isinstance(indices, list)
                or len(indices) != 3
                or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in indices)
            ):
                raise ValueError(
                    f"devices[{index}].axis_indices must contain X/Y/Z indices"
                )
            device_ids.append(device_id)
            device_paths.append(parsed_path)
            axis_indices.append(tuple(indices))
        deadman = value.get("deadman")
        if not isinstance(deadman, Mapping):
            raise ValueError("teleop config deadman must be an object")
        startup_gate = value.get("startup_gate", {})
        if not isinstance(startup_gate, Mapping):
            raise ValueError("teleop config startup_gate must be an object")
        if device_paths[0] == device_paths[1]:
            raise ValueError("devices device_path values must be distinct")
        config = cls(
            orin_host=str(value.get("orin_host", "")),
            orin_port=int(value.get("orin_port", 0)),
            rate_hz=int(value.get("rate_hz", 0)),
            mapping_id=str(value.get("mapping_id", "")),
            calibration_id=str(value.get("calibration_id", "")),
            device_ids=(device_ids[0], device_ids[1]),
            device_paths=(device_paths[0], device_paths[1]),
            axis_indices=(axis_indices[0], axis_indices[1]),
            deadman_slot=int(deadman.get("controller_slot", 0)),
            deadman_button=int(deadman.get("button_index", -1)),
            startup_axis_abs_max=float(
                startup_gate.get("axis_abs_max", 0.15)
            ),
            startup_stable_samples=int(
                startup_gate.get("stable_samples", 10)
            ),
            startup_timeout_s=float(startup_gate.get("timeout_s", 5.0)),
        )
        if not config.orin_host or not 1 <= config.orin_port <= 65535:
            raise ValueError("orin_host and orin_port must identify a valid UDP endpoint")
        if config.rate_hz != 20:
            raise ValueError("demonstration teleop rate_hz must be 20")
        if not config.mapping_id or not config.calibration_id:
            raise ValueError("mapping_id and calibration_id must be non-empty")
        if config.deadman_slot not in (1, 2) or config.deadman_button < 0:
            raise ValueError("deadman must identify a non-negative button on slot 1 or 2")
        if not 0.0 < config.startup_axis_abs_max < 1.0:
            raise ValueError("startup_gate.axis_abs_max must be within (0, 1)")
        if config.startup_stable_samples <= 0:
            raise ValueError("startup_gate.stable_samples must be positive")
        if not math.isfinite(config.startup_timeout_s) or config.startup_timeout_s <= 0:
            raise ValueError("startup_gate.timeout_s must be finite and positive")
        return config


def select_configured_device_indices(
    config: TeleopConfig,
    available: Sequence[ConnectedJoystick],
) -> tuple[int, int]:
    selected: list[int] = []
    for device_id, device_path in zip(
        config.device_ids, config.device_paths, strict=True
    ):
        try:
            expected_path = device_path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"configured joystick path is unavailable: {device_path}") from exc
        matches = [
            device.index
            for device in available
            if device.device_id == device_id
            and device.device_path.resolve(strict=True) == expected_path
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"configured joystick identity did not match exactly one device: "
                f"{device_id} at {device_path}"
            )
        selected.append(matches[0])
    if len(set(selected)) != len(selected):
        raise RuntimeError("configured joystick paths must resolve to distinct physical devices")
    return selected[0], selected[1]


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
        for connected in _connected_pygame_devices(pygame):
            joystick = pygame.joystick.Joystick(connected.index)
            joystick.init()
            devices.append(
                {
                    "index": connected.index,
                    "device_id": connected.device_id,
                    "device_path": str(connected.device_path),
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


def _sdl_joystick_path_function(pygame: Any):
    sdl_version = tuple(pygame.get_sdl_version())
    if sdl_version < (2, 24, 0):
        rendered_version = ".".join(str(part) for part in sdl_version)
        raise RuntimeError(
            "stable joystick device paths require SDL 2.24 or newer; "
            f"loaded SDL {rendered_version}"
        )
    candidates: set[Path] = set()
    maps_path = Path("/proc/self/maps")
    if maps_path.is_file():
        for line in maps_path.read_text(encoding="utf-8").splitlines():
            raw_path = line.split()[-1]
            if "libSDL2" in raw_path and raw_path.startswith("/"):
                candidates.add(Path(raw_path))
    package_root = Path(pygame.__file__).resolve().parent.parent
    candidates.update((package_root / "pygame.libs").glob("libSDL2-*.so*"))

    for candidate in sorted(candidates):
        try:
            library = ctypes.CDLL(str(candidate))
            function = library.SDL_JoystickPathForIndex
        except (OSError, AttributeError):
            continue
        function.argtypes = [ctypes.c_int]
        function.restype = ctypes.c_char_p
        return function
    raise RuntimeError("pygame SDL does not expose joystick device paths")


def _connected_pygame_devices(pygame: Any) -> tuple[ConnectedJoystick, ...]:
    path_for_index = _sdl_joystick_path_function(pygame)
    connected: list[ConnectedJoystick] = []
    for index in range(pygame.joystick.get_count()):
        joystick = pygame.joystick.Joystick(index)
        joystick.init()
        raw_path = path_for_index(index)
        if raw_path is None:
            raise RuntimeError(f"SDL did not report a path for joystick index {index}")
        connected.append(
            ConnectedJoystick(
                index=index,
                device_id=joystick.get_guid(),
                device_path=Path(raw_path.decode("utf-8")),
                instance_id=joystick.get_instance_id(),
            )
        )
    return tuple(connected)


def _open_configured_devices(pygame: Any, config: TeleopConfig) -> tuple[Any, Any]:
    connected = _connected_pygame_devices(pygame)
    indices = select_configured_device_indices(config, connected)
    snapshots = {device.index: device for device in connected}
    selected = tuple(pygame.joystick.Joystick(index) for index in indices)
    for index, joystick in zip(indices, selected, strict=True):
        joystick.init()
        snapshot = snapshots[index]
        if (
            joystick.get_instance_id() != snapshot.instance_id
            or joystick.get_guid() != snapshot.device_id
        ):
            for opened in selected:
                if hasattr(opened, "quit"):
                    opened.quit()
            raise RuntimeError("joystick device changed during startup")
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


def _wait_for_safe_startup(
    pygame: Any,
    devices: tuple[Any, Any],
    config: TeleopConfig,
) -> None:
    """Wait for stable neutral hardware before any UDP socket can exist."""
    deadline = time.monotonic() + config.startup_timeout_s
    stable_samples = 0
    period_s = 1.0 / config.rate_hz
    while stable_samples < config.startup_stable_samples:
        pygame.event.pump()
        snapshots = (
            _snapshot(devices[0], config.axis_indices[0]),
            _snapshot(devices[1], config.axis_indices[1]),
        )
        deadman_buttons = snapshots[config.deadman_slot - 1].buttons
        if config.deadman_button >= len(deadman_buttons):
            raise RuntimeError("configured deadman button does not exist")
        neutral = (
            not deadman_buttons[config.deadman_button]
            and all(
                abs(axis) <= config.startup_axis_abs_max
                for snapshot in snapshots
                for axis in snapshot.axes
            )
        )
        stable_samples = stable_samples + 1 if neutral else 0
        if stable_samples >= config.startup_stable_samples:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "joystick startup gate timed out: release deadman and center all configured X/Y/Z axes"
            )
        time.sleep(period_s)


def run_teleop(config: TeleopConfig, *, print_every: int = 20) -> None:
    """Continuously send numeric joystick samples at the configured 20 Hz."""
    pygame = _load_pygame()
    pygame.init()
    pygame.joystick.init()
    sock: socket.socket | None = None
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
        _wait_for_safe_startup(pygame, devices, config)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
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
            payload = encode_joystick_packet(packet)
            sock.sendto(payload, destination)
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
        if sock is not None:
            sock.close()
        pygame.joystick.quit()
        pygame.quit()

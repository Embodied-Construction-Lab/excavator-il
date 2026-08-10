"""Strict configuration boundary for the Orin demonstration collector."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


COLLECTION_CONFIG_SCHEMA_VERSION = "excavator_collection_config.v1"


@dataclass(frozen=True)
class JoystickUdpConfig:
    bind_host: str
    port: int
    allowed_pc_host: str
    timeout_ms: int


@dataclass(frozen=True)
class ControllerConfig:
    device_ids: tuple[str, str]
    mapping_id: str
    calibration_id: str
    deadzone: float


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baudrate: int


@dataclass(frozen=True)
class CameraConfig:
    device: str
    width: int
    height: int
    nominal_fps: int
    jpeg_quality: int


@dataclass(frozen=True)
class EpisodeDefaults:
    dig_target_m: tuple[float, float, float]
    material_id: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class CollectionConfig:
    data_root: Path
    joystick: JoystickUdpConfig
    controllers: ControllerConfig
    serial: SerialConfig
    camera: CameraConfig
    episode_control_socket: Path
    episode_defaults: EpisodeDefaults


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} must be in [{minimum}, {maximum}]")
    return value


def load_collection_config(path: str | Path) -> CollectionConfig:
    config_path = Path(path).expanduser()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load collection config {config_path}: {exc}") from exc
    root = _object(raw, "config")
    if root.get("schema_version") != COLLECTION_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {COLLECTION_CONFIG_SCHEMA_VERSION}"
        )

    joystick = _object(root.get("joystick_udp"), "joystick_udp")
    controllers = _object(root.get("controllers"), "controllers")
    serial = _object(root.get("stm32_serial"), "stm32_serial")
    camera = _object(root.get("camera_front"), "camera_front")
    defaults = _object(root.get("episode_defaults"), "episode_defaults")

    device_ids = controllers.get("device_ids")
    if (
        not isinstance(device_ids, list)
        or len(device_ids) != 2
        or any(not isinstance(value, str) or not value for value in device_ids)
    ):
        raise ValueError("controllers.device_ids must contain two IDs")
    deadzone = float(controllers.get("deadzone", -1.0))
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("controllers.deadzone must be in [0, 1)")
    baudrate = _integer(
        serial.get("baudrate"), "stm32_serial.baudrate", minimum=1, maximum=10_000_000
    )
    if baudrate != 460800:
        raise ValueError("stm32_serial.baudrate must be 460800 for telemetry v2")
    target = defaults.get("dig_target_m")
    if not isinstance(target, list) or len(target) != 3:
        raise ValueError("episode_defaults.dig_target_m must contain three numbers")
    target_values = tuple(float(value) for value in target)
    if any(not math.isfinite(value) for value in target_values):
        raise ValueError("episode_defaults.dig_target_m must be finite")

    return CollectionConfig(
        data_root=Path(_text(root.get("data_root"), "data_root")).expanduser(),
        joystick=JoystickUdpConfig(
            bind_host=_text(joystick.get("bind_host"), "joystick_udp.bind_host"),
            port=_integer(
                joystick.get("port"), "joystick_udp.port", minimum=1, maximum=65535
            ),
            allowed_pc_host=_text(
                joystick.get("allowed_pc_host"), "joystick_udp.allowed_pc_host"
            ),
            timeout_ms=_integer(
                joystick.get("timeout_ms"),
                "joystick_udp.timeout_ms",
                minimum=50,
                maximum=1000,
            ),
        ),
        controllers=ControllerConfig(
            device_ids=(device_ids[0], device_ids[1]),
            mapping_id=_text(controllers.get("mapping_id"), "controllers.mapping_id"),
            calibration_id=_text(
                controllers.get("calibration_id"), "controllers.calibration_id"
            ),
            deadzone=deadzone,
        ),
        serial=SerialConfig(
            port=_text(serial.get("port"), "stm32_serial.port"),
            baudrate=baudrate,
        ),
        camera=CameraConfig(
            device=_text(camera.get("device"), "camera_front.device"),
            width=_integer(
                camera.get("width"), "camera_front.width", minimum=16, maximum=8192
            ),
            height=_integer(
                camera.get("height"), "camera_front.height", minimum=16, maximum=8192
            ),
            nominal_fps=_integer(
                camera.get("fps"), "camera_front.fps", minimum=1, maximum=120
            ),
            jpeg_quality=_integer(
                camera.get("jpeg_quality"),
                "camera_front.jpeg_quality",
                minimum=1,
                maximum=100,
            ),
        ),
        episode_control_socket=Path(
            _text(root.get("episode_control_socket"), "episode_control_socket")
        ).expanduser(),
        episode_defaults=EpisodeDefaults(
            dig_target_m=target_values,
            material_id=_text(defaults.get("material_id"), "episode_defaults.material_id"),
            provenance=MappingProxyType(
                dict(_object(defaults.get("provenance"), "episode_defaults.provenance"))
            ),
        ),
    )

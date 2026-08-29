"""Strict configuration boundary for the Orin demonstration collector."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


COLLECTION_CONFIG_SCHEMA_VERSION = "excavator_collection_config.v2"
LEGACY_COLLECTION_CONFIG_SCHEMA_VERSION = "excavator_collection_config.v1"
COLLECTION_TASK_VARIANTS = frozenset({"dig_only", "dig_transport_dump"})
COLLECTION_ZONE_IDS = frozenset(f"zone_{index:02d}" for index in range(1, 7))
RECORDING_PURPOSES = frozenset({"demonstration", "diagnostic"})
_PROTOCOL_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TARGET_SOURCE_FIELDS = frozenset(
    {"repository", "path", "sha256", "commit", "dirty"}
)


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
class CameraPreviewConfig:
    bind_host: str
    port: int


@dataclass(frozen=True)
class MachineStateUdpConfig:
    host: str
    port: int
    machine_id: str


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
    camera_front: CameraConfig
    camera_dump: CameraConfig | None
    episode_control_socket: Path
    episode_defaults: EpisodeDefaults
    camera_preview: CameraPreviewConfig | None = None
    machine_state_udp: MachineStateUdpConfig | None = None

    @property
    def camera(self) -> CameraConfig:
        """Backward-compatible alias for the authoritative front camera."""
        return self.camera_front

    @property
    def cameras(self) -> Mapping[str, CameraConfig]:
        """Configured RGB streams indexed by their semantic role."""
        values = {"front": self.camera_front}
        if self.camera_dump is not None:
            values["dump"] = self.camera_dump
        return MappingProxyType(values)


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


def validate_collection_protocol(
    *,
    task_variant: str | None,
    soil_reset_block_id: str | None,
    dig_point_id: str | None,
) -> Mapping[str, str]:
    """Validate the indivisible trial labels used for grouped experiments."""
    values = (task_variant, soil_reset_block_id, dig_point_id)
    if all(value is None for value in values):
        return MappingProxyType({})
    if any(value is None for value in values):
        raise ValueError(
            "task_variant, soil_reset_block_id and dig_point_id must be provided together"
        )
    assert task_variant is not None
    assert soil_reset_block_id is not None
    assert dig_point_id is not None
    if task_variant not in COLLECTION_TASK_VARIANTS:
        raise ValueError("task_variant must be dig_only or dig_transport_dump")
    if (
        len(soil_reset_block_id) > 48
        or _PROTOCOL_ID.fullmatch(soil_reset_block_id) is None
    ):
        raise ValueError(
            "soil_reset_block_id must be a normalized lowercase underscore "
            "identifier of at most 48 characters"
        )
    if _PROTOCOL_ID.fullmatch(dig_point_id) is None:
        raise ValueError(
            "dig_point_id must be a normalized lowercase underscore identifier"
        )
    return MappingProxyType(
        {
            "task_variant": task_variant,
            "soil_reset_block_id": soil_reset_block_id,
            "dig_point_id": dig_point_id,
        }
    )


def validate_collection_labels(
    *,
    collection_zone_id: str | None,
    dig_repeat_index: int | None,
    operator_note: str | None = None,
) -> Mapping[str, str | int]:
    """Validate optional operator labels without changing the trial protocol."""
    if collection_zone_id is None and dig_repeat_index is None and operator_note is None:
        return MappingProxyType({})
    if collection_zone_id is None or dig_repeat_index is None:
        raise ValueError(
            "collection_zone_id and dig_repeat_index must be provided together"
        )
    if collection_zone_id not in COLLECTION_ZONE_IDS:
        raise ValueError("collection_zone_id must be zone_01 through zone_06")
    repeat_index = _integer(
        dig_repeat_index,
        "dig_repeat_index",
        minimum=1,
        maximum=3,
    )
    if operator_note is None:
        note = ""
    elif not isinstance(operator_note, str):
        raise ValueError("operator_note must be text")
    else:
        note = operator_note.strip()
    if len(note) > 200 or any(character in note for character in "\r\n"):
        raise ValueError(
            "operator_note must be a single line of at most 200 characters"
        )
    return MappingProxyType(
        {
            "collection_zone_id": collection_zone_id,
            "dig_repeat_index": repeat_index,
            "operator_note": note,
        }
    )


def validate_recording_purpose(value: object) -> str:
    """Validate whether an Episode is training data or a hardware diagnostic."""
    if not isinstance(value, str) or value not in RECORDING_PURPOSES:
        raise ValueError(
            "recording_purpose must be demonstration or diagnostic"
        )
    return value


def validate_target_source_provenance(
    value: object,
) -> Mapping[str, str | bool]:
    """Validate the immutable PC-side source used to select a DIG target."""
    if not isinstance(value, Mapping) or set(value) != _TARGET_SOURCE_FIELDS:
        raise ValueError(
            "target_source_provenance must contain exactly repository, path, "
            "sha256, commit and dirty"
        )
    if value["repository"] != "airylidar":
        raise ValueError("target_source_provenance.repository must be airylidar")
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("target_source_provenance.path must be non-empty text")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != raw_path:
        raise ValueError(
            "target_source_provenance.path must be a normalized "
            "repository-relative path"
        )
    sha256 = value["sha256"]
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise ValueError("target_source_provenance.sha256 must be lowercase SHA-256")
    commit = value["commit"]
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError(
            "target_source_provenance.commit must be a lowercase 40-character "
            "Git commit"
        )
    if value["dirty"] is not False:
        raise ValueError("target_source_provenance.dirty must be exactly false")
    return MappingProxyType(
        {
            "repository": "airylidar",
            "path": raw_path,
            "sha256": sha256,
            "commit": commit,
            "dirty": False,
        }
    )


def load_collection_config(path: str | Path) -> CollectionConfig:
    config_path = Path(path).expanduser()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load collection config {config_path}: {exc}") from exc
    root = _object(raw, "config")
    schema_version = root.get("schema_version")
    if schema_version not in {
        COLLECTION_CONFIG_SCHEMA_VERSION,
        LEGACY_COLLECTION_CONFIG_SCHEMA_VERSION,
    }:
        raise ValueError(
            "schema_version must be "
            f"{COLLECTION_CONFIG_SCHEMA_VERSION} or "
            f"{LEGACY_COLLECTION_CONFIG_SCHEMA_VERSION}"
        )

    joystick = _object(root.get("joystick_udp"), "joystick_udp")
    controllers = _object(root.get("controllers"), "controllers")
    serial = _object(root.get("stm32_serial"), "stm32_serial")
    camera = _object(root.get("camera_front"), "camera_front")
    dump_camera_value = root.get("camera_dump")
    dump_camera = (
        None
        if dump_camera_value is None
        else _object(dump_camera_value, "camera_dump")
    )
    if (
        schema_version == LEGACY_COLLECTION_CONFIG_SCHEMA_VERSION
        and dump_camera is not None
    ):
        raise ValueError(
            f"camera_dump requires schema_version {COLLECTION_CONFIG_SCHEMA_VERSION}"
        )
    preview_value = root.get("camera_preview_http")
    preview = (
        None
        if preview_value is None
        else _object(preview_value, "camera_preview_http")
    )
    state_udp_value = root.get("machine_state_udp")
    state_udp = (
        None
        if state_udp_value is None
        else _object(state_udp_value, "machine_state_udp")
    )
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

    def _load_camera_config(raw_camera: dict[str, Any], field: str) -> CameraConfig:
        return CameraConfig(
            device=_text(raw_camera.get("device"), f"{field}.device"),
            width=_integer(
                raw_camera.get("width"), f"{field}.width", minimum=16, maximum=8192
            ),
            height=_integer(
                raw_camera.get("height"), f"{field}.height", minimum=16, maximum=8192
            ),
            nominal_fps=_integer(
                raw_camera.get("fps"), f"{field}.fps", minimum=1, maximum=120
            ),
            jpeg_quality=_integer(
                raw_camera.get("jpeg_quality"),
                f"{field}.jpeg_quality",
                minimum=1,
                maximum=100,
            ),
        )

    camera_front = _load_camera_config(camera, "camera_front")
    camera_dump = (
        None
        if dump_camera is None
        else _load_camera_config(dump_camera, "camera_dump")
    )
    if (
        camera_dump is not None
        and os.path.realpath(camera_front.device)
        == os.path.realpath(camera_dump.device)
    ):
        raise ValueError("camera_front and camera_dump must use distinct devices")

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
        camera_front=camera_front,
        camera_dump=camera_dump,
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
        camera_preview=(
            None
            if preview is None
            else CameraPreviewConfig(
                bind_host=_text(
                    preview.get("bind_host"), "camera_preview_http.bind_host"
                ),
                port=_integer(
                    preview.get("port"),
                    "camera_preview_http.port",
                    minimum=1,
                    maximum=65535,
                ),
            )
        ),
        machine_state_udp=(
            None
            if state_udp is None
            else MachineStateUdpConfig(
                host=_text(state_udp.get("host"), "machine_state_udp.host"),
                port=_integer(
                    state_udp.get("port"),
                    "machine_state_udp.port",
                    minimum=1,
                    maximum=65535,
                ),
                machine_id=_text(
                    state_udp.get("machine_id"), "machine_state_udp.machine_id"
                ),
            )
        ),
    )

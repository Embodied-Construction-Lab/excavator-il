"""Strict configuration boundary for online ACT inference on Orin."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
from types import MappingProxyType

from .collector.config import SerialConfig


SCHEMA_VERSION = "excavator_act_runtime_config.v3"
LEGACY_SCHEMA_VERSION = "excavator_act_runtime_config.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BACKEND_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_COMMON_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_path",
        "deployment_manifest_path",
        "machine_profile_path",
        "checkpoint_model_sha256",
        "checkpoint_files_sha256",
        "log_root",
        "device",
        "dig_policy_backend",
        "stm32_serial",
        "timing",
    }
)
_ROOT_FIELDS_BY_SCHEMA = MappingProxyType(
    {
        LEGACY_SCHEMA_VERSION: _COMMON_ROOT_FIELDS | {"camera_front"},
        SCHEMA_VERSION: _COMMON_ROOT_FIELDS | {"cameras"},
    }
)
_CAMERA_ROLES = ("front", "dump")


@dataclass(frozen=True)
class ActCameraConfig:
    device: str
    width: int
    height: int
    nominal_fps: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass(frozen=True)
class ActRuntimeConfig:
    checkpoint_path: Path
    deployment_manifest_path: Path
    machine_profile_path: Path
    checkpoint_model_sha256: str
    checkpoint_files_sha256: dict[str, str]
    log_root: Path
    device: str
    dig_policy_backend: str
    serial: SerialConfig
    cameras: Mapping[str, ActCameraConfig]
    max_inference_state_age_ms: float
    state_silence_timeout_ms: float
    max_camera_age_ms: float
    max_inference_ms: float

    @property
    def camera(self) -> ActCameraConfig:
        """Return the authoritative front camera for legacy single-view consumers."""

        return self.cameras["front"]

    @property
    def camera_roles(self) -> tuple[str, ...]:
        return tuple(self.cameras)


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _integer(value: Any, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{field} must be an integer in [{low}, {high}]")
    return value


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _normalized_backend_identifier(value: Any, field: str) -> str:
    normalized = _text(value, field).strip().lower().replace("-", "_")
    if _BACKEND_IDENTIFIER.fullmatch(normalized) is None:
        raise ValueError(
            f"{field} must be a normalized lowercase backend identifier"
        )
    return normalized


def _camera_config(value: Any, field: str) -> ActCameraConfig:
    camera = _object(value, field)
    expected = {"device", "width", "height", "fps"}
    if set(camera) != expected:
        raise ValueError(f"{field} must contain exactly {sorted(expected)}")
    fps = _integer(camera.get("fps"), f"{field}.fps", 1, 120)
    if fps != 30:
        raise ValueError(f"{field}.fps must be 30")
    width = _integer(camera.get("width"), f"{field}.width", 16, 8192)
    height = _integer(camera.get("height"), f"{field}.height", 16, 8192)
    if (width, height) != (640, 480):
        raise ValueError(f"{field} must match the trained 640x480 RGB contract")
    return ActCameraConfig(
        device=_text(camera.get("device"), f"{field}.device"),
        width=width,
        height=height,
        nominal_fps=fps,
    )


def _camera_configs(root: dict[str, Any], schema_version: str) -> Mapping[str, ActCameraConfig]:
    if schema_version == LEGACY_SCHEMA_VERSION:
        return MappingProxyType({"front": _camera_config(root.get("camera_front"), "camera_front")})
    raw_cameras = _object(root.get("cameras"), "cameras")
    roles = tuple(raw_cameras)
    if roles not in (("front",), _CAMERA_ROLES):
        raise ValueError("cameras must contain ordered roles front or front,dump")
    cameras = {
        role: _camera_config(raw_cameras[role], f"cameras.{role}") for role in roles
    }
    devices = [camera.device for camera in cameras.values()]
    if len(devices) != len(set(devices)):
        raise ValueError("camera devices must be distinct")
    return MappingProxyType(cameras)


def load_act_runtime_config(path: str | Path) -> ActRuntimeConfig:
    config_path = Path(path).expanduser()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load ACT runtime config {config_path}: {exc}") from exc
    root = _object(raw, "config")
    schema_version = root.get("schema_version")
    expected_fields = _ROOT_FIELDS_BY_SCHEMA.get(schema_version)
    if expected_fields is None:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION} or {LEGACY_SCHEMA_VERSION}"
        )
    unexpected = set(root) - expected_fields
    if unexpected:
        raise ValueError(
            f"ACT runtime config has unexpected fields: {sorted(unexpected)}"
        )
    missing = expected_fields - {"dig_policy_backend"} - set(root)
    if missing:
        raise ValueError(f"ACT runtime config is missing fields: {sorted(missing)}")
    checkpoint_sha = _text(
        root.get("checkpoint_model_sha256"), "checkpoint_model_sha256"
    )
    if _SHA256.fullmatch(checkpoint_sha) is None:
        raise ValueError("checkpoint_model_sha256 must be a lowercase SHA-256")
    raw_file_hashes = _object(
        root.get("checkpoint_files_sha256"), "checkpoint_files_sha256"
    )
    if "model.safetensors" not in raw_file_hashes:
        raise ValueError("checkpoint_files_sha256 must include model.safetensors")
    file_hashes: dict[str, str] = {}
    for name, digest in raw_file_hashes.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("checkpoint_files_sha256 contains an invalid entry")
        file_hashes[name] = digest
    if file_hashes["model.safetensors"] != checkpoint_sha:
        raise ValueError("checkpoint model hashes disagree")
    device = _text(root.get("device"), "device")
    if device != "cuda":
        raise ValueError("online ACT runtime device must be cuda")
    dig_policy_backend = _normalized_backend_identifier(
        root.get("dig_policy_backend", "lerobot_act"), "dig_policy_backend"
    )
    serial = _object(root.get("stm32_serial"), "stm32_serial")
    cameras = _camera_configs(root, schema_version)
    timing = _object(root.get("timing"), "timing")
    baudrate = _integer(serial.get("baudrate"), "stm32_serial.baudrate", 1, 10_000_000)
    if baudrate != 460800:
        raise ValueError("stm32_serial.baudrate must be 460800")
    return ActRuntimeConfig(
        checkpoint_path=Path(
            _text(root.get("checkpoint_path"), "checkpoint_path")
        ).expanduser(),
        deployment_manifest_path=Path(
            _text(root.get("deployment_manifest_path"), "deployment_manifest_path")
        ).expanduser(),
        machine_profile_path=Path(
            _text(root.get("machine_profile_path"), "machine_profile_path")
        ).expanduser(),
        checkpoint_model_sha256=checkpoint_sha,
        checkpoint_files_sha256=MappingProxyType(file_hashes),
        log_root=Path(_text(root.get("log_root"), "log_root")).expanduser(),
        device=device,
        dig_policy_backend=dig_policy_backend,
        serial=SerialConfig(
            port=_text(serial.get("port"), "stm32_serial.port"),
            baudrate=baudrate,
        ),
        cameras=cameras,
        max_inference_state_age_ms=_positive_number(
            timing.get("max_inference_state_age_ms"),
            "timing.max_inference_state_age_ms",
        ),
        state_silence_timeout_ms=_positive_number(
            timing.get("state_silence_timeout_ms"),
            "timing.state_silence_timeout_ms",
        ),
        max_camera_age_ms=_positive_number(
            timing.get("max_camera_age_ms"), "timing.max_camera_age_ms"
        ),
        max_inference_ms=_positive_number(
            timing.get("max_inference_ms"), "timing.max_inference_ms"
        ),
    )

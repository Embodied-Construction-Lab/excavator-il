"""Strict parsing and process helpers for the resident fixed-cycle client."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from .resident_fixed_cycle_visualization import ResidentFixedCycleRemoteStatus

COMMISSIONING_AUTHORIZATION = "ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING"
TRAJECTORY_CONTROLLER_COMMISSIONING_AUTHORIZATION = (
    "ALLOW_CARTESIAN_P_MACHINE_MOTION"
)
CONTROL_SCHEMA_VERSION = "resident_fixed_cycle_control.v4"
_READY_PREFIX = "RESIDENT_FIXED_CYCLE_READY"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


@dataclass(frozen=True)
class ResidentOwnerReadiness:
    control_socket: PurePosixPath
    act_socket: PurePosixPath
    trajectory_controller_backend: str
    mission_id: str
    mission_sha256: str
    act_worker_required: bool
    act_worker_behavior_id: str | None
    act_worker_model_sha256: str | None


def parse_owner_readiness(line: str) -> ResidentOwnerReadiness:
    """Extract and strictly parse the Orin-owned Mission contract."""

    marker = _READY_PREFIX + " "
    marker_index = line.find(marker)
    if marker_index < 0 or line.find(marker, marker_index + 1) >= 0:
        raise RuntimeError("V3-A owner readiness prefix is invalid")
    log_prefix = line[:marker_index]
    if log_prefix and not log_prefix.endswith(": "):
        raise RuntimeError("V3-A owner readiness prefix is invalid")
    parts = line[marker_index:].split()
    if not parts or parts[0] != _READY_PREFIX:
        raise RuntimeError("V3-A owner readiness prefix is invalid")
    fields: dict[str, str] = {}
    for item in parts[1:]:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in fields:
            raise RuntimeError("V3-A owner readiness fields are invalid")
        fields[name] = value
    if set(fields) != {
        "control_socket",
        "act_socket",
        "trajectory_controller_backend",
        "mission_id",
        "mission_sha256",
        "act_worker_required",
        "act_worker_behavior_id",
        "act_worker_model_sha256",
    }:
        raise RuntimeError("V3-A owner readiness fields are invalid")
    backend = fields["trajectory_controller_backend"]
    if backend not in {"onnx_rl", "cartesian_p"}:
        raise RuntimeError("V3-A owner readiness controller backend is invalid")
    mission_id = fields["mission_id"]
    if _SAFE_ID.fullmatch(mission_id) is None:
        raise RuntimeError("V3-A owner readiness mission_id is invalid")
    mission_sha256 = fields["mission_sha256"]
    if re.fullmatch(r"[0-9a-f]{64}", mission_sha256) is None:
        raise RuntimeError("V3-A owner readiness mission_sha256 is invalid")
    worker_text = fields["act_worker_required"]
    if worker_text not in {"true", "false"}:
        raise RuntimeError("V3-A owner readiness act_worker_required is invalid")
    behavior_id = fields["act_worker_behavior_id"]
    model_sha256 = fields["act_worker_model_sha256"]
    if worker_text == "true":
        if behavior_id not in {"act_dig_lift", "act_dig_transport_dump"}:
            raise RuntimeError("V3-A owner readiness ACT behavior is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", model_sha256) is None:
            raise RuntimeError("V3-A owner readiness ACT model is invalid")
    elif behavior_id != "none" or model_sha256 != "none":
        raise RuntimeError("V3-A owner readiness unexpected ACT identity")
    try:
        control_socket = absolute_posix(fields["control_socket"], "control_socket")
        act_socket = absolute_posix(fields["act_socket"], "act_socket")
    except ValueError as exc:
        raise RuntimeError("V3-A owner readiness socket is invalid") from exc
    return ResidentOwnerReadiness(
        control_socket=control_socket,
        act_socket=act_socket,
        trajectory_controller_backend=backend,
        mission_id=mission_id,
        mission_sha256=mission_sha256,
        act_worker_required=worker_text == "true",
        act_worker_behavior_id=None if behavior_id == "none" else behavior_id,
        act_worker_model_sha256=(
            None if model_sha256 == "none" else model_sha256
        ),
    )


def parse_control_response(
    payload: str, command: str
) -> ResidentFixedCycleRemoteStatus:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("V3-A control returned invalid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "ok",
        "command",
        "status",
        "error",
    }:
        raise RuntimeError("V3-A control response fields are invalid")
    if (
        value["schema_version"] != CONTROL_SCHEMA_VERSION
        or value["command"] != command
        or value["ok"] is not True
        or value["error"] is not None
    ):
        raise RuntimeError(f"V3-A {command} command was rejected")
    return ResidentFixedCycleRemoteStatus.from_mapping(value["status"])


def text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be text")
    return value


def absolute_posix(value: Any, field: str) -> PurePosixPath:
    path = PurePosixPath(text(value, field))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be an absolute normalized path")
    return path


def relative_script(value: Any, field: str) -> str:
    result = text(value, field)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a normalized relative path")
    return result


def commissioning_authorization(value: Any) -> str:
    result = text(value, "commissioning_authorization", allow_empty=True)
    if result not in {"", COMMISSIONING_AUTHORIZATION}:
        raise ValueError(
            "commissioning_authorization must be empty or the exact V3-A token"
        )
    return result


def bounded_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{field} is outside its allowed range")
    return result


def bounded_integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} is outside its allowed range")
    return value


def remote_pid(line: str, name: str) -> int:
    match = re.fullmatch(re.escape(name) + r"=([1-9][0-9]*)", line)
    if match is None:
        raise RuntimeError(f"invalid {name} readiness line")
    return int(match.group(1))


def wait_process(process: Any, *, allow_nonzero: bool) -> None:
    process.wait(timeout_s=10.0)
    if process.returncode not in (0, None) and not allow_nonzero:
        raise RuntimeError(
            f"remote process exited with return code {process.returncode}"
        )

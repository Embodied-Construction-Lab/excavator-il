"""Strict parsing and process helpers for the resident fixed-cycle client."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any, Mapping

from .resident_fixed_cycle_visualization import ResidentFixedCycleRemoteStatus

COMMISSIONING_AUTHORIZATION = "ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING"
CONTROL_SCHEMA_VERSION = "resident_fixed_cycle_control.v3"


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
        raise RuntimeError(f"remote process exited with return code {process.returncode}")

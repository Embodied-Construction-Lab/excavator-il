"""Strict configuration contract for the PC-side resident fixed cycle."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ._resident_fixed_cycle_support import (
    TRAJECTORY_CONTROLLER_COMMISSIONING_AUTHORIZATION,
    absolute_posix,
    bounded_integer,
    bounded_number,
    commissioning_authorization,
    relative_script,
    text,
)


CONFIG_SCHEMA_VERSION = "excavator_resident_fixed_cycle_pc.v6"
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "guided_config",
        "fixed_cycle_plan",
        "runtime_root",
        "owner_script",
        "act_worker_script",
        "control_socket",
        "ready_timeout_s",
        "status_poll_ms",
        "act_max_steps",
        "commissioning_authorization",
        "expected_mission_id",
        "expected_mission_sha256",
        "expected_act_worker_required",
        "expected_act_behavior_id",
        "expected_act_model_sha256",
        "act_runtime_config",
        "act_checkpoint_host_path",
        "act_deployment_host_path",
        "dig_point_catalog",
        "edge_runtime_config",
        "trajectory_controller_backend",
        "trajectory_controller_commissioning_authorization",
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


@dataclass(frozen=True)
class ResidentFixedCyclePcConfig:
    guided_config: Path
    fixed_cycle_plan: PurePosixPath
    runtime_root: PurePosixPath
    owner_script: str
    act_worker_script: str
    control_socket: PurePosixPath
    ready_timeout_s: float
    status_poll_s: float
    act_max_steps: int
    commissioning_authorization: str
    expected_mission_id: str
    expected_mission_sha256: str
    expected_act_worker_required: bool
    expected_act_behavior_id: str | None
    expected_act_model_sha256: str | None
    act_runtime_config: PurePosixPath | None = None
    act_checkpoint_host_path: PurePosixPath | None = None
    act_deployment_host_path: PurePosixPath | None = None
    dig_point_catalog: PurePosixPath | None = None
    edge_runtime_config: PurePosixPath = PurePosixPath(
        "deploy/edge_runtime.resident.remote.json"
    )
    trajectory_controller_backend: str = "onnx_rl"
    trajectory_controller_commissioning_authorization: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "ResidentFixedCyclePcConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load V3-A PC config: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("V3-A PC config fields are invalid")
        schema_version = value.get("schema_version")
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported V3-A PC config schema")
        if set(value) != _CONFIG_FIELDS:
            raise ValueError("V3-A PC config fields are invalid")
        (
            expected_mission_id,
            expected_mission_sha256,
            expected_act_worker_required,
            expected_act_behavior_id,
            expected_act_model_sha256,
            act_runtime_config,
            act_checkpoint_host_path,
            act_deployment_host_path,
            dig_point_catalog,
        ) = _policy_config(value)
        (
            edge_runtime_config,
            trajectory_controller_backend,
            trajectory_authorization,
        ) = _controller_config(value)
        plan = absolute_posix(value["fixed_cycle_plan"], "fixed_cycle_plan")
        root = absolute_posix(value["runtime_root"], "runtime_root")
        socket_path = absolute_posix(value["control_socket"], "control_socket")
        if root not in socket_path.parents:
            raise ValueError("control_socket must be inside runtime_root")
        ready_timeout_s = bounded_number(
            value["ready_timeout_s"], "ready_timeout_s", 1.0, 300.0
        )
        poll_ms = bounded_number(
            value["status_poll_ms"], "status_poll_ms", 20.0, 1000.0
        )
        return cls(
            guided_config=(
                config_path.parent / text(value["guided_config"], "guided_config")
            ).resolve(),
            fixed_cycle_plan=plan,
            runtime_root=root,
            owner_script=relative_script(value["owner_script"], "owner_script"),
            act_worker_script=relative_script(
                value["act_worker_script"], "act_worker_script"
            ),
            control_socket=socket_path,
            ready_timeout_s=ready_timeout_s,
            status_poll_s=poll_ms / 1000.0,
            act_max_steps=bounded_integer(
                value["act_max_steps"], "act_max_steps", 1, 2000
            ),
            commissioning_authorization=commissioning_authorization(
                value["commissioning_authorization"]
            ),
            expected_mission_id=expected_mission_id,
            expected_mission_sha256=expected_mission_sha256,
            expected_act_worker_required=expected_act_worker_required,
            expected_act_behavior_id=expected_act_behavior_id,
            expected_act_model_sha256=expected_act_model_sha256,
            act_runtime_config=act_runtime_config,
            act_checkpoint_host_path=act_checkpoint_host_path,
            act_deployment_host_path=act_deployment_host_path,
            dig_point_catalog=dig_point_catalog,
            edge_runtime_config=edge_runtime_config,
            trajectory_controller_backend=trajectory_controller_backend,
            trajectory_controller_commissioning_authorization=(
                trajectory_authorization
            ),
        )


def _policy_config(
    value: Mapping[str, Any],
) -> tuple[
    str,
    str,
    bool,
    str | None,
    str | None,
    PurePosixPath | None,
    PurePosixPath | None,
    PurePosixPath | None,
    PurePosixPath | None,
]:
    expected_mission_id = text(
        value["expected_mission_id"], "expected_mission_id"
    )
    if _SAFE_ID.fullmatch(expected_mission_id) is None:
        raise ValueError("expected_mission_id must be a safe identifier")
    expected_mission_sha256 = _sha256(
        value["expected_mission_sha256"], "expected_mission_sha256"
    )
    expected_act_worker_required = value["expected_act_worker_required"]
    if not isinstance(expected_act_worker_required, bool):
        raise ValueError("expected_act_worker_required must be boolean")
    raw_assets = tuple(
        value[field]
        for field in (
            "act_runtime_config",
            "act_checkpoint_host_path",
            "act_deployment_host_path",
        )
    )
    if all(item is None for item in raw_assets):
        runtime = checkpoint = deployment = None
    elif all(item is not None for item in raw_assets):
        runtime = absolute_posix(value["act_runtime_config"], "act_runtime_config")
        checkpoint = absolute_posix(
            value["act_checkpoint_host_path"], "act_checkpoint_host_path"
        )
        deployment = absolute_posix(
            value["act_deployment_host_path"], "act_deployment_host_path"
        )
    else:
        raise ValueError("ACT worker asset overrides must be all null or all set")
    if not expected_act_worker_required and any(
        item is not None for item in raw_assets
    ):
        raise ValueError("ACT assets require expected_act_worker_required=true")
    if expected_act_worker_required and any(
        item is None for item in raw_assets
    ):
        raise ValueError("required ACT worker assets must all be set")
    behavior_id = value["expected_act_behavior_id"]
    model_sha256 = value["expected_act_model_sha256"]
    if expected_act_worker_required:
        behavior_id = text(behavior_id, "expected_act_behavior_id")
        if behavior_id not in {"act_dig_lift", "act_dig_transport_dump"}:
            raise ValueError("expected_act_behavior_id is unsupported")
        model_sha256 = _sha256(model_sha256, "expected_act_model_sha256")
    elif behavior_id is not None or model_sha256 is not None:
        raise ValueError("ACT worker identity requires an ACT worker")
    catalog = _relative_json_path(
        value["dig_point_catalog"], "dig_point_catalog"
    )
    return (
        expected_mission_id,
        expected_mission_sha256,
        expected_act_worker_required,
        behavior_id,
        model_sha256,
        runtime,
        checkpoint,
        deployment,
        catalog,
    )


def _controller_config(
    value: Mapping[str, Any],
) -> tuple[PurePosixPath, str, str]:
    edge_config = _relative_json_path(
        value["edge_runtime_config"], "edge_runtime_config"
    )
    backend = text(
        value["trajectory_controller_backend"],
        "trajectory_controller_backend",
    )
    if backend not in {"onnx_rl", "cartesian_p"}:
        raise ValueError(
            "trajectory_controller_backend must be onnx_rl or cartesian_p"
        )
    authorization = text(
        value["trajectory_controller_commissioning_authorization"],
        "trajectory_controller_commissioning_authorization",
        allow_empty=True,
    )
    if backend == "cartesian_p":
        if authorization != TRAJECTORY_CONTROLLER_COMMISSIONING_AUTHORIZATION:
            raise ValueError(
                "cartesian_p requires the exact trajectory controller "
                "commissioning authorization"
            )
    elif authorization:
        raise ValueError(
            "onnx_rl must not carry trajectory controller commissioning authorization"
        )
    return edge_config, backend, authorization


def _relative_json_path(value: Any, field: str) -> PurePosixPath:
    path = PurePosixPath(text(value, field))
    if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
        raise ValueError(f"{field} must be a normalized relative JSON path")
    return path


def _sha256(value: Any, field: str) -> str:
    result = text(value, field)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{field} must be lowercase hexadecimal")
    return result

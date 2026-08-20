"""Authoritative staged Mission contract for RL/ACT excavator handoffs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


HYBRID_MISSION_CONFIG_SCHEMA_VERSION = "excavator_hybrid_mission_config.v1"
REQUIRED_HYBRID_MOTION_AUTHORIZATION = "ALLOW_HYBRID_MACHINE_MOTION"


class HybridMissionSegment(str, Enum):
    RL_TO_DIG = "rl_to_dig"
    ACT_DIG = "act_dig"
    RL_TO_DUMP_AND_DUMP = "rl_to_dump_and_dump"
    RL_RETURN_TO_DIG = "rl_return_to_dig"


_SEGMENT_ORDER = tuple(HybridMissionSegment)


class HybridMissionOperations(Protocol):
    def run_rl_to_dig(self, target_id: str) -> None: ...

    def run_act_dig(self, max_steps: int) -> None: ...

    def run_rl_to_dump_and_dump(self) -> None: ...

    def run_rl_return_to_dig(self, target_id: str) -> None: ...

    def safe_stop(self) -> None: ...


@dataclass(frozen=True)
class HybridMissionConfig:
    guided_config: Path
    act_max_steps: int
    act_ready_timeout_s: int
    act_run_timeout_s: int
    act_remote_script: str
    rl_behavior_port: int

    @classmethod
    def load(cls, path: str | Path) -> "HybridMissionConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load hybrid Mission config {config_path}: {exc}"
            ) from exc
        root = _object(raw, "config")
        expected_root = {"schema_version", "guided_config", "act", "rl"}
        if set(root) != expected_root:
            raise ValueError(
                f"hybrid Mission config fields must be {sorted(expected_root)}"
            )
        if root.get("schema_version") != HYBRID_MISSION_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {HYBRID_MISSION_CONFIG_SCHEMA_VERSION}"
            )
        act = _object(root.get("act"), "act")
        rl = _object(root.get("rl"), "rl")
        expected_act = {
            "max_steps",
            "ready_timeout_s",
            "run_timeout_s",
            "remote_script",
        }
        if set(act) != expected_act:
            raise ValueError(f"act fields must be {sorted(expected_act)}")
        if set(rl) != {"behavior_port"}:
            raise ValueError("rl fields must be ['behavior_port']")
        max_steps = _integer(act.get("max_steps"), "act.max_steps", 1, 2000)
        ready_timeout_s = _integer(
            act.get("ready_timeout_s"), "act.ready_timeout_s", 1, 600
        )
        run_timeout_s = _integer(
            act.get("run_timeout_s"), "act.run_timeout_s", 1, 3600
        )
        minimum_run_timeout_s = ready_timeout_s + (max_steps + 9) // 10
        if run_timeout_s < minimum_run_timeout_s:
            raise ValueError(
                "act.run_timeout_s must cover ready_timeout_s plus the 10 Hz step budget"
            )
        remote_script = _safe_relative_path(
            act.get("remote_script"), "act.remote_script"
        )
        return cls(
            guided_config=(
                config_path.parent
                / _text(root.get("guided_config"), "guided_config")
            ).resolve(),
            act_max_steps=max_steps,
            act_ready_timeout_s=ready_timeout_s,
            act_run_timeout_s=run_timeout_s,
            act_remote_script=remote_script,
            rl_behavior_port=_integer(
                rl.get("behavior_port"), "rl.behavior_port", 1, 65535
            ),
        )


def remaining_hybrid_segments(
    start: HybridMissionSegment,
) -> tuple[HybridMissionSegment, ...]:
    segment = HybridMissionSegment(start)
    return _SEGMENT_ORDER[_SEGMENT_ORDER.index(segment) :]


def next_hybrid_segment(
    completed: HybridMissionSegment,
) -> HybridMissionSegment | None:
    remaining = remaining_hybrid_segments(completed)
    return remaining[1] if len(remaining) > 1 else None


def execute_hybrid_segment(
    operations: HybridMissionOperations,
    *,
    segment: HybridMissionSegment,
    dig_target_id: str,
    act_max_steps: int,
    motion_authorization: str | None,
) -> None:
    """Execute exactly one segment; each Adapter owns its zero/release handoff."""

    selected = HybridMissionSegment(segment)
    if not isinstance(dig_target_id, str) or not dig_target_id.strip():
        raise ValueError("dig_target_id must be non-empty")
    if (
        isinstance(act_max_steps, bool)
        or not isinstance(act_max_steps, int)
        or act_max_steps <= 0
    ):
        raise ValueError("act_max_steps must be a positive integer")
    if selected is HybridMissionSegment.ACT_DIG:
        if motion_authorization != REQUIRED_HYBRID_MOTION_AUTHORIZATION:
            raise ValueError("ACT segment requires exact hybrid motion authorization")
        operations.run_act_dig(act_max_steps)
    elif selected is HybridMissionSegment.RL_TO_DIG:
        operations.run_rl_to_dig(dig_target_id)
    elif selected is HybridMissionSegment.RL_TO_DUMP_AND_DUMP:
        operations.run_rl_to_dump_and_dump()
    else:
        operations.run_rl_return_to_dig(dig_target_id)


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


def _safe_relative_path(value: Any, field: str) -> str:
    text = _text(value, field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"{field} must be a safe relative path")
    return str(path)

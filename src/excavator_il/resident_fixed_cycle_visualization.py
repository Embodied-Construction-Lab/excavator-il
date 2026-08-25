"""Strict V3-A trajectory status and read-only RViz file bridge."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping


_STATUS_FIELDS = frozenset(
    {
        "run_id",
        "stage",
        "requested_cycles",
        "completed_cycles",
        "current_dig_point_id",
        "terminal",
        "outcome",
        "reason_code",
        "active_trajectory",
    }
)
_STAGES = frozenset(
    {
        "IDLE",
        "FOLLOW_DIG",
        "ACT_DIG",
        "FOLLOW_DUMP",
        "EXECUTE_DUMP",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    }
)
_TERMINAL_STAGES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


@dataclass(frozen=True)
class ResidentTrajectoryVisualization:
    frame_id: str
    target_id: str
    waypoints: tuple[tuple[float, float, float], ...]
    current_waypoint_index: int
    waypoint_tolerance_m: float

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ResidentTrajectoryVisualization":
        fields = {
            "frame_id",
            "target_id",
            "waypoints",
            "current_waypoint_index",
            "waypoint_tolerance_m",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("V3-A active trajectory fields are invalid")
        if value["frame_id"] != "machine_root_ros":
            raise ValueError("V3-A active trajectory frame is invalid")
        raw_waypoints = value["waypoints"]
        if not isinstance(raw_waypoints, list) or not 1 <= len(raw_waypoints) <= 12:
            raise ValueError("V3-A active trajectory waypoints are invalid")
        waypoints = tuple(
            _point(point, index=index)
            for index, point in enumerate(raw_waypoints)
        )
        current = value["current_waypoint_index"]
        if isinstance(current, bool) or not isinstance(current, int):
            raise ValueError("V3-A active trajectory index is invalid")
        if not 0 <= current < len(waypoints):
            raise ValueError("V3-A active trajectory index is out of range")
        return cls(
            frame_id="machine_root_ros",
            target_id=_identifier(value["target_id"], "trajectory.target_id"),
            waypoints=waypoints,
            current_waypoint_index=current,
            waypoint_tolerance_m=_positive(
                value["waypoint_tolerance_m"],
                "trajectory.waypoint_tolerance_m",
            ),
        )


@dataclass(frozen=True)
class ResidentFixedCycleRemoteStatus:
    run_id: str
    stage: str
    requested_cycles: int
    completed_cycles: int
    current_dig_point_id: str
    terminal: bool
    outcome: str
    reason_code: str
    active_trajectory: ResidentTrajectoryVisualization | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ResidentFixedCycleRemoteStatus":
        if not isinstance(value, Mapping) or set(value) != _STATUS_FIELDS:
            raise ValueError("V3-A status fields are invalid")
        stage = _text(value["stage"], "status.stage")
        if stage not in _STAGES:
            raise ValueError("V3-A status stage is invalid")
        requested = _count(value["requested_cycles"], "status.requested_cycles")
        completed = _count(value["completed_cycles"], "status.completed_cycles")
        if completed > requested:
            raise ValueError("completed_cycles cannot exceed requested_cycles")
        terminal = value["terminal"]
        if not isinstance(terminal, bool):
            raise ValueError("status.terminal must be boolean")
        if terminal != (stage in _TERMINAL_STAGES):
            raise ValueError("V3-A terminal flag and stage disagree")
        raw_trajectory = value["active_trajectory"]
        return cls(
            run_id=_optional_identifier(value["run_id"], "status.run_id"),
            stage=stage,
            requested_cycles=requested,
            completed_cycles=completed,
            current_dig_point_id=_optional_identifier(
                value["current_dig_point_id"],
                "status.current_dig_point_id",
            ),
            terminal=terminal,
            outcome=_text(value["outcome"], "status.outcome", allow_empty=True),
            reason_code=_text(
                value["reason_code"],
                "status.reason_code",
                allow_empty=True,
            ),
            active_trajectory=(
                None
                if raw_trajectory is None
                else ResidentTrajectoryVisualization.from_mapping(raw_trajectory)
            ),
        )


class V3aTrajectoryFile:
    """Atomically expose one Orin-authoritative path to a ROS marker node."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, trajectory: ResidentTrajectoryVisualization | None) -> None:
        if trajectory is None:
            self.path.unlink(missing_ok=True)
            return
        if not isinstance(trajectory, ResidentTrajectoryVisualization):
            raise ValueError("trajectory must be a V3-A visualization snapshot")
        document = {
            "schema_version": "trajectory_command.v1",
            "frame_id": trajectory.frame_id,
            "target_id": trajectory.target_id,
            "waypoints_base": [list(point) for point in trajectory.waypoints],
            "waypoint_count": len(trajectory.waypoints),
            "current_waypoint_index": trajectory.current_waypoint_index,
            "waypoint_tolerance_m": trajectory.waypoint_tolerance_m,
        }
        payload = json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def v3a_trajectory_path(log_dir: str | Path) -> Path:
    return Path(log_dir).expanduser().resolve() / "v3a_active_trajectory.json"


def _point(value: Any, *, index: int) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"V3-A active trajectory point {index} is invalid")
    point = tuple(float(axis) for axis in value)
    if not all(math.isfinite(axis) for axis in point):
        raise ValueError(f"V3-A active trajectory point {index} must be finite")
    return point


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is invalid")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 5.0:
        raise ValueError(f"{name} is invalid")
    return parsed


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9:
        raise ValueError(f"{name} is invalid")
    return value


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{name} must be a string")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe identifier")
    return value


def _optional_identifier(value: Any, name: str) -> str:
    if value == "":
        return ""
    return _identifier(value, name)

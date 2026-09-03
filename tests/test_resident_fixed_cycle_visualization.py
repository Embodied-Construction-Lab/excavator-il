import json

import pytest

from excavator_il.resident_fixed_cycle_visualization import (
    ResidentFixedCycleRemoteStatus,
    ResidentTrajectoryVisualization,
    V3aTrajectoryFile,
)


def _status(active_trajectory):
    return {
        "run_id": "run-v3a-001",
        "mission_id": "fixed_target_hybrid",
        "active_behavior_id": "onnx_rl_tracking",
        "stage": "go_current_dig",
        "requested_cycles": 2,
        "completed_cycles": 0,
        "current_dig_point_id": "dig_01",
        "dig_group_id": "all",
        "terminal": False,
        "outcome": "",
        "reason_code": "",
        "active_trajectory": active_trajectory,
    }


def _trajectory():
    return {
        "frame_id": "machine_root_ros",
        "target_id": "dig_01",
        "waypoints": [
            [0.8, 0.2, -0.1],
            [0.9, 0.1, -0.05],
            [1.0, 0.0, 0.0],
        ],
        "current_waypoint_index": 1,
        "waypoint_tolerance_m": 0.40,
    }


def test_remote_status_strictly_parses_orin_active_trajectory():
    status = ResidentFixedCycleRemoteStatus.from_mapping(_status(_trajectory()))

    assert status.mission_id == "fixed_target_hybrid"
    assert status.active_behavior_id == "onnx_rl_tracking"
    assert status.active_trajectory == ResidentTrajectoryVisualization(
        frame_id="machine_root_ros",
        target_id="dig_01",
        waypoints=(
            (0.8, 0.2, -0.1),
            (0.9, 0.1, -0.05),
            (1.0, 0.0, 0.0),
        ),
        current_waypoint_index=1,
        waypoint_tolerance_m=0.40,
    )


def test_remote_status_accepts_declarative_fixed_dig_behavior_and_custom_stage():
    payload = _status(_trajectory())
    payload["mission_id"] = "fixed_dig_hybrid"
    payload["active_behavior_id"] = "fixed_dig"
    payload["stage"] = "dig_with_fixed_action"

    status = ResidentFixedCycleRemoteStatus.from_mapping(payload)

    assert status.mission_id == "fixed_dig_hybrid"
    assert status.active_behavior_id == "fixed_dig"
    assert status.stage == "dig_with_fixed_action"


def test_remote_status_requires_active_behavior_only_while_nonterminal():
    active = _status(None)
    active["active_behavior_id"] = ""
    with pytest.raises(ValueError, match="active_behavior_id"):
        ResidentFixedCycleRemoteStatus.from_mapping(active)

    terminal = _status(None)
    terminal.update(
        {
            "stage": "COMPLETED",
            "active_behavior_id": "",
            "terminal": True,
            "outcome": "SUCCEEDED",
        }
    )
    parsed = ResidentFixedCycleRemoteStatus.from_mapping(terminal)
    assert parsed.active_behavior_id == ""


@pytest.mark.parametrize(
    "change",
    [
        {"frame_id": "map"},
        {"target_id": "bad id"},
        {"waypoints": [[0.0, 0.0, float("nan")]]},
        {"current_waypoint_index": 3},
        {"waypoint_tolerance_m": 0.0},
    ],
)
def test_remote_status_rejects_invalid_active_trajectory(change):
    trajectory = {**_trajectory(), **change}
    with pytest.raises(ValueError, match="trajectory"):
        ResidentFixedCycleRemoteStatus.from_mapping(_status(trajectory))


def test_trajectory_file_publishes_authoritative_bytes_and_clears(tmp_path):
    path = tmp_path / "v3a-active-trajectory.json"
    writer = V3aTrajectoryFile(path)
    trajectory = ResidentFixedCycleRemoteStatus.from_mapping(
        _status(_trajectory())
    ).active_trajectory

    writer.update(trajectory)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": "trajectory_command.v1",
        "frame_id": "machine_root_ros",
        "target_id": "dig_01",
        "waypoints_base": _trajectory()["waypoints"],
        "waypoint_count": 3,
        "current_waypoint_index": 1,
        "waypoint_tolerance_m": 0.40,
    }
    assert not list(tmp_path.glob("*.tmp"))

    writer.update(None)
    assert not path.exists()

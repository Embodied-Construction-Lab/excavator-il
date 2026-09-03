import json
import math
import os
import signal
import time
from pathlib import Path

import pytest

from excavator_il.machine_state_telemetry import (
    MachineStateTelemetryReader,
    MachineStateTelemetryService,
)


def _machine_state_packet() -> dict[str, object]:
    return {
        "type": "machine_state_v1",
        "schema_version": "1.0",
        "seq": 42,
        "stamp_ms": 1,
        "source": "orin",
        "machine_id": "scale_excavator_v1",
        "safety": {
            "estop": False,
            "stm32_alive": True,
            "sensor_valid": True,
            "control_enabled": True,
            "fault_flags": [],
        },
        "actuator_state": {
            "boom": {"position_m": 0.101, "velocity_mps": 0.0},
            "stick": {"position_m": 0.202, "velocity_mps": 0.0},
            "bucket": {"position_m": 0.303, "velocity_mps": 0.0},
            "swing": {"position_rad": math.pi, "velocity_rad_s": 0.0},
        },
        "joint_state": {
            "position_rad": {
                "boom": math.pi / 6,
                "arm": math.pi / 3,
                "bucket": math.pi / 2,
                "swing": math.pi,
            },
            "velocity_rad_s": {
                "boom": 0.0,
                "arm": 0.0,
                "bucket": 0.0,
                "swing": 0.0,
            },
        },
    }


def test_machine_state_reader_exposes_one_fresh_snapshot_for_web_ui(tmp_path):
    snapshot_path = tmp_path / "latest_state.json"
    snapshot_path.write_text(json.dumps(_machine_state_packet()), encoding="utf-8")

    telemetry = MachineStateTelemetryReader(snapshot_path).snapshot()

    assert telemetry["source"] == "machine_state_v1/udp:18081"
    assert telemetry["seq"] == 42
    assert telemetry["sensor_valid"] is True
    assert telemetry["control_enabled"] is True
    assert telemetry["joint_angles_deg"] == {
        "boom": pytest.approx(30.0),
        "arm": pytest.approx(60.0),
        "bucket": pytest.approx(90.0),
        "swing": pytest.approx(180.0),
    }
    assert telemetry["cylinders_mm"] == {
        "boom": pytest.approx(101.0),
        "stick": pytest.approx(202.0),
        "bucket": pytest.approx(303.0),
    }
    assert 0.0 <= telemetry["age_ms"] < 1_000.0


def test_machine_state_reader_rejects_a_stale_snapshot(tmp_path):
    snapshot_path = tmp_path / "latest_state.json"
    snapshot_path.write_text(json.dumps(_machine_state_packet()), encoding="utf-8")
    stale_ns = time.time_ns() - 2_000_000_000
    snapshot_path.touch()
    os.utime(snapshot_path, ns=(stale_ns, stale_ns))

    with pytest.raises(RuntimeError, match="stale"):
        MachineStateTelemetryReader(snapshot_path, max_age_ms=500).snapshot()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda packet: packet.update(type="other"), "schema"),
        (lambda packet: packet.update(seq=-1), "seq"),
        (lambda packet: packet.update(safety=[]), "safety"),
        (
            lambda packet: packet["safety"].update(fault_flags=[1]),
            "fault_flags",
        ),
        (
            lambda packet: packet["joint_state"]["position_rad"].update(
                boom=float("nan")
            ),
            "finite number",
        ),
        (
            lambda packet: packet["actuator_state"].update(boom=[]),
            "actuator_state.boom",
        ),
    ],
)
def test_machine_state_reader_rejects_invalid_contract_fields(
    tmp_path, mutate, message
):
    packet = _machine_state_packet()
    mutate(packet)
    snapshot_path = tmp_path / "latest_state.json"
    snapshot_path.write_text(json.dumps(packet), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        MachineStateTelemetryReader(snapshot_path).snapshot()


def test_machine_state_reader_rejects_missing_and_invalid_json(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="unavailable"):
        MachineStateTelemetryReader(missing).snapshot()

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        MachineStateTelemetryReader(invalid).snapshot()


def test_machine_state_reader_validates_age_configuration(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        MachineStateTelemetryReader(tmp_path / "state.json", max_age_ms=0)


def test_machine_state_service_owns_one_pc_bridge_process(tmp_path):
    processes = []

    class _GuidedConfig:
        rl_ros_setup = tmp_path / "ros/setup.bash"
        rl_workspace_setup = tmp_path / "AiryLidar/install/setup.bash"
        rl_airy_repo = tmp_path / "AiryLidar"
        log_dir = tmp_path / "logs"

    class _Process:
        running = True

        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.kwargs = kwargs
            self.stop_calls = []
            processes.append(self)

        def wait_for(self, predicate, timeout_s, *, after_index=-1):
            assert timeout_s == 10
            assert after_index == -1
            line = "pc state bridge started: state <- ('192.168.50.1', 18081)"
            assert predicate(line)
            return 0, line

        def stop(self, signum, *, timeout_s=5.0):
            self.stop_calls.append((signum, timeout_s))
            self.running = False

        @property
        def lines(self):
            return ()

    service = MachineStateTelemetryService(
        guided_config=_GuidedConfig(),
        line_process_factory=_Process,
    )

    service.start()

    assert len(processes) == 1
    command = processes[0].argv[-1]
    assert "runtime_bridge/apps/pc_runtime_bridge.py" in command
    assert "--publish-joint-states" in command
    assert "--write-every 1" in command
    assert service.snapshot_path == (
        Path(_GuidedConfig.rl_airy_repo)
        / "runtime_bridge/exports/latest_state.json"
    )
    service.snapshot_path.parent.mkdir(parents=True)
    service.snapshot_path.write_text(
        json.dumps(_machine_state_packet()),
        encoding="utf-8",
    )
    assert service.snapshot()["seq"] == 42

    with pytest.raises(RuntimeError, match="already active"):
        service.start()

    service.close()
    assert processes[0].stop_calls == [(signal.SIGINT, 10.0)]


def test_machine_state_service_cleans_failed_bridge_start(tmp_path):
    processes = []

    class _GuidedConfig:
        rl_ros_setup = tmp_path / "ros/setup.bash"
        rl_workspace_setup = tmp_path / "AiryLidar/install/setup.bash"
        rl_airy_repo = tmp_path / "AiryLidar"
        log_dir = tmp_path / "logs"

    class _Process:
        running = True

        def __init__(self, _argv, **_kwargs):
            self.stop_calls = []
            processes.append(self)

        def wait_for(self, _predicate, _timeout_s, *, after_index=-1):
            assert after_index == -1
            raise RuntimeError("bridge failed")

        def stop(self, signum, *, timeout_s=5.0):
            self.stop_calls.append((signum, timeout_s))

    service = MachineStateTelemetryService(
        guided_config=_GuidedConfig(),
        line_process_factory=_Process,
    )

    with pytest.raises(RuntimeError, match="bridge failed"):
        service.start()

    assert processes[0].stop_calls == [(signal.SIGINT, 10.0)]
    with pytest.raises(RuntimeError, match="not active"):
        service.snapshot()

import json
import threading
import time
from pathlib import Path

import pytest

import excavator_il.resident_fixed_cycle_system as resident_module
from excavator_il.hybrid_mission import REQUIRED_HYBRID_MOTION_AUTHORIZATION
from excavator_il.guided_episode import GuidedEpisodeConfig
from excavator_il.resident_fixed_cycle_system import (
    ResidentFixedCyclePcConfig,
    ResidentFixedCycleProcesses,
    ResidentFixedCycleRemoteStatus,
    ResidentFixedCycleSupervisor,
    SshResidentFixedCycleOperations,
)


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    document = {
        "schema_version": "excavator_resident_fixed_cycle_pc.v2",
        "guided_config": "guided.json",
        "fixed_cycle_plan": "/home/jetson16/workspace_excavator/"
        "excavator-orin-runtime/deploy/v3a/fixed_cycle.field.json",
        "runtime_root": "/home/jetson16/.local/run/excavator-resident-v3a",
        "owner_script": "scripts/run_resident_mission_runtime.sh",
        "act_worker_script": "scripts/run_act_resident.sh",
        "control_socket": "/home/jetson16/.local/run/"
        "excavator-resident-v3a/fixed-cycle.sock",
        "ready_timeout_s": 120,
        "status_poll_ms": 100,
        "act_max_steps": 130,
        "commissioning_authorization": "",
    }
    document.update(overrides)
    path = tmp_path / "resident-fixed-cycle.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_fixed_cycle_pc_config_is_strict_and_resolves_guided_config(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))

    assert config.guided_config == (tmp_path / "guided.json").resolve()
    assert config.fixed_cycle_plan.is_absolute()
    assert config.status_poll_s == pytest.approx(0.1)
    assert config.act_max_steps == 130
    assert config.commissioning_authorization == ""

    invalid = _write_config(tmp_path, unexpected=True)
    with pytest.raises(ValueError, match="fields"):
        ResidentFixedCyclePcConfig.load(invalid)

    commissioned = ResidentFixedCyclePcConfig.load(
        _write_config(
            tmp_path,
            commissioning_authorization="ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING",
        )
    )
    assert commissioned.commissioning_authorization.startswith("ALLOW_V3A")

    with pytest.raises(ValueError, match="commissioning_authorization"):
        ResidentFixedCyclePcConfig.load(
            _write_config(tmp_path, commissioning_authorization="wrong")
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": "wrong"}, "schema"),
        ({"control_socket": "/tmp/outside.sock"}, "runtime_root"),
        ({"fixed_cycle_plan": "relative.json"}, "absolute"),
        ({"owner_script": "../owner.sh"}, "normalized relative"),
        ({"ready_timeout_s": True}, "numeric"),
        ({"status_poll_ms": 1}, "allowed range"),
        ({"act_max_steps": 0}, "allowed range"),
    ],
)
def test_fixed_cycle_pc_config_rejects_invalid_boundaries(
    tmp_path, overrides, message
):
    with pytest.raises(ValueError, match=message):
        ResidentFixedCyclePcConfig.load(_write_config(tmp_path, **overrides))


def test_fixed_cycle_pc_config_wraps_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load"):
        ResidentFixedCyclePcConfig.load(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"unexpected": 1}, "fields"),
        ({"stage": "UNKNOWN"}, "stage"),
        ({"requested_cycles": 0, "completed_cycles": 1}, "cannot exceed"),
        ({"terminal": 1}, "boolean"),
        ({"stage": "COMPLETED", "terminal": False}, "disagree"),
        ({"run_id": "bad id"}, "identifier"),
    ],
)
def test_remote_status_rejects_invalid_wire_values(change, message):
    value = _status("FOLLOW_DIG")
    value.update(change)
    with pytest.raises(ValueError, match=message):
        ResidentFixedCycleRemoteStatus.from_mapping(value)


def test_ssh_operations_start_once_and_use_only_local_cycle_control(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    calls = []

    class Processes:
        def start(self):
            calls.append(("processes", "start"))

        def stop(self, *, terminal_disarmed=False):
            calls.append(("processes", "stop", terminal_disarmed))

        def require_running(self):
            calls.append(("processes", "running"))

    responses = {
        "start": _status("FOLLOW_DIG"),
        "heartbeat": _status("ACT_DIG"),
        "cancel": _status("CANCELLED", terminal=True, outcome="CANCELLED"),
    }

    class Host:
        def run(self, command, *, accepted_returncodes=(0,)):
            calls.append(("ssh", command, accepted_returncodes))
            selected = next(name for name in responses if command.endswith(name))
            return json.dumps({"schema_version": "resident_fixed_cycle_control.v1",
                               "ok": True, "command": selected,
                               "status": responses[selected], "error": None})

    operations = SshResidentFixedCycleOperations(
        config,
        guided_config=_guided(),
        processes=Processes(),
        remote_host=Host(),
    )

    start = operations.start(
        run_id="run-001", requested_cycles=3, first_dig_point_id="dig_02"
    )
    status = operations.status()
    cancelled = operations.cancel()

    assert start.stage == "FOLLOW_DIG"
    assert status.stage == "ACT_DIG"
    assert cancelled.terminal is True
    ssh_commands = [entry[1] for entry in calls if entry[0] == "ssh"]
    assert any("resident_fixed_cycle_control" in command for command in ssh_commands)
    assert any("--run-id run-001 --cycles 3 --first-dig-point-id dig_02 start" in command
               for command in ssh_commands)
    assert not any("18083" in command or "Plan" in command for command in ssh_commands)


def test_ssh_operations_accept_real_guided_posix_path_fields(tmp_path):
    """Exercise the production config types at the final shlex boundary."""

    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    repo_root = Path(__file__).parents[1]
    guided = GuidedEpisodeConfig.load(repo_root / "config/guided_episode.pc.json")
    commands = []

    class Processes:
        def start(self):
            return None

    class Host:
        def run(self, command, *, accepted_returncodes=(0,)):
            commands.append(command)
            return json.dumps(
                {
                    "schema_version": "resident_fixed_cycle_control.v1",
                    "ok": True,
                    "command": "start",
                    "status": _status("FOLLOW_DIG"),
                    "error": None,
                }
            )

    operations = SshResidentFixedCycleOperations(
        config,
        guided_config=guided,
        processes=Processes(),
        remote_host=Host(),
    )

    status = operations.start(
        run_id="run-real-config",
        requested_cycles=1,
        first_dig_point_id="dig_01",
    )

    assert status.stage == "FOLLOW_DIG"
    assert commands and str(guided.rl_orin_python) in commands[0]


def test_commissioning_owner_command_forwards_flat_exact_authorization(tmp_path):
    config = ResidentFixedCyclePcConfig.load(
        _write_config(
            tmp_path,
            commissioning_authorization="ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING",
        )
    )
    commands = []

    class Host:
        def argv(self, command):
            commands.append(command)
            return ["ssh", "orin", command]

    class Process:
        returncode = None

        def wait_for(self, predicate, _timeout):
            for line in (
                "RESIDENT_OWNER_PID=123",
                "RESIDENT_FIXED_CYCLE_READY control_socket=/tmp/fixed.sock",
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            ):
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=lambda *_args, **_kwargs: Process(),
    )

    processes._start_owner()

    assert len(commands) == 1
    assert (
        "--commissioning-authorization "
        "ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING"
    ) in commands[0]


def test_remote_process_lifecycle_starts_once_and_releases_owned_devices(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    spawned = []
    stopped = []

    class Host:
        def argv(self, command):
            return ["ssh", "orin", command]

        def stop_owned_process(self, **kwargs):
            stopped.append(kwargs)

    class Process:
        def __init__(self, lines):
            self.lines = lines
            self.returncode = None

        def wait_for(self, predicate, _timeout):
            for line in self.lines:
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

        def wait(self, *, timeout_s):
            assert timeout_s == 10.0
            self.returncode = 0

    def factory(argv, **kwargs):
        index = len(spawned)
        lines = (
            [
                "RESIDENT_OWNER_PID=321",
                "RESIDENT_FIXED_CYCLE_READY control_socket=/tmp/fixed.sock",
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            ]
            if index == 0
            else ["RESIDENT_ACT_PID=654", "ACT resident worker ready: connected"]
        )
        spawned.append((argv, kwargs))
        return Process(lines)

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=factory,
        timestamp="20260825_120000",
    )

    processes.start()
    processes.start()
    processes.stop()

    assert len(spawned) == 2
    assert spawned[0][1]["prefix"] == "v3a-owner"
    assert spawned[1][1]["prefix"] == "v3a-act"
    assert len(stopped) == 2
    assert stopped[0]["serial_path"] == Path("/dev/ttyTHS1")
    assert str(stopped[1]["serial_path"]) == "/dev/video0"


def test_process_start_failure_cleans_owner_and_reports_cleanup_error(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    messages = []

    class Host:
        def argv(self, command):
            return [command]

        def stop_owned_process(self, **_kwargs):
            raise OSError("cleanup failed")

    class Owner:
        returncode = None

        def wait_for(self, predicate, _timeout):
            for line in (
                "RESIDENT_OWNER_PID=321",
                "RESIDENT_FIXED_CYCLE_READY control_socket=/tmp/fixed.sock",
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            ):
                if predicate(line):
                    return 0, line
            raise AssertionError

    calls = 0

    def factory(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Owner()
        raise RuntimeError("ACT failed to launch")

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=factory,
        output=messages.append,
    )

    with pytest.raises(RuntimeError, match="ACT failed"):
        processes.start()

    assert any("startup cleanup failed" in message for message in messages)


def test_process_require_running_and_wait_fail_closed(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    processes = ResidentFixedCycleProcesses(config, guided_config=_guided())
    with pytest.raises(RuntimeError, match="not started"):
        processes.require_running()

    class Failed:
        returncode = 3

        def wait(self, *, timeout_s):
            assert timeout_s == 10.0

    with pytest.raises(RuntimeError, match="return code 3"):
        resident_module._wait_process(Failed(), allow_nonzero=False)
    resident_module._wait_process(Failed(), allow_nonzero=True)


def test_supervisor_maps_complete_local_cycle_to_existing_ui_snapshot():
    operations = _ScriptedOperations(
        [
            _remote_status("FOLLOW_DIG", completed=0),
            _remote_status("ACT_DIG", completed=0),
            _remote_status("FOLLOW_DUMP", completed=0),
            _remote_status("EXECUTE_DUMP", completed=0),
            _remote_status(
                "COMPLETED", completed=2, terminal=True, outcome="SUCCEEDED"
            ),
        ]
    )
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
    )

    supervisor.start(
        "dig_02",
        automatic=True,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        cycle_count=2,
    )
    deadline = time.monotonic() + 1.0
    while supervisor.snapshot().stage != "completed" and time.monotonic() < deadline:
        time.sleep(0.005)

    snapshot = supervisor.snapshot()
    assert snapshot.stage == "completed"
    assert snapshot.run_completed_cycles == 2
    assert snapshot.requested_cycles == 2
    assert snapshot.dig_target_id == "dig_02"
    assert snapshot.can_stop is False
    assert operations.started == [(snapshot.run_id, 2, "dig_02")]
    assert operations.released == [True]


def test_supervisor_records_v3a_status_and_finalizes_experiment_evidence():
    operations = _ScriptedOperations(
        [
            _remote_status("FOLLOW_DIG", completed=0),
            _remote_status("ACT_DIG", completed=0),
            _remote_status("COMPLETED", completed=1, terminal=True, outcome="SUCCEEDED"),
        ]
    )

    class Evidence:
        run_id = "v3a_evidence_001"

        def __init__(self):
            self.events = []
            self.finalizations = []

        def append_event(self, event_type, payload):
            self.events.append((event_type, dict(payload)))

        def finalize(self, status, *, metrics, summary):
            self.finalizations.append((status, dict(metrics), summary))

    evidence = Evidence()
    requests = []
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
        config_path=Path("/configs/resident-fixed.json"),
        evidence_run_factory=lambda request: requests.append(request) or evidence,
    )

    supervisor.start(
        "dig_01",
        automatic=True,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        cycle_count=1,
    )
    deadline = time.monotonic() + 1.0
    while supervisor.snapshot().stage != "completed" and time.monotonic() < deadline:
        time.sleep(0.005)

    assert supervisor.snapshot().run_id == "v3a_evidence_001"
    assert requests[0].config_path == Path("/configs/resident-fixed.json")
    assert any(event == "resident_fixed_cycle_status" for event, _ in evidence.events)
    assert evidence.finalizations[0][0] == "success"
    assert evidence.finalizations[0][1]["completed_cycles"] == 1


def test_initial_v3a_evidence_failure_prevents_remote_motion_start():
    operations = _ScriptedOperations([])

    class Evidence:
        run_id = "v3a_evidence_failed"

        def __init__(self):
            self.finalizations = []

        def append_event(self, _event_type, _payload):
            raise OSError("evidence disk unavailable")

        def finalize(self, status, *, metrics, summary):
            self.finalizations.append((status, dict(metrics), summary))

    evidence = Evidence()
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
        config_path=Path("/configs/resident-fixed.json"),
        evidence_run_factory=lambda _request: evidence,
    )

    with pytest.raises(RuntimeError, match="initial V3-A evidence"):
        supervisor.start(
            "dig_01",
            automatic=True,
            motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
            cycle_count=1,
        )

    assert operations.started == []
    assert evidence.finalizations[0][0] == "failure"


def test_supervisor_rejects_segmented_v3_and_cancel_is_immediate():
    operations = _BlockingOperations()
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
    )
    with pytest.raises(ValueError, match="automatic"):
        supervisor.start(
            "dig_01", automatic=False, motion_authorization=None, cycle_count=1
        )

    supervisor.start(
        "dig_01",
        automatic=True,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        cycle_count=1,
    )
    assert operations.entered.wait(1.0)
    supervisor.stop()
    operations.cancelled.wait(1.0)
    deadline = time.monotonic() + 1.0
    while supervisor.snapshot().stage != "cancelled" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert supervisor.snapshot().stage == "cancelled"

    with pytest.raises(RuntimeError, match="not support segmented"):
        supervisor.advance(motion_authorization=None)


def test_cancel_transport_failure_stops_heartbeats_and_forces_owner_release():
    operations = _CancelFailureOperations()
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
    )
    supervisor.start(
        "dig_01",
        automatic=True,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        cycle_count=1,
    )
    assert operations.entered.wait(1.0)

    supervisor.stop()
    assert operations.released.wait(1.0)

    snapshot = supervisor.snapshot()
    assert snapshot.stage == "failed"
    assert "cancel was not acknowledged" in snapshot.error
    assert operations.status_calls == 0
    assert operations.release_terminal_disarmed is False


def test_supervisor_maps_local_dump_and_return_phases_to_existing_ui_contract():
    supervisor = ResidentFixedCycleSupervisor(
        operations=_BlockingOperations(),
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
    )

    supervisor._apply_status(_remote_status("FOLLOW_DUMP", completed=0))
    assert supervisor.snapshot().stage == "running_rl_to_dump_and_dump"

    supervisor._apply_status(_remote_status("FOLLOW_DIG", completed=1))
    assert supervisor.snapshot().stage == "running_rl_return_to_dig"

    supervisor.append_external_log("[resident-owner] hardware ready")
    assert supervisor.snapshot().logs[-1] == "[resident-owner] hardware ready"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "invalid JSON"),
        (json.dumps({"schema_version": "wrong"}), "fields"),
        (
            json.dumps(
                {
                    "schema_version": "resident_fixed_cycle_control.v1",
                    "ok": False,
                    "command": "status",
                    "status": None,
                    "error": {"code": "failed", "message": "failed"},
                }
            ),
            "rejected",
        ),
    ],
)
def test_control_response_parser_rejects_untrusted_payload(payload, message):
    with pytest.raises(RuntimeError, match=message):
        resident_module._parse_control_response(payload, "status")


def test_readiness_pid_parser_rejects_malformed_line():
    with pytest.raises(RuntimeError, match="invalid RESIDENT_OWNER_PID"):
        resident_module._remote_pid("RESIDENT_OWNER_PID=0", "RESIDENT_OWNER_PID")


def _guided():
    class Guided:
        orin_ssh_host = "jetson16@192.168.50.2"
        rl_orin_repo = Path("/home/jetson16/workspace_excavator/excavator-orin-runtime")
        orin_repo = Path("/home/jetson16/workspace_excavator/excavator-il")
        rl_orin_python = "/opt/orin/bin/python"
        rl_pc_host = "192.168.50.1"
        rl_serial_port = Path("/dev/ttyTHS1")
        rl_serial_release_timeout_s = 8.0
        log_dir = Path("/tmp")

    return Guided()


def _status(stage, *, terminal=False, outcome=""):
    return {
        "run_id": "run-001",
        "stage": stage,
        "requested_cycles": 3,
        "completed_cycles": 0,
        "current_dig_point_id": "dig_02",
        "terminal": terminal,
        "outcome": outcome,
        "reason_code": "",
    }


def _remote_status(stage, *, completed, terminal=False, outcome=""):
    return ResidentFixedCycleRemoteStatus(
        run_id="run-local",
        stage=stage,
        requested_cycles=2,
        completed_cycles=completed,
        current_dig_point_id="dig_02",
        terminal=terminal,
        outcome=outcome,
        reason_code="",
    )


class _ScriptedOperations:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.started = []
        self.released = []

    def start(self, *, run_id, requested_cycles, first_dig_point_id):
        self.started.append((run_id, requested_cycles, first_dig_point_id))
        return self.statuses.pop(0)

    def status(self):
        return self.statuses.pop(0)

    def cancel(self):
        raise AssertionError("cancel is not expected")

    def release(self, *, terminal_disarmed):
        self.released.append(terminal_disarmed)


class _BlockingOperations:
    def __init__(self):
        self.entered = threading.Event()
        self.cancelled = threading.Event()

    def start(self, **_kwargs):
        self.entered.set()
        return _remote_status("FOLLOW_DIG", completed=0)

    def status(self):
        if self.cancelled.is_set():
            return _remote_status(
                "CANCELLED", completed=0, terminal=True, outcome="CANCELLED"
            )
        return _remote_status("FOLLOW_DIG", completed=0)

    def cancel(self):
        self.cancelled.set()
        return _remote_status(
            "CANCELLED", completed=0, terminal=True, outcome="CANCELLED"
        )

    def release(self, *, terminal_disarmed):
        assert terminal_disarmed is True


class _CancelFailureOperations:
    def __init__(self):
        self.entered = threading.Event()
        self.released = threading.Event()
        self.status_calls = 0
        self.release_terminal_disarmed = None

    def start(self, **_kwargs):
        self.entered.set()
        return _remote_status("FOLLOW_DIG", completed=0)

    def status(self):
        self.status_calls += 1
        return _remote_status("FOLLOW_DIG", completed=0)

    def cancel(self):
        raise OSError("control socket unavailable")

    def release(self, *, terminal_disarmed):
        self.release_terminal_disarmed = terminal_disarmed
        self.released.set()

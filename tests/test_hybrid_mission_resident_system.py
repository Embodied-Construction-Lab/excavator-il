from pathlib import PurePosixPath

import pytest

from excavator_il.hybrid_mission import HybridMissionConfig, ResidentMissionConfig
from excavator_il.hybrid_mission_resident import (
    ResidentControlStatus,
    ResidentHybridMissionOperations,
    ResidentPolicyBinding,
)
from excavator_il.hybrid_mission_resident_system import (
    ResidentProcessAwareControlAdapter,
    ResidentMissionProcesses,
    SystemResidentHybridMissionOperations,
    _wait_for_local_process,
)
from excavator_il.resident_mission_lease import ResidentMissionLeaseHeartbeat
from excavator_il.resident_prepared_follow import SystemPreparedDumpAdapter


class _Guided:
    orin_ssh_host = "jetson16@192.168.50.2"
    rl_pc_host = "192.168.50.1"
    orin_repo = PurePosixPath("/srv/excavator-il")
    rl_orin_repo = PurePosixPath("/srv/excavator-orin-runtime")
    rl_orin_python = PurePosixPath("/opt/conda/bin/python")
    rl_serial_port = PurePosixPath("/dev/ttyTHS1")
    rl_serial_release_timeout_s = 8

    def __init__(self, log_dir):
        self.log_dir = log_dir


def _config(tmp_path):
    return HybridMissionConfig(
        guided_config=tmp_path / "guided.json",
        act_max_steps=130,
        act_ready_timeout_s=60,
        act_run_timeout_s=90,
        act_remote_script="scripts/run_act_motion.sh",
        rl_behavior_port=18083,
        runtime_backend="resident",
        resident=ResidentMissionConfig(
            owner_script="scripts/run_resident_mission_runtime.sh",
            act_worker_script="scripts/run_act_resident.sh",
            runtime_root=PurePosixPath(
                "/home/jetson16/.local/run/excavator-resident"
            ),
            ready_timeout_s=120,
            handoff_timeout_s=3,
            poll_interval_ms=50,
            prepared_dump_lead_steps=20,
            prepared_ready_grace_ms=300,
            prepared_start_tolerance_m=0.15,
        ),
    )


class _RemoteHost:
    def __init__(self):
        self.stop_calls = []

    def argv(self, command):
        return ["ssh", command]

    def stop_owned_process(self, **kwargs):
        self.stop_calls.append(kwargs)


class _ResidentProcess:
    returncode = None

    def __init__(self, argv, *, prefix, **_kwargs):
        self.argv = argv
        self.prefix = prefix
        self.stop_calls = []

    def wait_for(self, predicate, _timeout_s, *, after_index=-1):
        del after_index
        lines = {
            "resident-owner": (
                "RESIDENT_OWNER_PID=4100",
                "REMOTE EDGE CONTROL ARMED IDLE: behavior RPC 0.0.0.0:18083 from 192.168.50.1; actions use the resident STM32 command sink",
                "sent seq=0 stm32_t=42 sensor_valid=True boom=0.1",
            ),
            "resident-act": (
                "RESIDENT_ACT_PID=4200",
                "2026-08-21 14:03:07,315 INFO "
                "excavator_il.resident_act_runtime: "
                "ACT resident worker ready: owner connected",
            ),
        }[self.prefix]
        line = next(value for value in lines if predicate(value))
        return lines.index(line), line

    def wait(self, timeout_s=5.0):
        del timeout_s
        if self.returncode is None:
            self.returncode = 0

    def stop(self, signum, *, timeout_s=5.0):
        self.stop_calls.append((signum, timeout_s))


def test_resident_processes_start_once_and_keep_act_away_from_serial(tmp_path):
    processes = []

    def factory(*args, **kwargs):
        process = _ResidentProcess(*args, **kwargs)
        processes.append(process)
        return process

    manager = ResidentMissionProcesses(
        _config(tmp_path),
        guided_config=_Guided(tmp_path),
        remote_host=_RemoteHost(),
        line_process_factory=factory,
        output=lambda _message: None,
    )

    manager.start()
    manager.start()

    assert manager.started
    assert [process.prefix for process in processes] == [
        "resident-owner",
        "resident-act",
    ]
    owner_command = processes[0].argv[-1]
    act_command = processes[1].argv[-1]
    assert "run_resident_mission_runtime.sh" in owner_command
    assert "RESIDENT_PYTHON=/opt/conda/bin/python" in owner_command
    assert "--authorization ALLOW_HYBRID_MACHINE_MOTION" in owner_command
    assert "run_act_resident.sh" in act_command
    assert "--authorization ALLOW_HYBRID_MACHINE_MOTION" in act_command
    assert "/dev/ttyTHS1" not in act_command


def test_resident_processes_can_expose_owner_before_act_worker_ready(tmp_path):
    processes = []

    def factory(*args, **kwargs):
        process = _ResidentProcess(*args, **kwargs)
        processes.append(process)
        return process

    manager = ResidentMissionProcesses(
        _config(tmp_path),
        guided_config=_Guided(tmp_path),
        remote_host=_RemoteHost(),
        line_process_factory=factory,
        output=lambda _message: None,
    )

    manager.start_owner()

    assert manager.owner_started
    assert not manager.started
    assert [process.prefix for process in processes] == ["resident-owner"]

    manager.start_act_worker()

    assert manager.started
    assert [process.prefix for process in processes] == [
        "resident-owner",
        "resident-act",
    ]


def test_resident_processes_stop_worker_before_owner_and_release_devices(tmp_path):
    remote = _RemoteHost()
    manager = ResidentMissionProcesses(
        _config(tmp_path),
        guided_config=_Guided(tmp_path),
        remote_host=remote,
        line_process_factory=_ResidentProcess,
        output=lambda _message: None,
    )
    manager.start()

    manager.stop()

    assert not manager.started
    assert len(remote.stop_calls) == 2
    worker, owner = remote.stop_calls
    assert worker["pid"] == 4200
    assert worker["serial_path"] == PurePosixPath("/dev/video0")
    assert owner["pid"] == 4100
    assert owner["serial_path"] == PurePosixPath("/dev/ttyTHS1")
    assert owner["cleanup_paths"] == (
        PurePosixPath(
            "/home/jetson16/.local/run/excavator-resident/control.sock"
        ),
        PurePosixPath("/home/jetson16/.local/run/excavator-resident/act.sock"),
    )


def test_stop_accepts_owner_driven_act_fail_closed_exit_after_device_release(
    tmp_path,
):
    created = []

    def factory(*args, **kwargs):
        process = _ResidentProcess(*args, **kwargs)
        created.append(process)
        return process

    manager = ResidentMissionProcesses(
        _config(tmp_path),
        guided_config=_Guided(tmp_path),
        remote_host=_RemoteHost(),
        line_process_factory=factory,
        output=lambda _message: None,
    )
    manager.start()
    created[0].returncode = 0
    created[1].returncode = 17

    manager.stop()

    assert not manager.owner_started
    assert not manager.started


def test_resident_process_health_reports_an_exited_act_worker_immediately(tmp_path):
    created = []

    def factory(*args, **kwargs):
        process = _ResidentProcess(*args, **kwargs)
        created.append(process)
        return process

    manager = ResidentMissionProcesses(
        _config(tmp_path),
        guided_config=_Guided(tmp_path),
        remote_host=_RemoteHost(),
        line_process_factory=factory,
        output=lambda _message: None,
    )
    manager.start()
    created[1].returncode = 17

    with pytest.raises(
        RuntimeError, match="resident ACT worker exited with return code 17"
    ):
        manager.require_running()


def test_wait_for_local_process_rejects_nonzero_remote_exit():
    class FailedProcess:
        returncode = None

        def wait(self, timeout_s=2.0):
            del timeout_s
            self.returncode = 23

        def stop(self, *_args, **_kwargs):
            raise AssertionError("completed process must not be stopped")

    with pytest.raises(RuntimeError, match="return code 23"):
        _wait_for_local_process(FailedProcess())


def test_system_operations_reuse_one_resident_stack_and_cleanup_after_mission(
    tmp_path,
):
    calls = []

    class Processes:
        owner_started = False
        started = False

        def start_owner(self):
            if not self.owner_started:
                calls.append(("start_owner",))
                self.owner_started = True

        def start_act_worker(self):
            if not self.started:
                calls.append(("start_act",))
                self.started = True

        def stop(self):
            calls.append(("stop_processes",))
            self.owner_started = False
            self.started = False

    class Operations:
        def run_rl_to_dig(self, target):
            calls.append(("rl_to_dig", target))

        def run_act_dig(self, steps):
            calls.append(("act", steps))

        def run_rl_to_dump_and_dump(self):
            calls.append(("dump",))

        def run_rl_return_to_dig(self, target):
            calls.append(("return", target))

        def prewarm_next_act(self, steps):
            calls.append(("already_resident", steps))

        def safe_stop(self):
            calls.append(("terminal_disarm",))

    class Heartbeat:
        running = False

        def start(self):
            calls.append(("start_lease",))
            self.running = True

        def require_healthy(self):
            calls.append(("lease_healthy",))

        def request_stop(self):
            calls.append(("stop_lease_renewals",))

        def stop(self):
            calls.append(("stop_lease",))
            self.running = False

    system = SystemResidentHybridMissionOperations(
        _config(tmp_path),
        processes=Processes(),
        resident_operations=Operations(),
        lease_heartbeat=Heartbeat(),
    )

    system.run_rl_to_dig("dig_01")
    system.run_act_dig(130)
    system.run_rl_to_dump_and_dump()
    system.run_rl_return_to_dig("dig_02")
    system.safe_stop()

    assert calls == [
        ("start_owner",),
        ("start_lease",),
        ("lease_healthy",),
        ("start_act",),
        ("rl_to_dig", "dig_01"),
        ("lease_healthy",),
        ("lease_healthy",),
        ("act", 130),
        ("lease_healthy",),
        ("lease_healthy",),
        ("dump",),
        ("lease_healthy",),
        ("lease_healthy",),
        ("return", "dig_02"),
        ("lease_healthy",),
        ("stop_lease_renewals",),
        ("terminal_disarm",),
        ("stop_lease",),
        ("stop_processes",),
    ]


def test_system_safe_stop_releases_processes_even_if_disarm_reports_failure(
    tmp_path,
):
    calls = []

    class Processes:
        owner_started = True
        started = True

        def stop(self):
            calls.append("stop_processes")
            self.owner_started = False
            self.started = False

    class Operations:
        def safe_stop(self):
            calls.append("terminal_disarm")
            raise RuntimeError("disarm failed")

    class Heartbeat:
        running = True

        def request_stop(self):
            calls.append("stop_lease_renewals")

        def stop(self):
            calls.append("stop_lease")
            self.running = False

    system = SystemResidentHybridMissionOperations(
        _config(tmp_path),
        processes=Processes(),
        resident_operations=Operations(),
        lease_heartbeat=Heartbeat(),
    )

    try:
        system.safe_stop()
    except RuntimeError as exc:
        assert "disarm failed" in str(exc)
    else:
        raise AssertionError("safe_stop must surface the disarm failure")

    assert calls == [
        "stop_lease_renewals",
        "terminal_disarm",
        "stop_lease",
        "stop_processes",
    ]


def test_process_aware_control_forwards_lease_renewal_only_while_stack_runs():
    calls = []

    class Processes:
        def require_owner_running(self):
            calls.append("require_owner_running")

    class Control:
        def renew_lease(self):
            calls.append("renew_lease")
            return ResidentControlStatus(
                phase="idle",
                control_generation=7,
                active=None,
                target=None,
                last_handoff_latency_ms=None,
                rl_is_active=False,
                act_is_active=False,
                act_worker_ready=True,
                act_segment_generation=None,
                act_segment_max_steps=None,
                act_segment_completed_steps=0,
                act_segment_complete=False,
                mission_lease_active=True,
                is_operational=True,
            )

    adapter = ResidentProcessAwareControlAdapter(
        delegate=Control(),
        processes=Processes(),
    )

    assert adapter.renew_lease().mission_lease_active is True
    assert calls == [
        "require_owner_running",
        "renew_lease",
        "require_owner_running",
    ]


def test_process_aware_control_rejects_a_lease_renewal_not_confirmed_by_owner():
    class Processes:
        def require_owner_running(self):
            return None

    class Control:
        def renew_lease(self):
            return ResidentControlStatus(
                phase="idle",
                control_generation=7,
                active=None,
                target=None,
                last_handoff_latency_ms=None,
                rl_is_active=False,
                act_is_active=False,
                act_worker_ready=True,
                act_segment_generation=None,
                act_segment_max_steps=None,
                act_segment_completed_steps=0,
                act_segment_complete=False,
                mission_lease_active=False,
                is_operational=True,
            )

    adapter = ResidentProcessAwareControlAdapter(
        delegate=Control(),
        processes=Processes(),
    )

    with pytest.raises(RuntimeError, match="lease did not become active"):
        adapter.renew_lease()


def test_production_composition_wires_lease_and_prepared_dump_into_resident_ops(
    tmp_path,
):
    class Guided(_Guided):
        rl_airy_repo = tmp_path / "AiryLidar"
        rl_ros_setup = tmp_path / "ros" / "setup.bash"
        rl_workspace_setup = tmp_path / "AiryLidar" / "install" / "setup.bash"
        rl_mission_config = tmp_path / "AiryLidar" / "mission.json"
        ack_timeout_s = 5
        rl_timeout_s = 90

    class Processes:
        started = False

        def start(self):
            self.started = True

        def require_running(self):
            return None

        def stop(self):
            self.started = False

    system = SystemResidentHybridMissionOperations(
        _config(tmp_path),
        guided_config=Guided(tmp_path),
        processes=Processes(),
        rl_operations=object(),
        output=lambda _message: None,
        timestamp="20260821_140000",
    )

    assert isinstance(system._lease_heartbeat, ResidentMissionLeaseHeartbeat)
    assert isinstance(
        system._operations._prepared_dump,
        SystemPreparedDumpAdapter,
    )
    assert system._operations._prepared_dump_lead_steps == 20


def test_act_worker_loss_fails_fast_and_terminally_disarms_without_waiting_timeout():
    calls = []

    def status(*, worker_ready, active=False):
        binding = (
            ResidentPolicyBinding("act_dig", "manual_action")
            if active
            else None
        )
        return ResidentControlStatus(
            phase="active" if active else "idle",
            control_generation=7,
            active=binding,
            target=None,
            last_handoff_latency_ms=None,
            rl_is_active=False,
            act_is_active=active,
            act_worker_ready=worker_ready,
            act_segment_generation=7 if active else None,
            act_segment_max_steps=130 if active else None,
            act_segment_completed_steps=0,
            act_segment_complete=False,
            mission_lease_active=True,
            is_operational=True,
        )

    class Processes:
        def require_running(self):
            calls.append("require_running")

    class Control:
        def ensure_ready(self):
            calls.append("ensure_ready")
            return status(worker_ready=True)

        def activate_act(self, max_steps):
            calls.append(("activate_act", max_steps))
            return status(worker_ready=True, active=True)

        def status(self):
            calls.append("status")
            return status(worker_ready=False, active=True)

        def terminal_disarm(self):
            calls.append("terminal_disarm")
            return status(worker_ready=False)

    control = ResidentProcessAwareControlAdapter(
        delegate=Control(),
        processes=Processes(),
    )
    operations = ResidentHybridMissionOperations(
        control=control,
        behavior=object(),
        act_run_timeout_s=90,
        poll_interval_s=0.1,
        sleep=lambda seconds: calls.append(("sleep", seconds)),
    )

    with pytest.raises(RuntimeError, match="ACT worker is not ready"):
        operations.run_act_dig(130)

    assert calls.count("status") == 1
    assert calls[-1] == "terminal_disarm"

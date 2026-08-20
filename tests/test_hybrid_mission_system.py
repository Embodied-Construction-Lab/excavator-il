import signal

import pytest

from excavator_il.hybrid_mission import HybridMissionConfig
from excavator_il.hybrid_mission_system import SystemHybridMissionOperations


class _RlOperations:
    def __init__(self, *, fail_follow=False):
        self.calls = []
        self.fail_follow = fail_follow

    def start_rl_runtime(self):
        self.calls.append("start_rl_runtime")

    def run_rl_follow(self, phase, *, target_id=None):
        self.calls.append(("run_rl_follow", phase, target_id))
        if self.fail_follow:
            raise RuntimeError("follow failed")

    def run_rl_fixed_action(self, behavior, *, behavior_port):
        self.calls.append(("run_rl_fixed_action", behavior, behavior_port))

    def stop_rl_runtime_and_wait_for_serial(self):
        self.calls.append("stop_rl_runtime_and_wait_for_serial")


def _config(tmp_path):
    return HybridMissionConfig(
        guided_config=tmp_path / "guided.json",
        act_max_steps=130,
        act_ready_timeout_s=60,
        act_run_timeout_s=90,
        act_remote_script="scripts/run_act_motion.sh",
        rl_behavior_port=18083,
    )


def _guided(tmp_path):
    class _GuidedConfig:
        orin_ssh_host = "jetson16@192.168.50.2"
        orin_repo = "/srv/excavator-il"
        rl_serial_port = "/dev/ttyTHS1"
        rl_serial_release_timeout_s = 8
        log_dir = tmp_path

    return _GuidedConfig()


def test_hybrid_system_adapter_hands_serial_back_after_each_rl_segment(tmp_path):
    rl = _RlOperations()
    operations = SystemHybridMissionOperations(
        _config(tmp_path),
        guided_config=_guided(tmp_path),
        rl_operations=rl,
        output=lambda _message: None,
    )

    operations.run_rl_to_dig("dig_03")
    operations.run_rl_to_dump_and_dump()
    operations.run_rl_return_to_dig("dig_03")

    assert rl.calls == [
        "start_rl_runtime",
        ("run_rl_follow", "dig", "dig_03"),
        "stop_rl_runtime_and_wait_for_serial",
        "start_rl_runtime",
        ("run_rl_follow", "dump", None),
        ("run_rl_fixed_action", "ExecuteDump", 18083),
        "stop_rl_runtime_and_wait_for_serial",
        "start_rl_runtime",
        ("run_rl_follow", "dig", "dig_03"),
        "stop_rl_runtime_and_wait_for_serial",
    ]


def test_hybrid_system_adapter_releases_rl_after_follow_failure(tmp_path):
    rl = _RlOperations(fail_follow=True)
    operations = SystemHybridMissionOperations(
        _config(tmp_path),
        guided_config=_guided(tmp_path),
        rl_operations=rl,
        output=lambda _message: None,
    )

    with pytest.raises(RuntimeError, match="follow failed"):
        operations.run_rl_to_dig("dig_01")

    assert rl.calls[-1] == "stop_rl_runtime_and_wait_for_serial"


def test_hybrid_system_adapter_runs_bounded_act_and_confirms_release(
    tmp_path, monkeypatch
):
    processes = []

    class _ActProcess:
        returncode = 0

        def __init__(self, argv, **_kwargs):
            self.argv = argv
            processes.append(self)

        def wait_for(self, predicate, timeout_s, *, after_index=-1):
            lines = ("HYBRID_ACT_PID=4242", "ACT hardware ready: mode=motion")
            line = next(candidate for candidate in lines if predicate(candidate))
            return lines.index(line), line

        def wait(self, timeout_s=5.0):
            assert timeout_s == 90

        def stop(self, signum, *, timeout_s=5.0):
            assert signum == signal.SIGTERM

    remote_checks = []
    rl = _RlOperations()
    operations = SystemHybridMissionOperations(
        _config(tmp_path),
        guided_config=_guided(tmp_path),
        rl_operations=rl,
        line_process_factory=_ActProcess,
        output=lambda _message: None,
    )
    monkeypatch.setattr(
        operations,
        "_run_remote",
        lambda command: remote_checks.append(command) or "released\n",
    )

    operations.run_act_dig(130)

    rendered = " ".join(processes[0].argv)
    assert "--authorization ALLOW_ACT_MACHINE_MOTION" in rendered
    assert "--max-steps 130" in rendered
    assert operations._act_process is None
    assert any("fuser" in command for command in remote_checks)

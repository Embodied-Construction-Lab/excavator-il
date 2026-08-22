import signal

import pytest

from excavator_il.hybrid_mission_resident import PreparedDumpActivation
from excavator_il.remote_runtime import LineWaitTimeout
from excavator_il.resident_prepared_follow import SystemPreparedDumpAdapter


def _adapter(tmp_path, process_factory):
    airy_repo = tmp_path / "AiryLidar"
    ros_setup = tmp_path / "opt" / "ros" / "setup.bash"
    workspace_setup = airy_repo / "ros2_ws" / "install" / "setup.bash"
    mission_config = airy_repo / "mission" / "config" / "mission.yaml"
    airy_repo.mkdir()
    ros_setup.parent.mkdir(parents=True)
    ros_setup.write_text("", encoding="utf-8")
    workspace_setup.parent.mkdir(parents=True)
    workspace_setup.write_text("", encoding="utf-8")
    mission_config.parent.mkdir(parents=True)
    mission_config.write_text("", encoding="utf-8")
    return SystemPreparedDumpAdapter(
        airy_repo=airy_repo,
        ros_setup=ros_setup,
        workspace_setup=workspace_setup,
        mission_config=mission_config,
        log_dir=tmp_path / "logs",
        wait_s=5,
        ready_grace_ms=300,
        run_timeout_s=90,
        start_tolerance_m=0.15,
        line_process_factory=process_factory,
        output=lambda _message: None,
        timestamp="20260821_140000",
    )


class _PreparedProcess:
    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.returncode = None
        self.wait_for_calls = []
        self.wait_calls = []
        self.stop_calls = []

    def wait_for(self, predicate, timeout_s, *, after_index=-1):
        self.wait_for_calls.append((timeout_s, after_index))
        line = "2026-08-21 INFO prepared follow ready: trajectory_id=traj-7"
        assert predicate(line)
        return 0, line

    def wait(self, timeout_s=5.0):
        self.wait_calls.append(timeout_s)
        self.returncode = 0

    def stop(self, signum, *, timeout_s=5.0):
        self.stop_calls.append((signum, timeout_s))
        self.returncode = 130


def test_prepare_starts_the_live_planner_without_waiting_for_readiness(tmp_path):
    created = []

    def factory(*args, **kwargs):
        process = _PreparedProcess(*args, **kwargs)
        created.append(process)
        return process

    adapter = _adapter(tmp_path, factory)

    adapter.start_prepare()

    assert len(created) == 1
    process = created[0]
    assert process.wait_for_calls == []
    assert process.wait_calls == []
    gate = tmp_path / "logs" / "hybrid_mission_20260821_140000.prepared-dump.start"
    assert process.argv == [
        "/bin/zsh",
        "-lc",
        " && ".join(
            (
                f"source {tmp_path / 'opt' / 'ros' / 'setup.bash'}",
                f"source {tmp_path / 'AiryLidar' / 'ros2_ws' / 'install' / 'setup.bash'}",
                f"cd {tmp_path / 'AiryLidar'}",
                " ".join(
                    (
                        "exec /usr/bin/python3 -m",
                        "mission.runtime_ros.run_prepared_plan_follow_live",
                        "dump",
                        "--mission",
                        str(tmp_path / "AiryLidar" / "mission" / "config" / "mission.yaml"),
                        "--wait-s 5",
                        "--start-gate",
                        str(gate),
                        "--first-waypoint-distance-m 0.15",
                    )
                ),
            )
        ),
    ]
    assert process.kwargs["prefix"] == "prepared-dump"
    assert not gate.exists()


def test_activate_waits_for_stable_ready_marker_then_releases_one_shot_gate(tmp_path):
    created = []

    class Process(_PreparedProcess):
        def wait(self, timeout_s=5.0):
            gate = (
                tmp_path
                / "logs"
                / "hybrid_mission_20260821_140000.prepared-dump.start"
            )
            assert gate.is_file()
            super().wait(timeout_s)

    def factory(*args, **kwargs):
        process = Process(*args, **kwargs)
        created.append(process)
        return process

    adapter = _adapter(tmp_path, factory)
    adapter.start_prepare()

    outcome = adapter.activate_prepared()

    assert outcome is PreparedDumpActivation.ACTIVATED
    assert created[0].wait_for_calls == [(0.3, -1)]
    assert created[0].wait_calls == [90.0]
    assert not (
        tmp_path
        / "logs"
        / "hybrid_mission_20260821_140000.prepared-dump.start"
    ).exists()


def test_explicit_safe_fallback_before_ready_is_the_only_fallback_result(tmp_path):
    class Process(_PreparedProcess):
        def wait_for(self, _predicate, _timeout_s, *, after_index=-1):
            del after_index
            self.returncode = 3
            raise RuntimeError("prepared-dump exited before readiness")

    adapter = _adapter(tmp_path, Process)
    adapter.start_prepare()

    assert adapter.activate_prepared() is PreparedDumpActivation.FALLBACK_SAFE


def test_readiness_timeout_cancels_then_explicitly_allows_safe_replan(tmp_path):
    created = []

    class Process(_PreparedProcess):
        def wait_for(self, _predicate, _timeout_s, *, after_index=-1):
            del after_index
            raise LineWaitTimeout("not ready")

    def factory(*args, **kwargs):
        process = Process(*args, **kwargs)
        created.append(process)
        return process

    adapter = _adapter(tmp_path, factory)
    adapter.start_prepare()

    outcome = adapter.activate_prepared()

    assert outcome is PreparedDumpActivation.FALLBACK_SAFE
    assert created[0].stop_calls == [(signal.SIGINT, 3.0)]

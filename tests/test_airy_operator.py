import signal

from excavator_il.airy_operator import AiryOperatorSupervisor


class _GuidedConfig:
    rl_ros_setup = "/opt/ros/humble/setup.zsh"
    rl_workspace_setup = "/workspace/AiryLidar/ros2_ws/install/setup.zsh"
    rl_airy_repo = "/workspace/AiryLidar"
    orin_ssh_host = "jetson16@192.168.50.2"
    log_dir = "/tmp/excavator-il-tests"


def test_airy_operator_supervisor_starts_existing_launch_and_stops_it():
    processes = []

    class _Process:
        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.kwargs = kwargs
            self.running = True
            processes.append(self)

        def wait_for(self, predicate, timeout_s, *, after_index=-1):
            assert timeout_s == 60
            assert after_index == -1
            line = "[live_plan_server]: live Plan ready: scope=execution_strict"
            assert predicate(line)
            return 0, line

        def stop(self, signum, *, timeout_s=5.0):
            assert signum == signal.SIGINT
            assert timeout_s == 10.0
            self.running = False

        @property
        def lines(self):
            return ("live Plan ready",)

    supervisor = AiryOperatorSupervisor(
        guided_config=_GuidedConfig(),
        behavior_port=18083,
        line_process_factory=_Process,
    )

    ready = supervisor.start()

    assert ready.stage == "ready"
    command = processes[0].argv[-1]
    assert "ros2 launch airy_excavator_bringup operator.launch.py" in command
    assert "profile:=live_commissioning" in command
    assert "motion_authorization:=ALLOW_LIVE_MACHINE_MOTION" in command
    assert "orin_host:=192.168.50.2" in command
    assert "orin_port:=18083" in command

    stopped = supervisor.stop()
    assert stopped.stage == "stopped"


def test_v3a_operator_starts_display_only_shadow_with_trajectory_file(tmp_path):
    processes = []
    trajectory = tmp_path / "v3a_active_trajectory.json"

    class _Process:
        running = True

        def __init__(self, argv, **_kwargs):
            self.argv = argv
            processes.append(self)

        def wait_for(self, predicate, timeout_s, *, after_index=-1):
            assert timeout_s == 60
            assert after_index == -1
            line = "live_shadow: live state/FK/perception with NoMotionBackend"
            assert predicate(line)
            return 0, line

        def stop(self, _signum, *, timeout_s=5.0):
            self.running = False

        @property
        def lines(self):
            return ()

    supervisor = AiryOperatorSupervisor(
        guided_config=_GuidedConfig(),
        behavior_port=None,
        profile="live_shadow",
        trajectory_path=trajectory,
        external_state_bridge=True,
        line_process_factory=_Process,
    )

    assert supervisor.start().stage == "ready"
    command = processes[0].argv[-1]
    assert "profile:=live_shadow" in command
    assert "motion_authorization:=LOCKED" in command
    assert f"v3a_trajectory_path:={trajectory.resolve()}" in command
    assert "external_state_bridge:=true" in command
    assert "ALLOW_LIVE_MACHINE_MOTION" not in command


def test_airy_operator_supervisor_cleans_process_tree_after_rviz_closes():
    processes = []

    class _Process:
        def __init__(self, _argv, **_kwargs):
            self.running = True
            self.stop_calls = []
            processes.append(self)

        def wait_for(self, predicate, _timeout_s, *, after_index=-1):
            assert after_index == -1
            assert predicate("[live_plan_server]: live Plan ready: scope=execution_strict")
            return 0, "live Plan ready"

        def stop(self, signum, *, timeout_s=5.0):
            self.stop_calls.append((signum, timeout_s))
            self.running = False

        @property
        def lines(self):
            return ("rviz exited",)

    supervisor = AiryOperatorSupervisor(
        guided_config=_GuidedConfig(),
        behavior_port=18083,
        line_process_factory=_Process,
    )
    supervisor.start()
    processes[0].running = False

    failed = supervisor.snapshot()

    assert failed.stage == "failed"
    assert failed.error == "AiryLidar Operator 已意外退出"
    assert processes[0].stop_calls == [(signal.SIGINT, 10.0)]
    assert supervisor.stop().stage == "stopped"


def test_airy_operator_supervisor_cleans_stale_process_before_restart():
    processes = []

    class _Process:
        def __init__(self, _argv, **_kwargs):
            self.running = True
            self.stop_calls = []
            processes.append(self)

        def wait_for(self, predicate, _timeout_s, *, after_index=-1):
            assert after_index == -1
            assert predicate("[live_plan_server]: live Plan ready: scope=execution_strict")
            return 0, "live Plan ready"

        def stop(self, signum, *, timeout_s=5.0):
            self.stop_calls.append((signum, timeout_s))
            self.running = False

        @property
        def lines(self):
            return ()

    supervisor = AiryOperatorSupervisor(
        guided_config=_GuidedConfig(),
        behavior_port=18083,
        line_process_factory=_Process,
    )
    supervisor.start()
    processes[0].running = False

    restarted = supervisor.start()

    assert restarted.stage == "ready"
    assert len(processes) == 2
    assert processes[0].stop_calls == [(signal.SIGINT, 10.0)]


def test_airy_operator_stop_cleans_children_after_launch_leader_exits():
    processes = []

    class _Process:
        def __init__(self, _argv, **_kwargs):
            self.running = True
            self.stop_calls = []
            processes.append(self)

        def wait_for(self, predicate, _timeout_s, *, after_index=-1):
            assert after_index == -1
            assert predicate("[live_plan_server]: live Plan ready: scope=execution_strict")
            return 0, "live Plan ready"

        def stop(self, signum, *, timeout_s=5.0):
            self.stop_calls.append((signum, timeout_s))
            self.running = False

        @property
        def lines(self):
            return ("rviz exited",)

    supervisor = AiryOperatorSupervisor(
        guided_config=_GuidedConfig(),
        behavior_port=18083,
        line_process_factory=_Process,
    )
    supervisor.start()
    processes[0].running = False

    stopped = supervisor.stop()

    assert stopped.stage == "stopped"
    assert processes[0].stop_calls == [(signal.SIGINT, 10.0)]

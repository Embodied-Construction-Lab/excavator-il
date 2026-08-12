import json
import io
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from excavator_il.guided_episode import (
    GuidedEpisodeConfig,
    SystemGuidedEpisodeOperations,
    run_guided_episode,
)


class _FakeOperations:
    def __init__(self):
        self.events = []
        self.episode_index = 0

    def preflight(self):
        self.events.append("preflight")

    def start_collector(self):
        self.events.append("start_collector")

    def start_teleop(self):
        self.events.append("start_teleop")

    def wait_for_ack(self, timeout_s):
        self.events.append(("wait_for_ack", timeout_s))

    def wait_for_deadman_pressed(self):
        self.events.append("wait_for_deadman_pressed")

    def wait_for_deadman_released(self):
        self.events.append("wait_for_deadman_released")

    def start_episode(self):
        self.events.append("start_episode")
        self.episode_index += 1
        return f"/data/raw/episode_{self.episode_index:04d}"

    def seal_episode(self):
        self.events.append("seal_episode")
        return f"/data/raw/episode_{self.episode_index:04d}"

    def finalize_episode(self, episode_path, result, reason=""):
        self.events.append(("finalize_episode", episode_path, result, reason))
        return episode_path

    def abort_episode(self, reason):
        self.events.append(("abort_episode", reason))
        return f"/data/raw/episode_{self.episode_index:04d}"

    def discard_episode(self, episode_path):
        self.events.append(("discard_episode", episode_path))
        self.episode_index -= 1

    def stop_teleop(self):
        self.events.append("stop_teleop")

    def stop_collector(self):
        self.events.append("stop_collector")

    def build_and_validate(self, episode_path):
        self.events.append(("build_and_validate", episode_path))


def test_guided_episode_config_resolves_pc_paths_and_validates_contract(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "guided.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_guided_episode_config.v2",
                "teleop_config": "teleop.pc.json",
                "orin": {
                    "ssh_host": "operator@192.0.2.10",
                    "repo": "/srv/excavator-il",
                    "executable": "/opt/excavator/bin/excavator-il",
                    "collection_config": "config/collection.orin.json",
                },
                "episode": {
                    "task": "DiagnosticBoomJog",
                    "operator_id": "operator_01",
                    "dig_target_m": [0.8, 0.0, -0.2],
                    "material_id": "soil_default",
                },
                "runtime": {
                    "collector_ready_timeout_s": 8,
                    "ack_timeout_s": 8,
                    "teleop_print_every": 1,
                    "zero_soak_duration_s": 30,
                    "log_dir": "../logs",
                },
            }
        ),
        encoding="utf-8",
    )

    config = GuidedEpisodeConfig.load(config_path)

    assert config.teleop_config == config_dir / "teleop.pc.json"
    assert config.log_dir == tmp_path / "logs"
    assert config.orin_ssh_host == "operator@192.0.2.10"
    assert config.dig_target_m == (0.8, 0.0, -0.2)
    assert config.teleop_print_every == 1
    assert config.zero_soak_duration_s == 30


def test_guided_episode_config_rejects_unsafe_or_inconsistent_values(tmp_path):
    config_path = tmp_path / "guided.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_guided_episode_config.v2",
                "teleop_config": "teleop.pc.json",
                "orin": {
                    "ssh_host": "operator@host; reboot",
                    "repo": "/srv/excavator-il",
                    "executable": "/opt/excavator/bin/excavator-il",
                    "collection_config": "config/collection.orin.json",
                },
                "episode": {
                    "task": "DiagnosticBoomJog",
                    "operator_id": "operator_01",
                    "dig_target_m": [0.8, 0.0, -0.2],
                    "material_id": "soil_default",
                },
                "runtime": {
                    "collector_ready_timeout_s": 8,
                    "ack_timeout_s": 8,
                    "teleop_print_every": 1,
                    "log_dir": "../logs",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        GuidedEpisodeConfig.load(config_path)


def _guided_config(tmp_path):
    return GuidedEpisodeConfig(
        teleop_config=tmp_path / "teleop.json",
        orin_ssh_host="operator@192.0.2.10",
        orin_repo="/srv/excavator-il",
        orin_executable="/opt/excavator/bin/excavator-il",
        orin_collection_config="config/collection.orin.json",
        task="ExecuteDig",
        operator_id="operator_01",
        dig_target_m=(0.8, 0.0, -0.2),
        material_id="soil_default",
        collector_ready_timeout_s=8,
        ack_timeout_s=8,
        teleop_print_every=1,
        log_dir=tmp_path / "logs",
        failure_reason="diagnostic_task_failed",
    )


def test_guided_episode_stands_by_before_deadman_and_seals_immediately_on_release(
    tmp_path,
):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()
    prompts = []

    episode_path = run_guided_episode(
        config,
        operations,
        input_fn=lambda prompt: prompts.append(prompt) or "成功",
        output=lambda message: None,
    )

    assert episode_path == "/data/raw/episode_0001"
    assert len(prompts) == 1
    assert operations.events == [
        "preflight",
        "start_collector",
        "start_episode",
        "start_teleop",
        ("wait_for_ack", 8),
        "wait_for_deadman_pressed",
        "wait_for_deadman_released",
        "seal_episode",
        (
            "finalize_episode",
            "/data/raw/episode_0001",
            "success",
            "",
        ),
        "stop_teleop",
        "stop_collector",
        ("build_and_validate", "/data/raw/episode_0001"),
    ]


def test_guided_episode_interrupt_aborts_before_stopping_motion_io(tmp_path):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()

    def interrupt_during_result(unused_prompt):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_guided_episode(
            config,
            operations,
            input_fn=interrupt_during_result,
            output=lambda message: None,
        )

    assert operations.events[-4:] == [
        (
            "finalize_episode",
            "/data/raw/episode_0001",
            "aborted",
            "guided_episode_interrupted",
        ),
        "stop_teleop",
        "stop_collector",
        ("build_and_validate", "/data/raw/episode_0001"),
    ]


def test_guided_episode_interrupt_while_standing_by_does_not_build_empty_attempt(
    tmp_path,
):
    config = _guided_config(tmp_path)

    class InterruptWhileStandingBy(_FakeOperations):
        def wait_for_deadman_pressed(self):
            self.events.append("wait_for_deadman_pressed")
            raise KeyboardInterrupt

    operations = InterruptWhileStandingBy()

    with pytest.raises(KeyboardInterrupt):
        run_guided_episode(
            config,
            operations,
            input_fn=lambda prompt: "成功",
            output=lambda message: None,
        )

    assert ("abort_episode", "guided_episode_interrupted") in operations.events
    assert not any(
        isinstance(event, tuple) and event[0] == "build_and_validate"
        for event in operations.events
    )


def test_guided_episode_retake_deletes_attempt_and_reuses_episode_id(tmp_path):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()
    outcomes = iter(("重录", "失败"))

    episode_path = run_guided_episode(
        config,
        operations,
        input_fn=lambda prompt: next(outcomes),
        output=lambda message: None,
    )

    assert episode_path == "/data/raw/episode_0001"
    assert operations.events == [
        "preflight",
        "start_collector",
        "start_episode",
        "start_teleop",
        ("wait_for_ack", 8),
        "wait_for_deadman_pressed",
        "wait_for_deadman_released",
        "seal_episode",
        ("discard_episode", "/data/raw/episode_0001"),
        "start_episode",
        "wait_for_deadman_pressed",
        "wait_for_deadman_released",
        "seal_episode",
        (
            "finalize_episode",
            "/data/raw/episode_0001",
            "failure",
            "diagnostic_task_failed",
        ),
        "stop_teleop",
        "stop_collector",
        ("build_and_validate", "/data/raw/episode_0001"),
    ]


def test_guided_episode_can_retain_a_failed_attempt_and_validate_it(tmp_path):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()

    episode_path = run_guided_episode(
        config,
        operations,
        input_fn=lambda prompt: "f",
        output=lambda message: None,
    )

    assert episode_path == "/data/raw/episode_0001"
    assert (
        "finalize_episode",
        "/data/raw/episode_0001",
        "failure",
        "diagnostic_task_failed",
    ) in operations.events
    assert operations.events[-1] == (
        "build_and_validate",
        "/data/raw/episode_0001",
    )


def test_system_operations_manage_exact_collector_and_episode_commands(
    tmp_path, monkeypatch
):
    (tmp_path / "teleop.json").write_text("{}", encoding="utf-8")
    config = GuidedEpisodeConfig(
        teleop_config=tmp_path / "teleop.json",
        orin_ssh_host="operator@192.0.2.10",
        orin_repo="/srv/excavator-il",
        orin_executable="/opt/excavator/bin/excavator-il",
        orin_collection_config="config/collection.orin.json",
        task="DiagnosticBoomJog",
        operator_id="operator_01",
        dig_target_m=(0.8, 0.0, -0.2),
        material_id="soil_default",
        collector_ready_timeout_s=8,
        ack_timeout_s=8,
        teleop_print_every=1,
        log_dir=tmp_path / "logs",
    )
    popen_calls = []
    signal_calls = []
    run_calls = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            popen_calls.append(argv)
            if "collect" in " ".join(argv):
                output = "GUIDED_COLLECTOR_PID=4321\ncollector ready: serial=/dev/ttyTHS1\n"
            else:
                output = (
                    "teleop seq=20 ack=19 ack_lag=1 accepted_acks=20 "
                    "rejected_acks=0 deadman=False axes=(0,0,0,0,0,0)\n"
                    "teleop seq=21 ack=20 ack_lag=1 accepted_acks=21 "
                    "rejected_acks=0 deadman=True axes=(0,0,0,0,0,0)\n"
                    "teleop seq=22 ack=21 ack_lag=1 accepted_acks=22 "
                    "rejected_acks=0 deadman=False axes=(0,0,0,0,0,0)\n"
                )
            self.stdout = io.StringIO(output)
            self.returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, signum):
            signal_calls.append(signum)
            self.returncode = 0

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    def fake_run(argv, **kwargs):
        run_calls.append(argv)
        rendered = " ".join(argv)
        if " episode " in rendered and " start" in rendered:
            stdout = json.dumps(
                {"ok": True, "active": True, "path": "/data/raw/episode_0001"}
            )
        elif " episode " in rendered and " seal" in rendered:
            stdout = json.dumps(
                {
                    "ok": True,
                    "active": False,
                    "status": "pending_review",
                    "path": "/data/raw/episode_0001",
                }
            )
        elif " episode " in rendered and " finalize" in rendered:
            stdout = json.dumps(
                {
                    "ok": True,
                    "active": False,
                    "status": "complete",
                    "path": "/data/raw/episode_0001",
                }
            )
        elif " episode " in rendered and " stop" in rendered:
            stdout = json.dumps(
                {"ok": True, "active": False, "path": "/data/raw/episode_0001"}
            )
        elif " episode " in rendered and " abort" in rendered:
            stdout = json.dumps(
                {
                    "ok": True,
                    "active": False,
                    "status": "aborted",
                    "path": "/data/raw/episode_0001",
                }
            )
        elif "quality_report.json" in rendered:
            stdout = json.dumps({"episode_id": "episode_0001", "valid": True})
        else:
            stdout = "{}"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("excavator_il.guided_episode.subprocess.Popen", FakePopen)
    monkeypatch.setattr("excavator_il.guided_episode.subprocess.run", fake_run)
    operations = SystemGuidedEpisodeOperations(
        config,
        output=lambda message: None,
        timestamp="20260811_200000",
    )

    operations.preflight()
    operations.start_collector()
    assert operations.start_episode() == "/data/raw/episode_0001"
    operations.start_teleop()
    operations.wait_for_ack(8)
    operations.wait_for_deadman_pressed()
    operations.wait_for_deadman_released()
    assert operations.seal_episode() == "/data/raw/episode_0001"
    assert (
        operations.finalize_episode("/data/raw/episode_0001", "success")
        == "/data/raw/episode_0001"
    )
    operations.stop_teleop()
    operations.stop_collector()
    operations.build_and_validate("/data/raw/episode_0001")

    assert len(popen_calls) == 2
    teleop_argv = next(call for call in popen_calls if "teleop" in call)
    assert teleop_argv[1:3] == ["-u", "-m"]
    assert any("kill -TERM 4321" in " ".join(call) for call in run_calls)
    assert any("build-steps" in " ".join(call) for call in run_calls)
    assert any("validate" in " ".join(call) for call in run_calls)
    assert signal_calls


def test_inspect_zero_soak_preserves_failed_quality_report(tmp_path, monkeypatch):
    (tmp_path / "teleop.json").write_text("{}", encoding="utf-8")
    config = _guided_config(tmp_path)
    report = {
        "episode_id": "episode_0001",
        "passed": False,
        "failure_reasons": ["new_sensor_state rate is outside its allowed range"],
    }

    def fake_run(argv, **kwargs):
        return SimpleNamespace(
            returncode=3,
            stdout=json.dumps(report),
            stderr="",
        )

    monkeypatch.setattr("excavator_il.guided_episode.subprocess.run", fake_run)
    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)

    assert operations.inspect_zero_soak("/data/raw/episode_0001") == report


def test_system_operations_only_discard_retake_episode_started_by_this_run(
    tmp_path, monkeypatch
):
    (tmp_path / "teleop.json").write_text("{}", encoding="utf-8")
    config = _guided_config(tmp_path)
    run_calls = []

    def fake_run(argv, **kwargs):
        run_calls.append(argv)
        rendered = " ".join(argv)
        if " episode " in rendered and " start" in rendered:
            stdout = json.dumps(
                {"ok": True, "active": True, "path": "/data/raw/episode_0007"}
            )
        elif " episode " in rendered and " seal" in rendered:
            stdout = json.dumps(
                {
                    "ok": True,
                    "active": False,
                    "status": "pending_review",
                    "path": "/data/raw/episode_0007",
                }
            )
        else:
            stdout = ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("excavator_il.guided_episode.subprocess.run", fake_run)
    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)

    with pytest.raises(RuntimeError, match="unapproved"):
        operations.discard_episode("/data/raw/episode_0006")

    assert operations.start_episode() == "/data/raw/episode_0007"
    assert operations.seal_episode() == "/data/raw/episode_0007"
    operations.discard_episode("/data/raw/episode_0007")

    delete_command = " ".join(run_calls[-1])
    assert "rm -rf -- /data/raw/episode_0007" in delete_command
    assert "/data/raw/episode_0006" not in delete_command

    with pytest.raises(RuntimeError, match="unapproved"):
        operations.discard_episode("/data/raw/episode_0007")


def test_stop_collector_does_not_kill_remote_pid_after_collector_exited(
    tmp_path, monkeypatch
):
    config = _guided_config(tmp_path)
    remote_commands = []

    class ExitedCollector:
        running = False

        def wait(self):
            return None

    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)
    operations._collector = ExitedCollector()
    operations._collector_pid = 15303
    monkeypatch.setattr(
        operations,
        "_run_ssh",
        lambda command: remote_commands.append(command) or "",
    )

    operations.stop_collector()

    assert remote_commands == []
    assert operations._collector is None
    assert operations._collector_pid is None


def test_collector_extra_includes_episode_image_validation_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    collector_dependencies = pyproject["project"]["optional-dependencies"]["collector"]

    assert any(dependency.startswith("Pillow>=") for dependency in collector_dependencies)

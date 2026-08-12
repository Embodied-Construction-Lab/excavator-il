import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from excavator_il.cli import main


@dataclass(frozen=True)
class _Result:
    value: str = "ok"


@dataclass(frozen=True)
class _ZeroResult:
    passed: bool
    episode_id: str


def test_validate_command_prints_machine_readable_report(rgb_episode_factory, capsys):
    episode = rgb_episode_factory()

    exit_code = main(["validate", str(episode)])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["episode_id"] == "episode_0001"
    assert report["step_count"] == 3
    assert report["image_shape"] == [24, 32, 3]


def test_validate_command_returns_nonzero_for_invalid_episode(tmp_path, capsys):
    exit_code = main(["validate", str(tmp_path / "missing")])

    assert exit_code == 2
    assert "missing required file" in capsys.readouterr().err


def test_cli_dispatches_collection_tools_without_importing_training_stack(monkeypatch, capsys):
    from excavator_il import episode_builder, joystick_diagnostic, teleop
    from excavator_il.collector import client, config, service

    calls = []
    monkeypatch.setattr(teleop.TeleopConfig, "load", lambda path: f"teleop:{path}")
    monkeypatch.setattr(
        teleop, "run_teleop", lambda loaded, print_every: calls.append((loaded, print_every))
    )
    monkeypatch.setattr(teleop, "list_pygame_devices", lambda: [{"device_id": "one"}])
    monkeypatch.setattr(
        joystick_diagnostic,
        "run_joystick_diagnostic",
        lambda loaded: SimpleNamespace(matches_config=True),
    )
    monkeypatch.setattr(service, "run_collector", lambda path: calls.append(("collect", path)))
    monkeypatch.setattr(episode_builder, "build_steps", lambda *args, **kwargs: _Result())
    monkeypatch.setattr(
        config,
        "load_collection_config",
        lambda path: SimpleNamespace(episode_control_socket="fixture.sock"),
    )
    monkeypatch.setattr(
        client,
        "send_episode_command",
        lambda path, request: {"ok": True, "path": str(path), "request": request},
    )

    assert main(["teleop", "--config", "teleop.json", "--print-every", "7"]) == 0
    assert main(["list-joysticks"]) == 0
    assert main(["diagnose-joysticks", "--config", "teleop.json"]) == 0
    assert main(["collect", "--config", "collection.json"]) == 0
    assert main(["build-steps", "episode_0001"]) == 0
    assert main(
        [
            "episode",
            "--config",
            "collection.json",
            "start",
            "--task",
            "ExecuteDig",
            "--operator",
            "operator_01",
            "--dig-target-m",
            "1",
            "2",
            "3",
            "--material-id",
            "soil",
        ]
    ) == 0
    assert main(["episode", "stop", "--failure-reason", "bucket_empty"]) == 0
    assert main(["episode", "abort", "--reason", "emergency_stop"]) == 0
    assert main(["episode", "seal"]) == 0
    assert main(
        [
            "episode",
            "finalize",
            "/data/raw/episode_0001",
            "--result",
            "failure",
            "--failure-reason",
            "diagnostic_task_failed",
        ]
    ) == 0
    assert calls == [("teleop:teleop.json", 7), ("collect", "collection.json")]
    assert "device_id" in capsys.readouterr().out


def test_diagnose_joysticks_returns_nonzero_when_mapping_does_not_match(monkeypatch):
    from excavator_il import joystick_diagnostic, teleop

    monkeypatch.setattr(teleop.TeleopConfig, "load", lambda path: f"teleop:{path}")
    monkeypatch.setattr(
        joystick_diagnostic,
        "run_joystick_diagnostic",
        lambda loaded: SimpleNamespace(matches_config=False),
    )

    assert main(["diagnose-joysticks"]) == 3


def test_inspect_zero_soak_returns_nonzero_for_unsafe_episode(monkeypatch, capsys):
    from excavator_il import zero_soak

    monkeypatch.setattr(
        zero_soak,
        "inspect_zero_command_episode",
        lambda path: _ZeroResult(passed=False, episode_id=str(path)),
    )

    assert main(["inspect-zero-soak", "episode_0007"]) == 3
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_diagnose_joysticks_handles_operator_interrupt_without_traceback(
    monkeypatch, capsys
):
    from excavator_il import joystick_diagnostic, teleop

    monkeypatch.setattr(teleop.TeleopConfig, "load", lambda path: f"teleop:{path}")

    def interrupt(unused_config):
        raise KeyboardInterrupt

    monkeypatch.setattr(joystick_diagnostic, "run_joystick_diagnostic", interrupt)

    assert main(["diagnose-joysticks"]) == 130
    assert "diagnostic interrupted" in capsys.readouterr().err


def test_cli_dispatches_optional_training_commands(monkeypatch, capsys):
    pytest.importorskip("lerobot")
    from excavator_il import act_smoke, lerobot_conversion

    inference_arguments = {}

    def infer(**kwargs):
        inference_arguments.update(kwargs)
        return _Result()

    monkeypatch.setattr(lerobot_conversion, "convert_episodes", lambda *a, **k: _Result())
    monkeypatch.setattr(act_smoke, "run_act_smoke_train_step", lambda **k: _Result())
    monkeypatch.setattr(act_smoke, "run_act_checkpoint_inference", infer)

    assert main(["convert", "ep", "--output-root", "out"]) == 0
    assert main(["smoke-train"]) == 0
    assert main(
        [
            "smoke-infer",
            "checkpoint",
            "--dataset-root",
            "dataset",
            "--repo-id",
            "local/dataset",
            "--warmup-runs",
            "2",
            "--timed-runs",
            "3",
            "--max-inference-ms",
            "100",
        ]
    ) == 0
    assert inference_arguments["warmup_runs"] == 2
    assert inference_arguments["timed_runs"] == 3
    assert inference_arguments["max_inference_ms"] == 100.0
    assert capsys.readouterr().out.count('"value": "ok"') == 3


def test_cli_synthesizes_pipeline_validation_episodes(monkeypatch, capsys):
    from excavator_il import synthetic_episodes

    monkeypatch.setattr(
        synthetic_episodes,
        "synthesize_episodes",
        lambda *args, **kwargs: _Result(),
    )

    assert main(
        [
            "synthesize-episodes",
            "episode_0004",
            "--output-root",
            "data/raw/synthetic",
            "--count",
            "10",
        ]
    ) == 0
    assert '"value": "ok"' in capsys.readouterr().out

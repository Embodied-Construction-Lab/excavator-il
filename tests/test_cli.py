import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from excavator_il.cli import main


@dataclass(frozen=True)
class _Result:
    value: str = "ok"


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
    from excavator_il import episode_builder, teleop
    from excavator_il.collector import client, config, service

    calls = []
    monkeypatch.setattr(teleop.TeleopConfig, "load", lambda path: f"teleop:{path}")
    monkeypatch.setattr(
        teleop, "run_teleop", lambda loaded, print_every: calls.append((loaded, print_every))
    )
    monkeypatch.setattr(teleop, "list_pygame_devices", lambda: [{"device_id": "one"}])
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
    assert calls == [("teleop:teleop.json", 7), ("collect", "collection.json")]
    assert "device_id" in capsys.readouterr().out


def test_cli_dispatches_optional_training_commands(monkeypatch, capsys):
    pytest.importorskip("lerobot")
    from excavator_il import act_smoke, lerobot_conversion

    monkeypatch.setattr(lerobot_conversion, "convert_episodes", lambda *a, **k: _Result())
    monkeypatch.setattr(act_smoke, "run_act_smoke_train_step", lambda **k: _Result())

    assert main(["convert", "ep", "--output-root", "out"]) == 0
    assert main(["smoke-train"]) == 0
    assert capsys.readouterr().out.count('"value": "ok"') == 2

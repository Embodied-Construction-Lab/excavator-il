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


@dataclass(frozen=True)
class _CheckpointResult:
    value: str = "ok"
    selected_checkpoint: str = "checkpoint-a"


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
    episode_requests = []
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
        lambda path, request: episode_requests.append(dict(request))
        or {"ok": True, "path": str(path), "request": request},
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
            "--task-variant",
            "dig_transport_dump",
            "--soil-reset-block-id",
            "block_07",
            "--dig-point-id",
            "dig_03",
            "--collection-zone-id",
            "zone_06",
            "--dig-repeat-index",
            "3",
            "--operator-note",
            "远排右侧第三次",
            "--recording-purpose",
            "diagnostic",
            "--target-source-provenance-json",
            json.dumps(
                {
                    "repository": "airylidar",
                    "path": "mission/config/excavation_demo.json",
                    "sha256": "a" * 64,
                    "commit": "b" * 40,
                    "dirty": False,
                }
            ),
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
    assert episode_requests[0] == {
        "command": "start",
        "task": "ExecuteDig",
        "operator_id": "operator_01",
        "dig_target_m": [1.0, 2.0, 3.0],
        "material_id": "soil",
        "task_variant": "dig_transport_dump",
        "soil_reset_block_id": "block_07",
        "dig_point_id": "dig_03",
        "collection_zone_id": "zone_06",
        "dig_repeat_index": 3,
        "operator_note": "远排右侧第三次",
        "recording_purpose": "diagnostic",
        "target_source_provenance": {
            "repository": "airylidar",
            "path": "mission/config/excavation_demo.json",
            "sha256": "a" * 64,
            "commit": "b" * 40,
            "dirty": False,
        },
    }
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


def test_act_runtime_cli_defaults_to_shadow_and_passes_exact_motion_authorization(
    monkeypatch,
):
    from excavator_il import cli
    from excavator_il import act_runtime_service

    calls = []
    logging_calls = []
    monkeypatch.setattr(
        cli.logging,
        "basicConfig",
        lambda **kwargs: logging_calls.append(kwargs),
    )
    monkeypatch.setattr(
        act_runtime_service,
        "run_act_runtime",
        lambda path, motion_authorization=None, max_steps=None,
        hardware_start_gate=None, operator_observation_config=None: calls.append(
            (
                path,
                motion_authorization,
                max_steps,
                hardware_start_gate,
                operator_observation_config,
            )
        ),
    )

    assert main(["act-runtime", "--config", "runtime.json"]) == 0
    assert main(
        [
            "act-runtime",
            "--config",
            "runtime.json",
            "--motion-authorization",
            "ALLOW_ACT_MACHINE_MOTION",
            "--hardware-start-gate",
            "/opt/act-control/hybrid_001.start",
        ]
    ) == 0
    assert calls == [
        ("runtime.json", None, None, None, None),
        (
            "runtime.json",
            "ALLOW_ACT_MACHINE_MOTION",
            None,
            "/opt/act-control/hybrid_001.start",
            None,
        ),
    ]
    assert all(call["force"] is True for call in logging_calls)


def test_camera_preview_cli_dispatches_collection_config(monkeypatch):
    from excavator_il import camera_preview_service

    calls = []
    monkeypatch.setattr(
        camera_preview_service,
        "run_camera_preview",
        lambda path: calls.append(path),
    )

    assert main(["camera-preview", "--config", "collection.json"]) == 0
    assert calls == ["collection.json"]


def test_record_collection_run_cli_prints_stable_json(monkeypatch, capsys):
    from excavator_il import collection_experiment_run

    calls = []

    class _Config:
        def request_for_episode(self, episode):
            calls.append(("request", episode))
            return "request"

    monkeypatch.setattr(
        collection_experiment_run,
        "load_collection_experiment_run_config",
        lambda path: calls.append(("config", path)) or _Config(),
    )
    monkeypatch.setattr(
        collection_experiment_run,
        "record_collection_experiment_run",
        lambda request: calls.append(("record", request))
        or SimpleNamespace(
            run_id="collection_episode_0042",
            state="success",
            run_dir="/evidence/runs/collection_episode_0042",
            artifacts=({"artifact_id": "raw_episode"}, {"artifact_id": "quality_report"}),
            start={"task_context": {"task_variant": "dig_only"}},
            final={"metrics": {"episode_id": "episode_0042"}},
        ),
    )

    exit_code = main(
        [
            "record-collection-run",
            "/data/raw/episode_0042",
            "--config",
            "config/evidence.json",
        ]
    )

    assert exit_code == 0
    assert calls == [
        ("config", "config/evidence.json"),
        ("request", "/data/raw/episode_0042"),
        ("record", "request"),
    ]
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "collection_experiment_run_result.v1",
        "run_id": "collection_episode_0042",
        "state": "success",
        "run_dir": "/evidence/runs/collection_episode_0042",
        "episode_id": "episode_0042",
        "task_context": {"task_variant": "dig_only"},
        "artifact_count": 2,
    }


def test_record_collection_run_cli_returns_nonzero_on_evidence_failure(
    monkeypatch, capsys
):
    from excavator_il import collection_experiment_run

    monkeypatch.setattr(
        collection_experiment_run,
        "load_collection_experiment_run_config",
        lambda _path: (_ for _ in ()).throw(ValueError("evidence drift")),
    )

    assert main(["record-collection-run", "episode_0001"]) == 2
    assert "evidence drift" in capsys.readouterr().err


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


def test_teleop_handles_operator_interrupt_without_traceback(monkeypatch, capsys):
    from excavator_il import teleop

    monkeypatch.setattr(teleop.TeleopConfig, "load", lambda path: f"teleop:{path}")

    def interrupt(unused_config, *, print_every):
        raise KeyboardInterrupt

    monkeypatch.setattr(teleop, "run_teleop", interrupt)

    assert main(["teleop"]) == 130
    assert "teleop interrupted" in capsys.readouterr().err


def test_cli_dispatches_optional_training_commands(monkeypatch, capsys):
    pytest.importorskip("lerobot")
    from excavator_il import (
        action_dataset_transform,
        act_smoke,
        checkpoint_evaluation,
        lerobot_conversion,
        training_split,
    )

    inference_arguments = {}

    def infer(**kwargs):
        inference_arguments.update(kwargs)
        return _Result()

    conversion_arguments = []
    split_arguments = []
    monkeypatch.setattr(
        lerobot_conversion,
        "convert_episodes",
        lambda *args, **kwargs: conversion_arguments.append((args, kwargs))
        or _Result(),
    )
    monkeypatch.setattr(act_smoke, "run_act_smoke_train_step", lambda **k: _Result())
    monkeypatch.setattr(act_smoke, "run_act_checkpoint_inference", infer)
    monkeypatch.setattr(
        training_split,
        "prepare_training_split",
        lambda **kwargs: split_arguments.append(kwargs) or _Result(),
    )
    monkeypatch.setattr(training_split, "materialize_training_split", lambda **k: _Result())
    monkeypatch.setattr(
        action_dataset_transform, "derive_zero_swing_split", lambda **k: _Result()
    )
    monkeypatch.setattr(
        checkpoint_evaluation, "evaluate_act_checkpoints", lambda **k: _CheckpointResult()
    )

    assert main(["convert", "ep", "--output-root", "out"]) == 0
    assert main(
        [
            "convert",
            "ep",
            "--output-root",
            "out-front",
            "--camera-roles",
            "front",
        ]
    ) == 0
    assert main(
        [
            "convert",
            "ep",
            "--output-root",
            "out-task-override",
            "--task-variant-override",
            "dig_transport_dump",
        ]
    ) == 0
    assert conversion_arguments[0][1]["camera_roles"] is None
    assert conversion_arguments[1][1]["camera_roles"] == ("front",)
    assert (
        conversion_arguments[2][1]["task_variant_override"]
        == "dig_transport_dump"
    )
    assert main(
        [
            "prepare-training-split",
            "--dataset-root",
            "dataset",
            "--repo-id",
            "local/dataset",
            "--output",
            "training_split.json",
            "--train-ratio",
            "0.8",
            "--seed",
            "7",
            "--grouping",
            "episode",
        ]
    ) == 0
    assert split_arguments[0]["grouping"] == "episode"
    assert main(
        [
            "derive-zero-swing-split",
            "--source-root",
            "data/lerobot/source_split",
            "--output-root",
            "data/lerobot/swing_zero_split",
            "--repo-suffix",
            "swing_zero",
        ]
    ) == 0
    assert main(
        [
            "evaluate-checkpoints",
            "checkpoint-a",
            "checkpoint-b",
            "--split-root",
            "data/lerobot/split",
            "--device",
            "cuda",
        ]
    ) == 0
    assert main(
        [
            "materialize-training-split",
            "--manifest",
            "training_split.json",
            "--output-root",
            "data/lerobot/splits",
        ]
    ) == 0
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
    assert capsys.readouterr().out.count('"value": "ok"') == 9


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

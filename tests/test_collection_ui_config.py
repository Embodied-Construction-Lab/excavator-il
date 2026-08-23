import json
from pathlib import Path

import pytest

from excavator_il.collection_ui_config import load_collection_ui_config
from excavator_il.hybrid_experiment_run import (
    HybridExperimentRunConfig,
    HybridExperimentRunFactory,
)


def test_collection_ui_config_is_local_only_and_resolves_guided_config(tmp_path):
    path = tmp_path / "collection-ui.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_collection_ui_config.v2",
                "guided_config": "guided.json",
                "hybrid_mission_config": "hybrid.json",
                "hybrid_evidence_config": "hybrid-evidence.json",
                "server": {"host": "127.0.0.1", "port": 8088},
                "camera_preview_url": (
                    "http://192.168.50.2:18092/camera/front.mjpg"
                ),
                "camera_dump_preview_url": (
                    "http://192.168.50.2:18092/camera/dump.mjpg"
                ),
                "telemetry_url": "http://192.168.50.2:18092/telemetry/latest.json",
                "visualization_url": "",
            }
        ),
        encoding="utf-8",
    )

    config = load_collection_ui_config(path)

    assert config.guided_config == tmp_path / "guided.json"
    assert config.hybrid_mission_config == tmp_path / "hybrid.json"
    assert config.hybrid_evidence_config == tmp_path / "hybrid-evidence.json"
    assert config.host == "127.0.0.1"
    assert config.port == 8088
    assert config.camera_preview_url.endswith("/camera/front.mjpg")
    assert config.camera_dump_preview_url.endswith("/camera/dump.mjpg")
    assert config.telemetry_url.endswith("/telemetry/latest.json")
    assert config.visualization_url == ""


def test_collection_ui_config_rejects_non_loopback_server(tmp_path):
    path = tmp_path / "collection-ui.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_collection_ui_config.v1",
                "guided_config": "guided.json",
                "server": {"host": "0.0.0.0", "port": 8088},
                "camera_preview_url": (
                    "http://192.168.50.2:18092/camera/front.mjpg"
                ),
                "telemetry_url": "http://192.168.50.2:18092/telemetry/latest.json",
                "visualization_url": "",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="loopback"):
        load_collection_ui_config(path)


def test_collection_ui_v1_remains_compatible_without_hybrid_evidence(tmp_path):
    path = tmp_path / "collection-ui.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_collection_ui_config.v1",
                "guided_config": "guided.json",
                "server": {"host": "127.0.0.1", "port": 8088},
                "camera_preview_url": "http://192.168.50.2:18092/camera/front.mjpg",
                "telemetry_url": "http://192.168.50.2:18092/telemetry/latest.json",
                "visualization_url": "",
            }
        ),
        encoding="utf-8",
    )

    config = load_collection_ui_config(path)

    assert config.hybrid_evidence_config is None


def test_commissioned_ui_config_binds_preflighted_hybrid_evidence():
    config_dir = Path(__file__).parents[1] / "config"

    ui_config = load_collection_ui_config(config_dir / "collection_ui.pc.json")
    assert ui_config.hybrid_evidence_config == (
        config_dir / "hybrid_evidence.pc.json"
    ).resolve()

    evidence = HybridExperimentRunConfig.load(ui_config.hybrid_evidence_config)
    HybridExperimentRunFactory(evidence).preflight()

    assert evidence.policy_ids == {
        "dig_policy": "lerobot_act:swing_zero_seed2026_step200000",
        "trajectory_controller": "onnx_rl:scale_v3_deadzone_reward_03_p003",
    }

import json
from pathlib import Path

from excavator_il.collection_ui_config import load_collection_ui_config
from excavator_il.resident_fixed_cycle_system import ResidentFixedCyclePcConfig


def test_fixed_dig_ui_config_selects_parallel_commissioning_paths():
    config_dir = Path(__file__).parents[1] / "config"

    ui_config = load_collection_ui_config(
        config_dir / "collection_ui.fixed_dig.commissioning.pc.json"
    )

    assert ui_config.resident_fixed_cycle_config == (
        config_dir / "resident_fixed_cycle.fixed_dig.commissioning.pc.json"
    ).resolve()
    assert ui_config.hybrid_evidence_config == (
        config_dir / "hybrid_evidence.fixed_dig.commissioning.pc.json"
    ).resolve()


def test_fixed_dig_resident_config_uses_dedicated_edge_runtime():
    config_dir = Path(__file__).parents[1] / "config"

    config = ResidentFixedCyclePcConfig.load(
        config_dir / "resident_fixed_cycle.fixed_dig.commissioning.pc.json"
    )

    assert config.expected_mission_id == "fixed_dig_hybrid"
    assert config.expected_act_worker_required is False
    assert config.act_runtime_config is None
    assert config.act_checkpoint_host_path is None
    assert config.act_deployment_host_path is None
    assert config.trajectory_controller_backend == "onnx_rl"
    assert str(config.edge_runtime_config) == (
        "deploy/edge_runtime.resident.fixed_dig.commissioning.json"
    )
    assert str(config.fixed_cycle_plan) == (
        "/home/jetson16/workspace_excavator/excavator-orin-runtime/"
        "deploy/v3b/fixed-dig/catalog/candidate/fixed_cycle.candidate.json"
    )


def test_fixed_dig_evidence_config_declares_fixed_action_and_rl_artifacts():
    config_dir = Path(__file__).parents[1] / "config"
    document = json.loads(
        (config_dir / "hybrid_evidence.fixed_dig.commissioning.pc.json").read_text(
            encoding="utf-8"
        )
    )

    assert document["policy_ids"] == {
        "dig_policy": (
            "fixed_action:icra2027_fixed_dig_fixed_dump_commissioning_v1"
        ),
        "trajectory_controller": "onnx_rl:scale_v3_deadzone_reward_03_p003",
    }
    assert document["config_paths"]["edge_runtime"] == (
        "../../excavator-orin-runtime/deploy/"
        "edge_runtime.resident.fixed_dig.commissioning.json"
    )
    assert document["task_context"]["task_variant"] == "dig_transport_dump"
    assert {item["role"] for item in document["artifacts"]} == {
        "fixed_action_profile",
        "rl_onnx_model",
    }

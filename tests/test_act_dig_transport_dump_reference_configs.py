import json
from pathlib import Path

from excavator_il.act_runtime_config import load_act_runtime_config
from excavator_il.collection_ui_config import load_collection_ui_config
from excavator_il.dig_point_catalog import load_dig_point_catalog
from excavator_il.hybrid_experiment_run import HybridExperimentRunConfig
from excavator_il.resident_fixed_cycle_system import ResidentFixedCyclePcConfig


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "config"
ORIN_REPOSITORY = REPOSITORY.parent / "excavator-orin-runtime"
AIRY_REPOSITORY = REPOSITORY.parent / "AiryLidar"


def test_act_dig_transport_dump_reference_configs_are_explicitly_engineering_only():
    runtime = load_act_runtime_config(
        CONFIG / "act_runtime.icra2027_transport_dump_dual_rgb.orin.json"
    )
    cycle = ResidentFixedCyclePcConfig.load(
        CONFIG
        / "resident_fixed_cycle.act_dig_transport_dump_reference.commissioning.pc.json"
    )
    evidence = HybridExperimentRunConfig.load(
        CONFIG / "hybrid_evidence.act_dig_transport_dump_reference.pc.json"
    )
    ui = load_collection_ui_config(
        CONFIG
        / "collection_ui.act_dig_transport_dump_reference.commissioning.pc.json"
    )

    assert runtime.camera_roles == ("front", "dump")
    assert runtime.checkpoint_model_sha256 == (
        "54a3ba90e6c2186787b8b7eb1b9e5211e2bcf81e41551e866283ace41ed04f4a"
    )
    assert cycle.expected_mission_id == "engineering_act_transport_reference"
    assert cycle.expected_act_worker_required is True
    assert cycle.act_max_steps == 260
    assert str(cycle.act_deployment_host_path).endswith(
        "models/icra2027_transport_dump_dual_rgb_step115000/deployment"
    )
    assert "act-dig-transport-dump-reference" in str(cycle.runtime_root)
    assert "deploy/v3b/act-dig-transport-dump-reference/catalog/candidate" in str(
        cycle.fixed_cycle_plan
    )
    assert str(cycle.dig_point_catalog).endswith(
        "mission/config/excavation_dig_point_catalog.v1.json"
    )
    plan = json.loads(
        (
            ORIN_REPOSITORY
            / "deploy/v3b/act-dig-transport-dump-reference/catalog/candidate/"
            "fixed_cycle.candidate.json"
        ).read_text(encoding="utf-8")
    )
    assert plan["schema_version"] == "resident_fixed_cycle_plan.v5"
    assert plan["mission"]["mission_id"] == cycle.expected_mission_id
    assert [
        behavior["behavior_id"]
        for behavior in plan["mission"]["cycle_behaviors"]
    ] == ["act_dig_transport_dump", "onnx_rl_tracking"]
    catalog = load_dig_point_catalog(
        AIRY_REPOSITORY
        / "mission/config/excavation_dig_point_catalog.v1.json"
    )
    assert plan["dig_sequence"] == list(catalog.points)
    assert plan["dig_groups"] == {
        group_id: list(point_ids)
        for group_id, point_ids in catalog.groups.items()
    }
    assert evidence.policy_ids["dig_policy"].endswith("step115000")
    assert evidence.config_paths["edge_runtime"] == (
        ORIN_REPOSITORY / "deploy/edge_runtime.resident.remote.json"
    ).resolve()
    assert "experiment_profile" not in evidence.config_paths
    assert evidence.artifacts[0].metadata["mission_id"] == (
        "engineering_act_transport_reference"
    )
    assert ui.resident_fixed_cycle_config == (
        CONFIG
        / "resident_fixed_cycle.act_dig_transport_dump_reference.commissioning.pc.json"
    ).resolve()


def test_active_v3b_config_selects_the_fixed_target_hybrid_mission():
    active = ResidentFixedCyclePcConfig.load(CONFIG / "resident_fixed_cycle.pc.json")

    assert active.expected_mission_id == "fixed_target_hybrid"
    assert active.expected_act_worker_required is True
    assert active.expected_act_behavior_id == "act_dig_lift"
    assert active.expected_act_model_sha256 == (
        "742a07ad6175af60ab0f14e4cdf409b790b35aad5a33f1e26d3d378952b7a475"
    )
    assert str(active.act_runtime_config).endswith("config/act_runtime.orin.json")
    assert active.act_checkpoint_host_path is not None
    assert active.act_deployment_host_path is not None

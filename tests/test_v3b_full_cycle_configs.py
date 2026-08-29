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


def test_v3b_full_cycle_configs_form_an_isolated_parallel_profile():
    runtime = load_act_runtime_config(
        CONFIG / "act_runtime.icra2027_transport_dump_dual_rgb.orin.json"
    )
    cycle = ResidentFixedCyclePcConfig.load(
        CONFIG / "resident_fixed_cycle.full_cycle.commissioning.pc.json"
    )
    evidence = HybridExperimentRunConfig.load(
        CONFIG / "hybrid_evidence.full_cycle.pc.json"
    )
    ui = load_collection_ui_config(
        CONFIG / "collection_ui.v3b-full-cycle-commissioning.pc.json"
    )

    assert runtime.camera_roles == ("front", "dump")
    assert runtime.checkpoint_model_sha256 == (
        "54a3ba90e6c2186787b8b7eb1b9e5211e2bcf81e41551e866283ace41ed04f4a"
    )
    assert cycle.mission_profile == "act_full_cycle"
    assert cycle.act_max_steps == 260
    assert str(cycle.act_deployment_host_path).endswith(
        "models/icra2027_transport_dump_dual_rgb_step115000/deployment"
    )
    assert "v3b-full-cycle" in str(cycle.runtime_root)
    assert "deploy/v3b/act-full-cycle/catalog/candidate" in str(
        cycle.fixed_cycle_plan
    )
    assert str(cycle.dig_point_catalog).endswith(
        "mission/config/excavation_dig_point_catalog.v1.json"
    )
    plan = json.loads(
        (
            ORIN_REPOSITORY
            / "deploy/v3b/act-full-cycle/catalog/candidate/fixed_cycle.candidate.json"
        ).read_text(encoding="utf-8")
    )
    assert plan["schema_version"] == "resident_fixed_cycle_plan.v4"
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
    assert evidence.artifacts[0].metadata["mission_profile"] == "act_full_cycle"
    assert ui.resident_fixed_cycle_config == (
        CONFIG / "resident_fixed_cycle.full_cycle.commissioning.pc.json"
    ).resolve()


def test_active_v3b_config_keeps_the_regime_factorized_profile():
    active = ResidentFixedCyclePcConfig.load(CONFIG / "resident_fixed_cycle.pc.json")

    assert active.mission_profile == "regime_factorized"
    assert active.act_runtime_config is None
    assert active.act_checkpoint_host_path is None
    assert active.act_deployment_host_path is None

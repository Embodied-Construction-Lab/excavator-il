import json
from pathlib import Path

from excavator_il.act_runtime_config import load_act_runtime_config
from excavator_il.collection_ui_config import load_collection_ui_config
from excavator_il.hybrid_experiment_run import HybridExperimentRunConfig
from excavator_il.resident_fixed_cycle_system import ResidentFixedCyclePcConfig


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "config"
ORIN_REPOSITORY = REPOSITORY.parent / "excavator-orin-runtime"
MODEL_SHA256 = "8faf364022695312714ccbac3299635e8fc8ee30b1935b0f3122b1c1fdedd637"


def test_three_phase_reference_is_a_separate_strictly_bound_runtime():
    runtime = load_act_runtime_config(
        CONFIG / "act_runtime.icra2027_transport_dump_three_phase.orin.json"
    )
    cycle = ResidentFixedCyclePcConfig.load(
        CONFIG
        / (
            "resident_fixed_cycle.act_dig_transport_dump_three_phase_reference."
            "commissioning.pc.json"
        )
    )
    evidence = HybridExperimentRunConfig.load(
        CONFIG / "hybrid_evidence.act_dig_transport_dump_three_phase_reference.pc.json"
    )
    ui = load_collection_ui_config(
        CONFIG
        / (
            "collection_ui.act_dig_transport_dump_three_phase_reference."
            "commissioning.pc.json"
        )
    )

    assert runtime.act_behavior_id == "act_dig_transport_dump_three_phase"
    assert runtime.checkpoint_model_sha256 == MODEL_SHA256
    assert runtime.camera_roles == ("front", "dump")
    assert runtime.phase_schedule is not None
    assert runtime.phase_schedule.dig_to_transport_step == 93
    assert runtime.phase_schedule.transport_to_dump_step == 142
    assert len(runtime.policy_state_fields) == 14

    assert (
        cycle.expected_mission_id
        == "engineering_act_transport_three_phase_reference"
    )
    assert cycle.expected_act_behavior_id == runtime.act_behavior_id
    assert cycle.expected_act_model_sha256 == MODEL_SHA256
    assert cycle.expected_act_worker_required is True
    assert cycle.act_max_steps == 241

    plan = json.loads(
        (
            ORIN_REPOSITORY
            / (
                "deploy/v3b/act-dig-transport-dump-three-phase-reference/"
                "catalog/candidate/fixed_cycle.candidate.json"
            )
        ).read_text(encoding="utf-8")
    )
    assert plan["mission"]["mission_id"] == cycle.expected_mission_id
    assert plan["mission_sha256"] == cycle.expected_mission_sha256
    assert [
        behavior["behavior_id"]
        for behavior in plan["mission"]["cycle_behaviors"]
    ] == ["act_dig_transport_dump_three_phase", "onnx_rl_tracking"]
    assert plan["mission"]["cycle_behaviors"][0]["max_steps"] == 241

    assert evidence.mission_id == cycle.expected_mission_id
    assert evidence.mission_sha256 == cycle.expected_mission_sha256
    assert evidence.policy_ids["dig_policy"].endswith("step140000")
    assert "experiment_profile" not in evidence.config_paths
    assert ui.resident_fixed_cycle_config == (
        CONFIG
        / (
            "resident_fixed_cycle.act_dig_transport_dump_three_phase_reference."
            "commissioning.pc.json"
        )
    ).resolve()


def test_three_phase_reference_does_not_replace_the_default_mainline():
    mainline = ResidentFixedCyclePcConfig.load(
        CONFIG / "resident_fixed_cycle.pc.json"
    )

    assert mainline.expected_mission_id == "fixed_target_hybrid"
    assert mainline.expected_act_behavior_id == "act_dig_lift"
    assert mainline.expected_act_model_sha256 != MODEL_SHA256
    assert str(mainline.act_runtime_config).endswith("config/act_runtime.orin.json")

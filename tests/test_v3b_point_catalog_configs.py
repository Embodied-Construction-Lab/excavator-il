import json
from pathlib import Path

from excavator_il.collection_ui_config import load_collection_ui_config
from excavator_il.dig_point_catalog import load_dig_point_catalog
from excavator_il.resident_fixed_cycle_system import ResidentFixedCyclePcConfig


ROOT = Path(__file__).resolve().parents[1]
AIRY_ROOT = ROOT.parent / "AiryLidar"
ORIN_ROOT = ROOT.parent / "excavator-orin-runtime"


def test_active_v3b_runtime_uses_catalog_driven_points_and_groups():
    pc_path = ROOT / "config/resident_fixed_cycle.pc.json"
    ui_path = ROOT / "config/collection_ui.pc.json"
    catalog_path = (
        AIRY_ROOT / "mission/config/excavation_dig_point_catalog.v1.json"
    )
    mission_path = AIRY_ROOT / "mission/config/excavation_cycle.json"
    plan_path = (
        ORIN_ROOT
        / "deploy/v3b/catalog/candidate/fixed_cycle.candidate.json"
    )

    config = ResidentFixedCyclePcConfig.load(pc_path)
    ui_config = load_collection_ui_config(ui_path)
    catalog = load_dig_point_catalog(catalog_path)
    mission = json.loads(mission_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    target_catalog = json.loads(
        (plan_path.parent / "target_catalog.candidate.json").read_text(
            encoding="utf-8"
        )
    )

    assert ui_config.resident_fixed_cycle_config == pc_path
    assert config.dig_point_catalog == Path(
        "mission/config/excavation_dig_point_catalog.v1.json"
    )
    assert config.fixed_cycle_plan.as_posix().endswith(
        "/deploy/v3b/catalog/candidate/fixed_cycle.candidate.json"
    )
    assert catalog.default_group_id == "all"
    assert catalog.groups["all"] == tuple(catalog.points)
    assert plan["schema_version"] == "resident_fixed_cycle_plan.v5"
    assert plan["mission"]["mission_id"] == config.expected_mission_id
    assert plan["validation_status"] == "candidate"
    assert mission["limits"]["waypoint_tolerance_m"] == 0.25
    assert target_catalog["waypoint_tolerance_m"] == 0.25
    assert target_catalog["intermediate_waypoint_tolerance_m"] == 0.40
    assert plan["dig_sequence"] == list(catalog.points)
    assert plan["dig_groups"] == {
        group_id: list(point_ids)
        for group_id, point_ids in catalog.groups.items()
    }
    assert set(path.name for path in plan_path.parent.iterdir()) == {
        "fixed_cycle.candidate.json",
        "target_catalog.candidate.json",
    }

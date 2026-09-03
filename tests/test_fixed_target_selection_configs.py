from pathlib import Path

from excavator_il.collection_ui_config import load_collection_ui_config


def test_fixed_target_selection_ui_binds_profile_evidence_without_changing_default():
    config_dir = Path(__file__).parents[1] / "config"

    experiment = load_collection_ui_config(
        config_dir
        / "collection_ui.fixed_target_selection.commissioning.pc.json"
    )
    default = load_collection_ui_config(config_dir / "collection_ui.pc.json")

    assert experiment.resident_fixed_cycle_config == (
        config_dir / "resident_fixed_cycle.pc.json"
    ).resolve()
    assert experiment.hybrid_evidence_config == (
        config_dir
        / "hybrid_evidence.fixed_target_selection.commissioning.pc.json"
    ).resolve()
    assert default.hybrid_evidence_config != experiment.hybrid_evidence_config

import json
from pathlib import Path

import pytest

from excavator_il.icra2027_experiment_profile import (
    Icra2027ExperimentProfile,
    Icra2027ExperimentSuite,
    main,
)
from excavator_il.hybrid_experiment_run import HybridExperimentRunConfig


PROJECT_ROOT = Path(__file__).parents[1]
ACTIVE_SUITE = PROJECT_ROOT / "config/experiments/icra2027"
ACTIVE_EVIDENCE = (
    PROJECT_ROOT
    / "config/hybrid_evidence.fixed_target_selection.commissioning.pc.json"
)
ACTIVE_RESIDENT = PROJECT_ROOT / "config/resident_fixed_cycle.pc.json"
ACTIVE_UI = (
    PROJECT_ROOT
    / "config/collection_ui.fixed_target_selection.commissioning.pc.json"
)
ACTIVE_CLASSICAL_PROFILE = (
    PROJECT_ROOT / "config/experiments/icra2027/classical_tracking.json"
)
ACTIVE_CLASSICAL_RESIDENT = (
    PROJECT_ROOT
    / "config/resident_fixed_cycle.classical_tracking.commissioning.pc.json"
)
ACTIVE_CLASSICAL_EVIDENCE = (
    PROJECT_ROOT
    / "config/hybrid_evidence.classical_tracking.commissioning.pc.json"
)
ACTIVE_CLASSICAL_UI = (
    PROJECT_ROOT
    / "config/collection_ui.classical_tracking.commissioning.pc.json"
)
ACTIVE_FIXED_DIG_RESIDENT = (
    PROJECT_ROOT
    / "config/resident_fixed_cycle.fixed_dig.commissioning.pc.json"
)
ACTIVE_FIXED_DIG_EVIDENCE = (
    PROJECT_ROOT
    / "config/hybrid_evidence.fixed_dig.commissioning.pc.json"
)
ACTIVE_FIXED_DIG_UI = (
    PROJECT_ROOT / "config/collection_ui.fixed_dig.commissioning.pc.json"
)
ORIN_REPOSITORY = PROJECT_ROOT.parent / "excavator-orin-runtime"
ACTIVE_ONNX_EDGE_RUNTIME = (
    ORIN_REPOSITORY / "deploy/edge_runtime.resident.remote.json"
)
ACTIVE_CARTESIAN_EDGE_RUNTIME = (
    ORIN_REPOSITORY
    / "deploy/edge_runtime.resident.cartesian_p.commissioning.json"
)
MISSION_SHA256 = {
    "fixed_target_hybrid": "3a5c7edd6a228863e3d5eefe3228173848756a46e9ce441da53cc2b0c164d786",
    "classical_tracking_hybrid": "629ecaa1dcff9b17c8b5497d7fae7dc8e0223b0eb1c1c5d8aec22465eea1e1a7",
    "fixed_dig_hybrid": "a52462867a4c81e28e623d02c736b3f6e91cec62cbb71fa0505ad6035ad80101",
}


def _profile_payload() -> dict[str, object]:
    return {
        "schema_version": "excavator_icra2027_experiment_profile.v2",
        "profile_id": "fixed_target_selection",
        "expected_mission_id": "fixed_target_hybrid",
        "expected_mission_sha256": "3a5c7edd6a228863e3d5eefe3228173848756a46e9ce441da53cc2b0c164d786",
        "study_kind": "live_mission",
        "readiness": "ready",
        "reference_profile_id": "proposed_hybrid",
        "isolated_factor": "target_selection",
        "method_factors": {
            "software_architecture": "regime_factorized",
            "target_selection": "fixed_catalog",
            "trajectory_tracking": "tc_btf",
            "task_policy": "act_dig_lift_fixed_dump",
        },
        "bindings": {
            "collection_ui_config": "../../collection_ui.pc.json",
            "resident_fixed_cycle_config": "../../resident_fixed_cycle.pc.json",
            "hybrid_evidence_config": "../../hybrid_evidence.pc.json",
        },
        "required_metrics": [
            "cycle_success_rate",
            "cycle_duration_s",
            "payload_mass_kg",
            "intervention_count",
        ],
    }


def test_active_icra_suite_excludes_the_act_transport_engineering_reference():
    suite = Icra2027ExperimentSuite.load_directory(ACTIVE_SUITE)

    assert "act_full_cycle" not in suite.profiles
    assert "act_dig_transport_dump_reference" not in suite.profiles


def test_profile_loads_strict_frozen_method_and_resolves_bindings(tmp_path: Path):
    path = tmp_path / "config/experiments/icra2027/fixed.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_profile_payload()), encoding="utf-8")

    profile = Icra2027ExperimentProfile.load(path)

    assert profile.profile_id == "fixed_target_selection"
    assert profile.expected_mission_id == "fixed_target_hybrid"
    assert profile.method_factors["target_selection"] == "fixed_catalog"
    assert profile.bindings["resident_fixed_cycle_config"] == (
        path.parent / "../../resident_fixed_cycle.pc.json"
    ).resolve()
    assert profile.required_metrics == (
        "cycle_success_rate",
        "cycle_duration_s",
        "payload_mass_kg",
        "intervention_count",
    )
    with pytest.raises(TypeError):
        profile.method_factors["target_selection"] = "tadps"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("unexpected", True), "must contain exactly"),
        (("study_kind", "demo"), "study_kind must be one of"),
        (("readiness", "maybe"), "readiness must be one of"),
    ],
)
def test_profile_rejects_unknown_fields_and_uncontrolled_vocabulary(
    tmp_path: Path,
    mutation: tuple[str, object],
    message: str,
):
    payload = _profile_payload()
    payload[mutation[0]] = mutation[1]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        Icra2027ExperimentProfile.load(path)


@pytest.mark.parametrize(
    ("factor", "value"),
    [
        ("software_architecture", "ad_hoc_threads"),
        ("target_selection", "operator_guess"),
        ("trajectory_tracking", "unknown_controller"),
        ("task_policy", "unversioned_script"),
    ],
)
def test_profile_rejects_unknown_method_factor_levels(
    tmp_path: Path,
    factor: str,
    value: str,
):
    payload = _profile_payload()
    payload["method_factors"][factor] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=f"method_factors.{factor} must be one of"):
        Icra2027ExperimentProfile.load(path)


def test_planned_profile_can_describe_missing_runtime_without_becoming_runnable(
    tmp_path: Path,
):
    payload = _profile_payload()
    payload["readiness"] = "planned"
    payload["bindings"] = {
        "collection_ui_config": None,
        "resident_fixed_cycle_config": None,
        "hybrid_evidence_config": None,
    }
    path = tmp_path / "planned.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = Icra2027ExperimentProfile.load(path)

    assert profile.bindings["resident_fixed_cycle_config"] is None
    with pytest.raises(ValueError, match="profile is planned, not runnable"):
        profile.preflight()


def test_suite_accepts_an_ablation_that_changes_only_its_declared_factor(
    tmp_path: Path,
):
    root = tmp_path / "profiles"
    root.mkdir()
    proposed = _profile_payload()
    proposed.update(
        {
            "profile_id": "proposed_hybrid",
            "reference_profile_id": None,
            "isolated_factor": None,
        }
    )
    proposed["method_factors"]["target_selection"] = "tadps"
    (root / "proposed_hybrid.json").write_text(
        json.dumps(proposed), encoding="utf-8"
    )
    fixed = _profile_payload()
    (root / "fixed_target_selection.json").write_text(
        json.dumps(fixed), encoding="utf-8"
    )

    suite = Icra2027ExperimentSuite.load_directory(root)

    assert tuple(suite.profiles) == (
        "fixed_target_selection",
        "proposed_hybrid",
    )
    assert suite.profiles["fixed_target_selection"].isolated_factor == (
        "target_selection"
    )


def test_suite_rejects_an_ablation_with_an_undeclared_second_change(tmp_path: Path):
    root = tmp_path / "profiles"
    root.mkdir()
    proposed = _profile_payload()
    proposed.update(
        {
            "profile_id": "proposed_hybrid",
            "reference_profile_id": None,
            "isolated_factor": None,
        }
    )
    proposed["method_factors"]["target_selection"] = "tadps"
    (root / "proposed_hybrid.json").write_text(
        json.dumps(proposed), encoding="utf-8"
    )
    confounded = _profile_payload()
    confounded["method_factors"]["trajectory_tracking"] = "cartesian_p"
    (root / "fixed_target_selection.json").write_text(
        json.dumps(confounded), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="changes factors.*target_selection"):
        Icra2027ExperimentSuite.load_directory(root)


def test_active_suite_declares_paper_conditions_without_engineering_reference():
    suite = Icra2027ExperimentSuite.load_directory(ACTIVE_SUITE)

    assert tuple(suite.profiles) == (
        "classical_tracking",
        "fixed_dig",
        "fixed_target_selection",
        "proposed_hybrid",
        "tadps",
    )
    assert {
        profile_id
        for profile_id, profile in suite.profiles.items()
        if profile.readiness == "ready"
    } == set()
    assert {
        profile_id
        for profile_id, profile in suite.profiles.items()
        if profile.readiness == "commissioning"
    } == {
        "classical_tracking",
        "fixed_dig",
        "fixed_target_selection",
    }
    assert {
        profile_id: profile.expected_mission_id
        for profile_id, profile in suite.profiles.items()
        if profile.readiness == "commissioning"
    } == {
        "classical_tracking": "classical_tracking_hybrid",
        "fixed_dig": "fixed_dig_hybrid",
        "fixed_target_selection": "fixed_target_hybrid",
    }
    suite.profiles["fixed_dig"].preflight()
    suite.profiles["fixed_target_selection"].preflight()
    suite.profiles["classical_tracking"].preflight()


def test_classical_profile_changes_only_tracking_and_binds_cartesian_runtime():
    suite = Icra2027ExperimentSuite.load_directory(ACTIVE_SUITE)
    profile = suite.profiles["classical_tracking"]

    assert profile.reference_profile_id == "fixed_target_selection"
    assert profile.method_factors == {
        "software_architecture": "regime_factorized",
        "target_selection": "fixed_catalog",
        "task_policy": "act_dig_lift_fixed_dump",
        "trajectory_tracking": "cartesian_p",
    }
    assert profile.bindings == {
        "collection_ui_config": ACTIVE_CLASSICAL_UI,
        "hybrid_evidence_config": ACTIVE_CLASSICAL_EVIDENCE,
        "resident_fixed_cycle_config": ACTIVE_CLASSICAL_RESIDENT,
    }
    profile.preflight()


def test_fixed_dig_profile_changes_only_task_policy_and_preflights():
    suite = Icra2027ExperimentSuite.load_directory(ACTIVE_SUITE)
    profile = suite.profiles["fixed_dig"]

    assert profile.readiness == "commissioning"
    assert profile.reference_profile_id == "fixed_target_selection"
    assert profile.isolated_factor == "task_policy"
    assert profile.method_factors == {
        "software_architecture": "regime_factorized",
        "target_selection": "fixed_catalog",
        "task_policy": "fixed_dig_fixed_dump",
        "trajectory_tracking": "tc_btf",
    }
    assert profile.bindings == {
        "collection_ui_config": ACTIVE_FIXED_DIG_UI,
        "hybrid_evidence_config": ACTIVE_FIXED_DIG_EVIDENCE,
        "resident_fixed_cycle_config": ACTIVE_FIXED_DIG_RESIDENT,
    }
    profile.preflight()


def test_tadps_component_profile_uses_selector_and_planner_separated_metrics():
    profile = Icra2027ExperimentSuite.load_directory(ACTIVE_SUITE).profiles[
        "tadps"
    ]

    assert profile.study_kind == "component_benchmark"
    assert profile.readiness == "planned"
    assert profile.required_metrics == (
        "terrain_valid_output_rate",
        "target_displacement_m",
        "target_switch_rate",
        "dropout_rate",
        "runtime_ms",
        "downstream_planner_accept_rate",
    )


def _write_classical_profile_with_bindings(
    tmp_path: Path,
    *,
    runtime_payload: dict[str, object],
    evidence_payload: dict[str, object],
) -> Icra2027ExperimentProfile:
    profile_path = tmp_path / "classical_tracking.json"
    runtime_path = tmp_path / "resident.json"
    evidence_path = tmp_path / "evidence.json"
    ui_path = tmp_path / "collection_ui.json"
    profile_payload = json.loads(ACTIVE_CLASSICAL_PROFILE.read_text(encoding="utf-8"))
    profile_payload["bindings"] = {
        "collection_ui_config": str(ui_path),
        "resident_fixed_cycle_config": str(runtime_path),
        "hybrid_evidence_config": str(evidence_path),
    }
    evidence_payload["repository_paths"]["excavator_orin_runtime"] = str(
        ORIN_REPOSITORY
    )
    evidence_payload["mission_id"] = "classical_tracking_hybrid"
    evidence_payload["mission_sha256"] = MISSION_SHA256[
        "classical_tracking_hybrid"
    ]
    evidence_payload["config_paths"]["edge_runtime"] = str(
        ACTIVE_CARTESIAN_EDGE_RUNTIME
    )
    evidence_payload["config_paths"]["experiment_profile"] = str(profile_path)
    profile_path.write_text(json.dumps(profile_payload), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime_payload), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence_payload), encoding="utf-8")
    ui_payload = json.loads(ACTIVE_CLASSICAL_UI.read_text(encoding="utf-8"))
    ui_payload["guided_config"] = str(PROJECT_ROOT / "config/guided_episode.pc.json")
    ui_payload["resident_fixed_cycle_config"] = str(runtime_path)
    ui_payload["hybrid_evidence_config"] = str(evidence_path)
    ui_path.write_text(json.dumps(ui_payload), encoding="utf-8")
    return Icra2027ExperimentProfile.load(profile_path)


def test_classical_profile_rejects_an_onnx_runtime_binding(tmp_path: Path):
    runtime = json.loads(ACTIVE_CLASSICAL_RESIDENT.read_text(encoding="utf-8"))
    runtime.update(
        {
            "edge_runtime_config": "deploy/edge_runtime.resident.remote.json",
            "trajectory_controller_backend": "onnx_rl",
            "trajectory_controller_commissioning_authorization": "",
        }
    )
    evidence = json.loads(ACTIVE_CLASSICAL_EVIDENCE.read_text(encoding="utf-8"))
    profile = _write_classical_profile_with_bindings(
        tmp_path,
        runtime_payload=runtime,
        evidence_payload=evidence,
    )

    with pytest.raises(ValueError, match="trajectory_tracking"):
        profile.preflight()


def test_classical_profile_rejects_an_onnx_evidence_identity(tmp_path: Path):
    runtime = json.loads(ACTIVE_CLASSICAL_RESIDENT.read_text(encoding="utf-8"))
    evidence = json.loads(ACTIVE_EVIDENCE.read_text(encoding="utf-8"))
    profile = _write_classical_profile_with_bindings(
        tmp_path,
        runtime_payload=runtime,
        evidence_payload=evidence,
    )

    with pytest.raises(ValueError, match="trajectory_tracking"):
        profile.preflight()


def test_classical_profile_rejects_evidence_bound_to_another_edge_runtime(
    tmp_path: Path,
):
    runtime = json.loads(ACTIVE_CLASSICAL_RESIDENT.read_text(encoding="utf-8"))
    evidence = json.loads(ACTIVE_CLASSICAL_EVIDENCE.read_text(encoding="utf-8"))
    evidence["config_paths"]["edge_runtime"] = str(ACTIVE_ONNX_EDGE_RUNTIME)
    profile = _write_classical_profile_with_bindings(
        tmp_path,
        runtime_payload=runtime,
        evidence_payload=evidence,
    )
    evidence_path = profile.bindings["hybrid_evidence_config"]
    assert evidence_path is not None
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["config_paths"]["edge_runtime"] = str(ACTIVE_ONNX_EDGE_RUNTIME)
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="edge_runtime"):
        profile.preflight()


def test_classical_profile_rejects_edge_runtime_backend_mismatch(tmp_path: Path):
    runtime = json.loads(ACTIVE_CLASSICAL_RESIDENT.read_text(encoding="utf-8"))
    edge_path = tmp_path / "orin/deploy/cartesian.json"
    edge_path.parent.mkdir(parents=True)
    edge_path.write_text(
        json.dumps({"trajectory_controller_backend": "onnx_rl"}),
        encoding="utf-8",
    )
    runtime["edge_runtime_config"] = "deploy/cartesian.json"
    evidence = json.loads(ACTIVE_CLASSICAL_EVIDENCE.read_text(encoding="utf-8"))
    profile = _write_classical_profile_with_bindings(
        tmp_path,
        runtime_payload=runtime,
        evidence_payload=evidence,
    )
    evidence_path = profile.bindings["hybrid_evidence_config"]
    assert evidence_path is not None
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["repository_paths"]["excavator_orin_runtime"] = str(
        tmp_path / "orin"
    )
    payload["config_paths"]["edge_runtime"] = str(edge_path)
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="edge_runtime.*backend"):
        profile.preflight()


def _write_bound_profile(
    tmp_path: Path,
    *,
    evaluation_scope: str = "held_out_experiment",
    evidence_mission_id: str = "fixed_target_hybrid",
    experiment_profile_path: Path | None = None,
    resident_config: Path = ACTIVE_RESIDENT,
    expected_mission_id: str = "fixed_target_hybrid",
    task_policy: str = "act_dig_lift_fixed_dump",
) -> Icra2027ExperimentProfile:
    profile_path = tmp_path / "fixed_target_selection.json"
    evidence_path = tmp_path / "evidence.json"
    ui_path = tmp_path / "collection_ui.json"
    evidence = json.loads(ACTIVE_EVIDENCE.read_text(encoding="utf-8"))
    active_evidence = HybridExperimentRunConfig.load(ACTIVE_EVIDENCE)
    evidence["evidence_root"] = str(tmp_path / "evidence")
    evidence["machine_profile_path"] = str(active_evidence.machine_profile_path)
    evidence["repository_paths"] = {
        label: str(path)
        for label, path in active_evidence.repository_paths.items()
    }
    evidence["config_paths"] = {
        label: str(path) for label, path in active_evidence.config_paths.items()
    }
    evidence["artifacts"] = [
        {
            "artifact_id": artifact.artifact_id,
            "source_path": str(artifact.source_path),
            "role": artifact.role,
            "metadata": dict(artifact.metadata),
        }
        for artifact in active_evidence.artifacts
    ]
    evidence["evaluation_scope"] = evaluation_scope
    evidence["mission_id"] = evidence_mission_id
    evidence["mission_sha256"] = MISSION_SHA256[evidence_mission_id]
    evidence["config_paths"]["experiment_profile"] = str(
        experiment_profile_path or profile_path
    )
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    payload = _profile_payload()
    payload["expected_mission_id"] = expected_mission_id
    payload["expected_mission_sha256"] = MISSION_SHA256[expected_mission_id]
    payload["method_factors"] = {
        **payload["method_factors"],
        "task_policy": task_policy,
    }
    payload["bindings"] = {
        "collection_ui_config": str(ui_path),
        "resident_fixed_cycle_config": str(resident_config),
        "hybrid_evidence_config": str(evidence_path),
    }
    ui_payload = json.loads(ACTIVE_UI.read_text(encoding="utf-8"))
    ui_payload["guided_config"] = str(PROJECT_ROOT / "config/guided_episode.pc.json")
    ui_payload["resident_fixed_cycle_config"] = str(resident_config)
    ui_payload["hybrid_evidence_config"] = str(evidence_path)
    ui_path.write_text(json.dumps(ui_payload), encoding="utf-8")
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    return Icra2027ExperimentProfile.load(profile_path)


def test_ready_profile_rejects_training_internal_evidence(tmp_path: Path):
    profile = _write_bound_profile(
        tmp_path,
        evaluation_scope="training_internal",
    )

    with pytest.raises(ValueError, match="held_out_experiment"):
        profile.preflight()


def test_ready_profile_rejects_evidence_bound_to_another_profile(tmp_path: Path):
    profile = _write_bound_profile(
        tmp_path,
        experiment_profile_path=ACTIVE_SUITE / "proposed_hybrid.json",
    )

    with pytest.raises(ValueError, match="experiment_profile must bind"):
        profile.preflight()


def test_ready_profile_rejects_evidence_for_another_mission(tmp_path: Path):
    runtime_path = tmp_path / "fixed-target-runtime.json"
    runtime = json.loads(ACTIVE_CLASSICAL_RESIDENT.read_text(encoding="utf-8"))
    runtime.update(
        {
            "expected_mission_id": "fixed_target_hybrid",
            "expected_mission_sha256": MISSION_SHA256["fixed_target_hybrid"],
            "edge_runtime_config": "deploy/edge_runtime.resident.remote.json",
            "trajectory_controller_backend": "onnx_rl",
            "trajectory_controller_commissioning_authorization": "",
        }
    )
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    profile = _write_bound_profile(
        tmp_path,
        resident_config=runtime_path,
        evidence_mission_id="classical_tracking_hybrid",
    )

    with pytest.raises(ValueError, match="evidence mission_id does not match"):
        profile.preflight()


def test_ready_profile_rejects_runtime_with_wrong_expected_mission_id(
    tmp_path: Path,
):
    runtime_path = tmp_path / "wrong-mission.json"
    runtime = json.loads(ACTIVE_CLASSICAL_RESIDENT.read_text(encoding="utf-8"))
    runtime["expected_mission_id"] = "engineering_act_transport_reference"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    profile = _write_bound_profile(
        tmp_path,
        resident_config=runtime_path,
    )

    with pytest.raises(ValueError, match="expected_mission_id does not match"):
        profile.preflight()


def test_fixed_dig_profile_rejects_runtime_that_starts_act_worker(tmp_path: Path):
    runtime_path = tmp_path / "fixed-dig-with-act.json"
    runtime = json.loads(ACTIVE_FIXED_DIG_RESIDENT.read_text(encoding="utf-8"))
    runtime.update(
        {
            "expected_act_worker_required": True,
            "expected_act_behavior_id": "act_dig_lift",
            "expected_act_model_sha256": "a" * 64,
            "act_runtime_config": "/opt/act/runtime.json",
            "act_checkpoint_host_path": "/opt/act/checkpoint",
            "act_deployment_host_path": "/opt/act/deployment",
        }
    )
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    profile = _write_bound_profile(
        tmp_path,
        resident_config=runtime_path,
        evidence_mission_id="fixed_dig_hybrid",
        expected_mission_id="fixed_dig_hybrid",
        task_policy="fixed_dig_fixed_dump",
    )

    with pytest.raises(ValueError, match="ACT worker requirement"):
        profile.preflight()


def test_profile_preflight_rejects_tmp_profile_that_borrows_repo_assets(
    tmp_path: Path,
):
    profile = _write_bound_profile(tmp_path)

    with pytest.raises(ValueError, match="experiment profile.*repository-scoped"):
        profile.preflight()


def test_profile_loader_rejects_a_symlinked_profile(tmp_path: Path):
    redirect = tmp_path / "fixed_target_selection.json"
    redirect.symlink_to(ACTIVE_SUITE / "fixed_target_selection.json")

    with pytest.raises(ValueError, match="symlink"):
        Icra2027ExperimentProfile.load(redirect)


def test_inspector_reports_ready_and_planned_profiles(capsys):
    exit_code = main([str(ACTIVE_SUITE)])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["passed"] is True
    assert report["ready_profiles"] == []
    assert report["commissioning_profiles"] == [
        "classical_tracking",
        "fixed_dig",
        "fixed_target_selection",
    ]
    assert report["profiles"]["fixed_target_selection"]["method_factors"] == {
        "software_architecture": "regime_factorized",
        "target_selection": "fixed_catalog",
        "task_policy": "act_dig_lift_fixed_dump",
        "trajectory_tracking": "tc_btf",
    }
    assert report["profiles"]["fixed_target_selection"]["expected_mission_id"] == (
        "fixed_target_hybrid"
    )
    assert report["profiles"]["classical_tracking"]["isolated_factor"] == (
        "trajectory_tracking"
    )
    assert set(report["planned_profiles"]) == {
        "proposed_hybrid",
        "tadps",
    }


def test_inspector_fails_cleanly_when_requested_profile_is_not_ready(capsys):
    exit_code = main(
        [str(ACTIVE_SUITE), "--require-ready", "proposed_hybrid"]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["passed"] is False
    assert "proposed_hybrid" in report["profiles"]
    assert report["failure_reasons"] == [
        "profile proposed_hybrid is planned, not ready"
    ]

    exit_code = main(
        [str(ACTIVE_SUITE), "--require-ready", "fixed_target_selection"]
    )
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["failure_reasons"] == [
        "profile fixed_target_selection is commissioning, not ready"
    ]


def test_inspector_reports_invalid_suite_without_traceback(tmp_path: Path, capsys):
    exit_code = main([str(tmp_path / "missing")])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 2
    assert report["passed"] is False
    assert report["profiles"] == {}
    assert report["commissioning_profiles"] == []
    assert "does not exist" in report["failure_reasons"][0]
    assert captured.err == ""

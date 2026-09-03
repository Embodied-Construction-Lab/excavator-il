import json
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.inspect_icra2027_experiment_readiness import build_report


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/inspect_icra2027_experiment_readiness.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )


def test_active_readiness_report_keeps_commissioning_distinct_from_formal_ready():
    result = _run()

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == (
        "excavator_icra2027_experiment_readiness_report.v1"
    )
    assert report["passed"] is True
    assert report["summary"] == {
        "commissioning": 3,
        "planned": 3,
        "ready": 0,
        "total": 6,
    }
    assert tuple(report["experiments"]) == (
        "classical_tracking",
        "fixed_dig",
        "fixed_target_selection",
        "proposed_hybrid",
        "rl_sim_real",
        "tadps",
    )
    for profile_id in (
        "classical_tracking",
        "fixed_dig",
        "fixed_target_selection",
    ):
        experiment = report["experiments"][profile_id]
        assert experiment["declared_readiness"] == "commissioning"
        assert experiment["static_preflight_passed"] is True
        assert experiment["formal_ready"] is False
        assert {
            blocked["input_id"] for blocked in experiment["blocked_inputs"]
        } == {
            "engine_off_gate_evidence",
            "single_scoop_gate_evidence",
            "multi_scoop_gate_evidence",
            "held_out_experiment_evidence",
        }

    assert report["experiments"]["tadps"]["declared_readiness"] == "planned"
    tadps_assets = {
        asset["binding_id"]: asset
        for asset in report["experiments"]["tadps"]["assets"]
    }
    assert {
        "candidate_replay_exporter",
        "candidate_frame_schema_example",
        "live_candidate_frame_capture_cli",
        "live_candidate_frame_capture_module",
        "offline_evaluator",
        "selector_bridge_descriptor",
        "selector_bridge_patch",
    } <= set(tadps_assets)
    assert tadps_assets["live_candidate_frame_capture_cli"]["strict_load"] == (
        "imported"
    )
    assert {
        blocked["input_id"]
        for blocked in report["experiments"]["tadps"]["blocked_inputs"]
    } == {
        "frozen_real_candidate_replay",
        "held_out_tadps_split",
    }
    assert report["experiments"]["rl_sim_real"]["declared_readiness"] == (
        "planned"
    )
    assert report["experiments"]["rl_sim_real"]["study_kind"] == (
        "tracking_parity_pipeline"
    )
    assert report["experiments"]["rl_sim_real"]["formal_ready"] is False
    assert report["experiments"]["rl_sim_real"][
        "tracking_trace_schema_version"
    ] == "excavator_rl_control_trace.v3"
    assert report["experiments"]["rl_sim_real"][
        "trajectory_suite_contract"
    ] == {
        "exact_fields": ["suite_id", "sample_period_s", "sample_ids"],
        "suite_id": "nonempty_text",
        "sample_id_semantics": "elapsed_policy_decision_index",
        "sample_ids": "nonempty_contiguous_prefix_starting_at_zero",
        "sample_period_s": 0.1,
    }
    assert report["experiments"]["rl_sim_real"][
        "pair_alignment_contract"
    ] == {
        "alignment": "nonempty_common_prefix",
        "aggregate_denominator": "prefrozen_attempt_manifest_entries",
        "missing_tail_policy": "report_without_imputation",
        "zero_sample_attempt_policy": (
            "manifest_missing_pair_report_without_tracking_metrics"
        ),
        "required_outputs": [
            "simulation_only_tail_sample_ids",
            "real_machine_only_tail_sample_ids",
            "sample_coverage",
            "duration_s",
            "tracking.terminal_result",
            "tracking.bucket_tip_euclidean_error_m",
            "tracking.reference_waypoint_euclidean_error_m",
            "tracking.waypoint_index_agreement",
            "tracking.relative_sample_timing_error_s",
            "tracking.waypoint_distance_m",
        ],
    }
    assert report["experiments"]["rl_sim_real"][
        "formal_evidence_contract"
    ] == {
        "attempt_manifest_schema_version": (
            "excavator_rl_sim_real_attempt_manifest.v1"
        ),
        "attempt_order": "prefrozen_and_preserved",
        "aggregate_schema_version": "excavator_rl_sim_real_aggregate.v1",
        "required_aggregate_evidence_complete": True,
        "missing_pair_report": (
            "retained_in_attempt_denominator_without_tracking_metrics"
        ),
        "pair_trace_binding": {
            "trajectory_trace_schema_version": "excavator_rl_control_trace.v3",
            "trace_sha256": ["simulation", "real_machine"],
        },
    }
    assert {
        blocked["input_id"]
        for blocked in report["experiments"]["rl_sim_real"]["blocked_inputs"]
    } == {
        "simulation_tracking_trace",
        "real_tracking_trace",
        "frozen_attempt_manifest",
        "held_out_paired_tracking_runs",
    }
    rl_asset_ids = {
        asset["binding_id"]
        for asset in report["experiments"]["rl_sim_real"]["assets"]
    }
    assert {
        "simulation_control_audit_producer",
        "simulation_control_audit_writer",
        "simulation_control_trace_converter",
        "convert_simulation_trace_cli",
        "trajectory_suite_generator",
        "real_control_audit_producer",
        "real_control_audit_writer",
        "real_control_trace_producer",
        "record_real_parent_cli",
        "held_out_pair_aggregator",
        "attempt_manifest_aggregator",
        "aggregate_pairs_cli",
    } <= rl_asset_ids


def test_missing_planned_input_is_reported_without_failing_static_preflight():
    suite = REPOSITORY / "config/experiments/icra2027"
    first, first_code = build_report(suite_path=suite)
    second, second_code = build_report(suite_path=suite)

    assert first_code == second_code == 0
    assert first == second
    assert first["experiments"]["tadps"]["static_preflight_passed"] is True
    assert first["experiments"]["rl_sim_real"]["static_preflight_passed"] is (
        True
    )
    assert first["failure_reasons"] == []


def test_frozen_tadps_replay_can_be_strictly_loaded_without_changing_readiness(
    tmp_path: Path,
):
    replay = tmp_path / "candidate_replay.json"
    replay.write_text(
        json.dumps(
            {
                "schema_version": "tadps_candidate_replay.v1",
                "frame_id": "world",
                "sequences": [
                    {
                        "sequence_id": "held-out-001",
                        "frames": [
                            {
                                "frame_index": 0,
                                "stamp_s": 1.0,
                                "map_sha256": "a" * 64,
                                "candidates": [],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report, exit_code = build_report(
        suite_path=REPOSITORY / "config/experiments/icra2027",
        tadps_replay=replay,
    )

    assert exit_code == 0
    experiment = report["experiments"]["tadps"]
    assert experiment["declared_readiness"] == "planned"
    assert experiment["formal_ready"] is False
    assert {
        blocked["input_id"] for blocked in experiment["blocked_inputs"]
    } == {
        "frozen_real_candidate_replay",
        "held_out_tadps_split",
    }
    assert experiment["assets"][-1] == {
        "binding_id": "frozen_candidate_replay",
        "exists": True,
        "path": str(replay.resolve()),
        "strict_load": "passed",
    }


def test_invalid_provided_planned_input_is_a_configuration_error(tmp_path: Path):
    replay = tmp_path / "candidate_replay.json"
    replay.write_text("{}", encoding="utf-8")

    report, exit_code = build_report(
        suite_path=REPOSITORY / "config/experiments/icra2027",
        tadps_replay=replay,
    )

    assert exit_code == 2
    assert report["passed"] is False
    assert report["experiments"]["tadps"]["static_preflight_passed"] is False
    assert report["failure_reasons"][0].startswith("experiment tadps:")


def test_corrupt_tadps_bridge_descriptor_fails_static_preflight(tmp_path: Path):
    repository, suite = _workspace_fixture(tmp_path)
    airy_root = repository.parent / "AiryLidar"
    airy_root.unlink()
    shutil.copytree(REPOSITORY.parent / "AiryLidar/mission", airy_root / "mission")
    descriptor_path = (
        airy_root
        / "mission/bridges/excavator_dig_point_tadps_candidate_trace.v1.json"
    )
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["patch_sha256"] = "0" * 64
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    report, exit_code = build_report(suite_path=suite)

    assert exit_code == 2
    tadps = report["experiments"]["tadps"]
    assert tadps["declared_readiness"] == "planned"
    assert tadps["formal_ready"] is False
    assert tadps["static_preflight_passed"] is False
    assert any(
        reason.startswith("experiment tadps:")
        for reason in report["failure_reasons"]
    )


def test_unloadable_tadps_capture_cli_fails_static_preflight(tmp_path: Path):
    repository, suite = _workspace_fixture(tmp_path)
    airy_root = repository.parent / "AiryLidar"
    airy_root.unlink()
    shutil.copytree(REPOSITORY.parent / "AiryLidar/mission", airy_root / "mission")
    cli_path = airy_root / "mission/scripts/capture_tadps_candidate_frames.py"
    source = cli_path.read_text(encoding="utf-8")
    cli_path.write_text(
        "import readiness_test_missing_dependency\n" + source,
        encoding="utf-8",
    )

    report, exit_code = build_report(suite_path=suite)

    assert exit_code == 2
    tadps = report["experiments"]["tadps"]
    assert tadps["declared_readiness"] == "planned"
    assert tadps["formal_ready"] is False
    assert tadps["static_preflight_passed"] is False
    assert any(
        reason.startswith("experiment tadps:")
        for reason in report["failure_reasons"]
    )


def _workspace_fixture(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "RL_prj"
    repository = workspace / "excavator-il"
    shutil.copytree(REPOSITORY / "config", repository / "config")
    for name in ("scripts", "src", "models", "outputs"):
        (repository / name).symlink_to(REPOSITORY / name, target_is_directory=True)
    for name in (
        "AiryLidar",
        "RLExcavator",
        "excavator-orin-runtime",
        "shared",
        "urdf",
    ):
        (workspace / name).symlink_to(
            REPOSITORY.parent / name,
            target_is_directory=True,
        )
    return repository, repository / "config/experiments/icra2027"


def test_invalid_commissioning_binding_returns_exit_two(tmp_path: Path):
    repository, suite = _workspace_fixture(tmp_path)
    runtime_path = (
        repository
        / "config/resident_fixed_cycle.fixed_dig.commissioning.pc.json"
    )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["expected_mission_id"] = "unknown_profile"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

    report, exit_code = build_report(suite_path=suite)

    assert exit_code == 2
    fixed_dig = report["experiments"]["fixed_dig"]
    assert fixed_dig["declared_readiness"] == "commissioning"
    assert fixed_dig["static_preflight_passed"] is False
    assert fixed_dig["formal_ready"] is False
    assert any(
        reason.startswith("experiment fixed_dig:")
        for reason in report["failure_reasons"]
    )

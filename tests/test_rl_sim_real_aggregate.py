from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from excavator_il.rl_sim_real_aggregate import (
    aggregate_rl_sim_real_pair_reports,
    write_rl_sim_real_aggregate,
)
from excavator_il.rl_sim_real_attempt_manifest import (
    aggregate_rl_sim_real_attempt_manifest,
)


POLICY_SHA = "a" * 64
PROFILE_SHA = "b" * 64
SUITE_SHA = "c" * 64


def _write_report(
    path: Path,
    *,
    pair_id: str,
    simulation_result: str = "COMPLETED",
    real_result: str = "COMPLETED",
    aligned_count: int = 8,
    simulation_sample_count: int | None = None,
    real_machine_sample_count: int | None = None,
    suite_count: int = 10,
    bucket_mae: float = 0.1,
    sim_waypoint_final: float = 0.02,
    real_waypoint_final: float = 0.04,
    sim_waypoint_p95: float = 0.2,
    real_waypoint_p95: float = 0.3,
    sim_duration: float = 4.0,
    real_duration: float = 5.0,
    evaluation_scope: str = "held_out_experiment",
    binding_overrides: dict[str, object] | None = None,
) -> Path:
    terminal_labels = ("completed", "timeout", "rejected", "interrupted")
    binding: dict[str, object] = {
        "trajectory_controller_policy_id": "onnx_rl:held-out-controller",
        "trajectory_controller_onnx_sha256": POLICY_SHA,
        "machine_profile_sha256": PROFILE_SHA,
        "action_order": ["boom", "stick", "bucket", "swing"],
        "trajectory_suite_sha256": SUITE_SHA,
        "trajectory_suite_sample_period_s": 0.1,
        "trajectory_trace_schema_version": "excavator_rl_control_trace.v3",
        "trace_semantics": "commanded_normalized_action",
    }
    if binding_overrides:
        binding.update(binding_overrides)
    simulation_count = simulation_sample_count or aligned_count
    real_machine_count = real_machine_sample_count or aligned_count
    assert aligned_count == min(simulation_count, real_machine_count)
    report = {
        "schema_version": "excavator_rl_sim_real_pair.v1",
        "pair_id": pair_id,
        "evaluation_scope": evaluation_scope,
        "trace_semantics": "commanded_normalized_action",
        "simulation_run_id": f"sim-{pair_id}",
        "real_machine_run_id": f"real-{pair_id}",
        "simulation_sample_count": simulation_count,
        "real_machine_sample_count": real_machine_count,
        "aligned_sample_count": aligned_count,
        "simulation_only_tail_count": simulation_count - aligned_count,
        "simulation_only_tail_sample_ids": list(
            range(aligned_count, simulation_count)
        ),
        "real_machine_only_tail_count": real_machine_count - aligned_count,
        "real_machine_only_tail_sample_ids": list(
            range(aligned_count, real_machine_count)
        ),
        "sample_coverage": {
            "simulation": {
                "consumed_count": simulation_count,
                "suite_count": suite_count,
                "rate": simulation_count / suite_count,
            },
            "real_machine": {
                "consumed_count": real_machine_count,
                "suite_count": suite_count,
                "rate": real_machine_count / suite_count,
            },
        },
        "duration_s": {
            "simulation": sim_duration,
            "real_machine": real_duration,
        },
        "trace_sha256": {
            "simulation": (pair_id.encode("utf-8").hex() + "1" * 64)[:64],
            "real_machine": (pair_id.encode("utf-8").hex() + "2" * 64)[:64],
        },
        "nonzero_agreement_rate": 0.75,
        "axes": {
            axis: {
                "mae": 0.1,
                "rmse": 0.2,
                "max_abs": 0.3,
                "sign_agreement_rate": 0.8,
            }
            for axis in ("boom", "stick", "bucket", "swing")
        },
        "tracking": {
            "bucket_tip_euclidean_error_m": {
                "mae": bucket_mae,
                "rmse": bucket_mae + 0.02,
                "max": bucket_mae + 0.05,
            },
            "reference_waypoint_euclidean_error_m": {
                "mae": 0.02,
                "rmse": 0.03,
                "max": 0.05,
            },
            "waypoint_index_agreement": {
                "count": max(0, aligned_count - 1),
                "rate": max(0, aligned_count - 1) / aligned_count,
            },
            "relative_sample_timing_error_s": {
                "mae": 0.01,
                "rmse": 0.015,
                "max_abs": 0.03,
            },
            "waypoint_distance_m": {
                "simulation": {
                    "mean": sim_waypoint_p95 / 2,
                    "p95": sim_waypoint_p95,
                    "final": sim_waypoint_final,
                },
                "real_machine": {
                    "mean": real_waypoint_p95 / 2,
                    "p95": real_waypoint_p95,
                    "final": real_waypoint_final,
                },
            },
            "terminal_result": {
                "agreement": simulation_result == real_result,
                "simulation": simulation_result,
                "real_machine": real_result,
                **{
                    f"{label}_count": {
                        "simulation": int(simulation_result == label.upper()),
                        "real_machine": int(real_result == label.upper()),
                    }
                    for label in terminal_labels
                },
            },
        },
        "binding": binding,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_attempt_manifest(
    path: Path,
    *,
    attempts: list[dict[str, str]],
    binding_overrides: dict[str, object] | None = None,
) -> Path:
    binding: dict[str, object] = {
        "trajectory_controller_policy_id": "onnx_rl:held-out-controller",
        "trajectory_controller_onnx_sha256": POLICY_SHA,
        "machine_profile_sha256": PROFILE_SHA,
        "action_order": ["boom", "stick", "bucket", "swing"],
        "trajectory_suite_sha256": SUITE_SHA,
        "trajectory_suite_sample_period_s": 0.1,
        "trajectory_trace_schema_version": "excavator_rl_control_trace.v3",
        "trace_semantics": "commanded_normalized_action",
    }
    if binding_overrides:
        binding.update(binding_overrides)
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_rl_sim_real_attempt_manifest.v1",
                "evaluation_scope": "held_out_experiment",
                "study_id": "rl-sim-real-held-out-001",
                "binding": binding,
                "attempts": attempts,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_aggregate_reports_attempted_pair_denominator_and_pair_level_metrics(
    tmp_path: Path,
):
    first = _write_report(
        tmp_path / "pair-b.json",
        pair_id="pair-b",
        simulation_result="COMPLETED",
        real_result="TIMEOUT",
        aligned_count=5,
        simulation_sample_count=7,
        real_machine_sample_count=5,
        bucket_mae=0.3,
        sim_waypoint_final=0.03,
        real_waypoint_final=0.09,
        sim_duration=6.0,
        real_duration=8.0,
    )
    second = _write_report(
        tmp_path / "pair-a.json",
        pair_id="pair-a",
        simulation_result="INTERRUPTED",
        real_result="INTERRUPTED",
        aligned_count=8,
        simulation_sample_count=8,
        real_machine_sample_count=10,
        bucket_mae=0.1,
        sim_waypoint_final=0.01,
        real_waypoint_final=0.05,
        sim_duration=4.0,
        real_duration=6.0,
    )

    aggregate = aggregate_rl_sim_real_pair_reports([first, second])

    assert aggregate["schema_version"] == "excavator_rl_sim_real_aggregate.v1"
    assert aggregate["evaluation_scope"] == "held_out_experiment"
    assert aggregate["attempted_pair_count"] == 2
    assert [item["pair_id"] for item in aggregate["attempted_pairs"]] == [
        "pair-b",
        "pair-a",
    ]
    assert aggregate["attempted_pairs"][0]["simulation_only_tail_sample_ids"] == [
        5,
        6,
    ]
    assert aggregate["attempted_pairs"][1]["real_machine_only_tail_sample_ids"] == [
        8,
        9,
    ]
    assert aggregate["terminal"]["agreement"] == {"count": 1, "rate": 0.5}
    assert aggregate["terminal"]["simulation"]["completed"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert aggregate["terminal"]["simulation"]["interrupted"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert aggregate["terminal"]["real_machine"]["timeout"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert aggregate["terminal"]["real_machine"]["interrupted"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert aggregate["tracking"]["bucket_tip_euclidean_error_m"]["pair_mae"] == {
        "count": 2,
        "mean": 0.2,
        "median": 0.2,
        "p95": 0.29,
        "min": 0.1,
        "max": 0.3,
    }
    assert aggregate["tracking"]["waypoint_distance_m"]["real_machine"][
        "pair_final"
    ]["mean"] == pytest.approx(0.07)
    assert aggregate["tracking"]["reference_waypoint_euclidean_error_m"][
        "pair_mae"
    ]["mean"] == 0.02
    assert aggregate["tracking"]["relative_sample_timing_error_s"]["pair_mae"][
        "mean"
    ] == 0.01
    assert aggregate["tracking"]["waypoint_index_agreement"]["pair_rate"][
        "mean"
    ] == pytest.approx(0.8375)
    assert aggregate["duration_s"]["simulation"]["mean"] == 5.0
    assert aggregate["sample_counts"]["simulation"]["mean"] == 7.5
    assert aggregate["sample_counts"]["real_machine"]["mean"] == 7.5
    assert aggregate["sample_counts"]["aligned"]["mean"] == 6.5
    assert aggregate["tails"]["simulation"]["total_sample_count"] == 2
    assert aggregate["tails"]["simulation"]["affected_pairs"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert aggregate["tails"]["real_machine"]["total_sample_count"] == 2
    assert aggregate["sample_coverage"]["simulation"]["pair_rate"]["mean"] == 0.75
    assert aggregate["sample_coverage"]["real_machine"]["pair_rate"]["mean"] == 0.75
    assert aggregate["sample_coverage"]["simulation"][
        "total_consumed_sample_count"
    ] == 15
    assert aggregate["sample_coverage"]["real_machine"][
        "total_consumed_sample_count"
    ] == 15
    assert aggregate["sample_coverage"]["total_aligned_sample_count"] == 13


def test_formal_manifest_keeps_missing_pair_report_in_attempt_denominator(
    tmp_path: Path,
):
    _write_report(
        tmp_path / "reports" / "pair-present.json",
        pair_id="pair-present",
        simulation_result="COMPLETED",
        real_result="TIMEOUT",
    )
    manifest = _write_attempt_manifest(
        tmp_path / "attempts.json",
        attempts=[
            {
                "attempt_id": "attempt-002",
                "pair_id": "pair-missing",
                "simulation_run_id": "sim-pair-missing",
                "real_machine_run_id": "real-pair-missing",
                "pair_report_path": "reports/pair-missing.json",
            },
            {
                "attempt_id": "attempt-001",
                "pair_id": "pair-present",
                "simulation_run_id": "sim-pair-present",
                "real_machine_run_id": "real-pair-present",
                "pair_report_path": "reports/pair-present.json",
            },
        ],
    )

    aggregate = aggregate_rl_sim_real_attempt_manifest(manifest)

    assert aggregate["study_id"] == "rl-sim-real-held-out-001"
    assert aggregate["attempted_pair_count"] == 2
    assert aggregate["trace_bearing_pair_count"] == 1
    assert aggregate["evidence_complete"] is False
    assert [item["attempt_id"] for item in aggregate["attempted_pairs"]] == [
        "attempt-002",
        "attempt-001",
    ]
    assert aggregate["attempted_pairs"][0]["evidence_status"] == "missing_pair_report"
    assert aggregate["attempted_pairs"][0]["simulation_terminal"] == "UNKNOWN"
    assert aggregate["attempted_pairs"][1]["evidence_status"] == "trace_bearing"
    assert aggregate["terminal"]["simulation"]["completed"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert aggregate["terminal"]["simulation"]["unknown"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert aggregate["terminal"]["not_evaluable"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert aggregate["tracking"]["bucket_tip_euclidean_error_m"]["pair_mae"][
        "count"
    ] == 1
    assert aggregate["attempted_pairs"][1]["trace_sha256"]["simulation"]


def test_formal_manifest_with_only_pretrace_attempts_is_not_evaluable(
    tmp_path: Path,
):
    manifest = _write_attempt_manifest(
        tmp_path / "attempts.json",
        attempts=[
            {
                "attempt_id": "attempt-001",
                "pair_id": "pair-001",
                "simulation_run_id": "sim-pair-001",
                "real_machine_run_id": "real-pair-001",
                "pair_report_path": "reports/pair-001.json",
            }
        ],
    )

    aggregate = aggregate_rl_sim_real_attempt_manifest(manifest)

    assert aggregate["trace_bearing_pair_count"] == 0
    assert aggregate["missing_pair_report_count"] == 1
    assert aggregate["not_evaluable_attempt_count"] == 1
    assert aggregate["tracking"] is None
    assert aggregate["terminal"]["real_machine"]["unknown"] == {
        "count": 1,
        "rate": 1.0,
    }


@pytest.mark.parametrize(
    "pair_report_path",
    ["../escape.json", "/tmp/absolute.json", "reports/../../escape.json"],
)
def test_formal_manifest_rejects_pair_report_path_traversal(
    tmp_path: Path,
    pair_report_path: str,
):
    manifest = _write_attempt_manifest(
        tmp_path / "attempts.json",
        attempts=[
            {
                "attempt_id": "attempt-001",
                "pair_id": "pair-001",
                "simulation_run_id": "sim-pair-001",
                "real_machine_run_id": "real-pair-001",
                "pair_report_path": pair_report_path,
            }
        ],
    )

    with pytest.raises(ValueError, match="relative.*without traversal"):
        aggregate_rl_sim_real_attempt_manifest(manifest)


def test_formal_manifest_rejects_duplicate_attempt_ids_and_report_id_mismatch(
    tmp_path: Path,
):
    duplicate = {
        "attempt_id": "attempt-001",
        "pair_id": "pair-001",
        "simulation_run_id": "sim-pair-001",
        "real_machine_run_id": "real-pair-001",
        "pair_report_path": "reports/pair-001.json",
    }
    manifest = _write_attempt_manifest(
        tmp_path / "duplicate.json",
        attempts=[duplicate, {**duplicate, "pair_id": "pair-002"}],
    )
    with pytest.raises(ValueError, match="duplicate attempt_id"):
        aggregate_rl_sim_real_attempt_manifest(manifest)

    _write_report(tmp_path / "reports" / "pair.json", pair_id="pair-actual")
    manifest = _write_attempt_manifest(
        tmp_path / "mismatch.json",
        attempts=[
            {
                "attempt_id": "attempt-mismatch",
                "pair_id": "pair-expected",
                "simulation_run_id": "sim-pair-expected",
                "real_machine_run_id": "real-pair-expected",
                "pair_report_path": "reports/pair.json",
            }
        ],
    )
    with pytest.raises(ValueError, match="pair_id does not match"):
        aggregate_rl_sim_real_attempt_manifest(manifest)


def test_formal_manifest_rejects_duplicate_report_paths_and_binding_drift(
    tmp_path: Path,
):
    attempts = [
        {
            "attempt_id": f"attempt-{index}",
            "pair_id": f"pair-{index}",
            "simulation_run_id": f"sim-pair-{index}",
            "real_machine_run_id": f"real-pair-{index}",
            "pair_report_path": "reports/shared.json",
        }
        for index in (1, 2)
    ]
    manifest = _write_attempt_manifest(tmp_path / "duplicate.json", attempts=attempts)
    with pytest.raises(ValueError, match="duplicate resolved_report_path"):
        aggregate_rl_sim_real_attempt_manifest(manifest)

    _write_report(
        tmp_path / "reports" / "pair.json",
        pair_id="pair-001",
        binding_overrides={"trajectory_controller_onnx_sha256": "d" * 64},
    )
    manifest = _write_attempt_manifest(
        tmp_path / "drift.json",
        attempts=[
            {
                "attempt_id": "attempt-001",
                "pair_id": "pair-001",
                "simulation_run_id": "sim-pair-001",
                "real_machine_run_id": "real-pair-001",
                "pair_report_path": "reports/pair.json",
            }
        ],
    )
    with pytest.raises(ValueError, match="binding does not match"):
        aggregate_rl_sim_real_attempt_manifest(manifest)

    invalid_semantics = _write_attempt_manifest(
        tmp_path / "invalid-semantics.json",
        attempts=[attempts[0]],
        binding_overrides={"trace_semantics": "applied_actuator_feedback"},
    )
    with pytest.raises(ValueError, match="commanded_normalized_action"):
        aggregate_rl_sim_real_attempt_manifest(invalid_semantics)


def test_aggregate_rejects_missing_or_duplicate_attempts(tmp_path: Path):
    report = _write_report(tmp_path / "pair.json", pair_id="pair-001")

    with pytest.raises(ValueError, match="at least one attempted pair"):
        aggregate_rl_sim_real_pair_reports([])
    with pytest.raises(ValueError, match="must be a regular file"):
        aggregate_rl_sim_real_pair_reports([tmp_path / "missing.json"])
    with pytest.raises(ValueError, match="paths must be unique"):
        aggregate_rl_sim_real_pair_reports([report, report])


@pytest.mark.parametrize(
    ("first_kwargs", "second_kwargs", "message"),
    [
        (
            {},
            {"evaluation_scope": "training_internal"},
            "only held_out_experiment",
        ),
        (
            {},
            {
                "binding_overrides": {
                    "trajectory_controller_policy_id": "onnx_rl:different"
                }
            },
            "binding or hash drift",
        ),
        (
            {},
            {"binding_overrides": {"trajectory_controller_onnx_sha256": "d" * 64}},
            "binding or hash drift",
        ),
        (
            {},
            {"binding_overrides": {"machine_profile_sha256": "e" * 64}},
            "binding or hash drift",
        ),
        (
            {},
            {"binding_overrides": {"trajectory_suite_sha256": "f" * 64}},
            "binding or hash drift",
        ),
        (
            {},
            {
                "binding_overrides": {
                    "trajectory_trace_schema_version": "excavator_rl_control_trace.v2"
                }
            },
            "trajectory_trace_schema_version",
        ),
        (
            {},
            {
                "binding_overrides": {
                    "action_order": ["swing", "boom", "stick", "bucket"]
                }
            },
            "action_order",
        ),
    ],
)
def test_aggregate_rejects_scope_contract_or_hash_drift(
    tmp_path: Path,
    first_kwargs: dict[str, object],
    second_kwargs: dict[str, object],
    message: str,
):
    first = _write_report(
        tmp_path / "first.json", pair_id="pair-001", **first_kwargs
    )
    second = _write_report(
        tmp_path / "second.json", pair_id="pair-002", **second_kwargs
    )

    with pytest.raises(ValueError, match=message):
        aggregate_rl_sim_real_pair_reports([first, second])


def test_aggregate_rejects_duplicate_pair_and_parent_run_ids(tmp_path: Path):
    first = _write_report(tmp_path / "first.json", pair_id="pair-001")
    second = _write_report(tmp_path / "second.json", pair_id="pair-001")
    with pytest.raises(ValueError, match="duplicate pair_id"):
        aggregate_rl_sim_real_pair_reports([first, second])

    value = json.loads(second.read_text(encoding="utf-8"))
    value["pair_id"] = "pair-002"
    value["simulation_run_id"] = "sim-pair-001"
    second.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate simulation_run_id"):
        aggregate_rl_sim_real_pair_reports([first, second])


def test_aggregate_rejects_tampered_terminal_counts_and_sample_coverage(
    tmp_path: Path,
):
    terminal_tampered = _write_report(
        tmp_path / "terminal.json", pair_id="pair-terminal"
    )
    value = json.loads(terminal_tampered.read_text(encoding="utf-8"))
    value["tracking"]["terminal_result"]["timeout_count"]["simulation"] = 1
    terminal_tampered.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal count"):
        aggregate_rl_sim_real_pair_reports([terminal_tampered])

    coverage_tampered = _write_report(
        tmp_path / "coverage.json", pair_id="pair-coverage"
    )
    value = json.loads(coverage_tampered.read_text(encoding="utf-8"))
    value["sample_coverage"]["simulation"]["rate"] = 0.1
    coverage_tampered.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample_coverage.simulation.*rate"):
        aggregate_rl_sim_real_pair_reports([coverage_tampered])

    trace_tampered = _write_report(
        tmp_path / "trace.json", pair_id="pair-trace"
    )
    value = json.loads(trace_tampered.read_text(encoding="utf-8"))
    value["trace_sha256"]["simulation"] = "not-a-sha"
    trace_tampered.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trace_sha256.simulation"):
        aggregate_rl_sim_real_pair_reports([trace_tampered])


def test_aggregate_rejects_suite_count_drift_under_the_same_suite_hash(
    tmp_path: Path,
):
    first = _write_report(tmp_path / "first.json", pair_id="pair-001")
    second = _write_report(
        tmp_path / "second.json",
        pair_id="pair-002",
        aligned_count=8,
        suite_count=12,
    )

    with pytest.raises(ValueError, match="trajectory suite sample count drift"):
        aggregate_rl_sim_real_pair_reports([first, second])


def test_aggregate_rejects_tampered_common_prefix_tail_and_sample_period(
    tmp_path: Path,
):
    tail_tampered = _write_report(
        tmp_path / "tail.json",
        pair_id="pair-tail",
        aligned_count=6,
        simulation_sample_count=8,
        real_machine_sample_count=6,
    )
    value = json.loads(tail_tampered.read_text(encoding="utf-8"))
    value["simulation_only_tail_sample_ids"] = [7]
    tail_tampered.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit contiguous tail"):
        aggregate_rl_sim_real_pair_reports([tail_tampered])

    period_tampered = _write_report(
        tmp_path / "period.json", pair_id="pair-period"
    )
    value = json.loads(period_tampered.read_text(encoding="utf-8"))
    value["binding"]["trajectory_suite_sample_period_s"] = 0.2
    period_tampered.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample_period_s must be 0.1"):
        aggregate_rl_sim_real_pair_reports([period_tampered])


def test_write_aggregate_is_atomic_and_refuses_overwrite(tmp_path: Path):
    report = _write_report(tmp_path / "pair.json", pair_id="pair-001")
    aggregate = aggregate_rl_sim_real_pair_reports([report])
    output = tmp_path / "result" / "aggregate.json"

    assert write_rl_sim_real_aggregate(output, aggregate) == output
    assert json.loads(output.read_text(encoding="utf-8")) == aggregate
    with pytest.raises(ValueError, match="already exists"):
        write_rl_sim_real_aggregate(output, aggregate)


def test_cli_writes_explicit_attempted_pair_aggregate_and_refuses_overwrite(
    tmp_path: Path,
):
    first = _write_report(
        tmp_path / "first.json",
        pair_id="pair-001",
        simulation_result="REJECTED",
        real_result="REJECTED",
    )
    second = _write_report(
        tmp_path / "second.json",
        pair_id="pair-002",
        simulation_result="TIMEOUT",
        real_result="INTERRUPTED",
    )
    manifest = _write_attempt_manifest(
        tmp_path / "attempts.json",
        attempts=[
            {
                "attempt_id": "attempt-001",
                "pair_id": "pair-001",
                "simulation_run_id": "sim-pair-001",
                "real_machine_run_id": "real-pair-001",
                "pair_report_path": first.name,
            },
            {
                "attempt_id": "attempt-002",
                "pair_id": "pair-002",
                "simulation_run_id": "sim-pair-002",
                "real_machine_run_id": "real-pair-002",
                "pair_report_path": second.name,
            },
        ],
    )
    output = tmp_path / "aggregate.json"
    script = Path(__file__).parents[1] / "scripts" / "aggregate_rl_sim_real_pairs.py"
    command = [
        sys.executable,
        str(script),
        "--attempt-manifest",
        str(manifest),
        "--output",
        str(output),
    ]

    completed = subprocess.run(command, text=True, capture_output=True, check=False)

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary == {
        "attempted_pair_count": 2,
        "evidence_complete": True,
        "evaluation_scope": "held_out_experiment",
        "output": str(output.resolve()),
    }
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["terminal"]["simulation"]["rejected"]["count"] == 1
    assert aggregate["terminal"]["real_machine"]["interrupted"]["count"] == 1

    repeated = subprocess.run(command, text=True, capture_output=True, check=False)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr

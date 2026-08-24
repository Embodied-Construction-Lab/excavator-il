from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from excavator_il.evaluation_harness import (
    EvaluationError,
    evaluate_experiment_run,
    evaluate_experiment_runs,
    write_evaluation_outputs,
)
from excavator_il.experiment_run import ExperimentRun, TaskContext


class _Loader:
    def __init__(self, snapshot: SimpleNamespace) -> None:
        self._snapshot = snapshot

    def load(self, _run_path):
        return self._snapshot


class _MappingLoader:
    def __init__(self, snapshots: dict[str, SimpleNamespace]) -> None:
        self._snapshots = snapshots

    def load(self, run_path):
        return self._snapshots[str(run_path)]


def _snapshot(
    *,
    run_id: str = "run-0001",
    state: str = "success",
    events: tuple[dict[str, object], ...] = (),
    artifacts: tuple[dict[str, object], ...] = (),
    final: dict[str, object] | None = None,
    evidence_requirements: dict[str, object] | None = None,
    repositories: dict[str, object] | None = None,
    run_kind: str = "hybrid_live",
    integrity_error: Exception | None = None,
) -> SimpleNamespace:
    def verify_artifacts() -> None:
        if integrity_error is not None:
            raise integrity_error

    normalized_final = final
    if final is not None:
        metrics = dict(final.get("metrics", {}))
        if "evaluation_scope" not in metrics:
            metrics["evaluation_scope"] = (
                "training_internal"
                if run_kind == "training"
                else "held_out_experiment"
            )
        normalized_final = {**final, "metrics": metrics}
    run_dir = (
        Path(str(artifacts[0]["source_path"])).parent
        if artifacts
        else None
    )
    return SimpleNamespace(
        run_id=run_id,
        run_dir=run_dir,
        state=state,
        start={
            "schema_version": "experiment_run.v1",
            "run_id": run_id,
            "run_kind": run_kind,
            "task_context": {
                "task_variant": "load_truck",
                "soil_reset_block_id": "soil-block-03",
                "dig_point_id": "dig_02",
                "operator_id": "zhaoshuai",
                "material_id": "dry-soil",
            },
            "policy_ids": {
                "digging_policy": "act:swing-zero-200k",
                "trajectory_controller": "rl:7496592",
            },
            "repositories": repositories or {},
            "evidence_requirements": evidence_requirements or {},
        },
        events=events,
        artifacts=artifacts,
        final=normalized_final,
        manifest={} if normalized_final is not None else None,
        verify_artifacts=verify_artifacts,
    )


def _event(
    sequence: int,
    event_type: str,
    monotonic_ns: int,
    **payload: object,
) -> dict[str, object]:
    return {
        "schema_version": "experiment_run_event.v1",
        "sequence": sequence,
        "event_type": event_type,
        "wall_time_utc": f"2026-08-23T00:00:{sequence:02d}Z",
        "monotonic_ns": monotonic_ns,
        "payload": payload,
    }


def _artifact(sequence: int, role: str, path, **metadata: object) -> dict[str, object]:
    return {
        "schema_version": "experiment_run_artifact.v2",
        "sequence": sequence,
        "artifact_id": f"artifact-{sequence}",
        "role": role,
        "source_path": str(path.resolve()),
        "snapshot_path": path.name,
        "snapshot_method": "copy",
        "object_type": "directory" if path.is_dir() else "file",
        "sha256": "fixture-sha256",
        "size_bytes": 1,
        "file_count": 1,
        "metadata": metadata,
    }


def test_completed_run_reports_grouping_task_result_and_event_durations(tmp_path):
    events = (
        _event(0, "cycle_started", 1_000_000_000, cycle_index=0),
        _event(
            1,
            "phase_started",
            1_100_000_000,
            phase="act_dig",
            cycle_index=0,
        ),
        _event(
            2,
            "phase_completed",
            2_600_000_000,
            phase="act_dig",
            cycle_index=0,
        ),
        _event(3, "cycle_completed", 3_000_000_000, cycle_index=0),
    )
    snapshot = _snapshot(
        events=events,
        final={
            "status": "success",
            "metrics": {"requested_cycles": 1, "completed_cycles": 1},
            "summary": "one loading cycle completed",
        },
    )

    summary = evaluate_experiment_run(tmp_path / "run-0001", loader=_Loader(snapshot))

    assert summary["schema_version"] == "evaluation_summary.v1"
    assert summary["run_id"] == "run-0001"
    assert summary["passed"] is True
    assert summary["grouping"] == {
        "method": "digging_policy=act|trajectory_controller=rl",
        "policy": (
            "digging_policy=act:swing-zero-200k|"
            "trajectory_controller=rl:7496592"
        ),
        "task_variant": "load_truck",
        "soil_reset_block_id": "soil-block-03",
        "dig_point_id": "dig_02",
    }
    assert summary["task"] == {
        "success": True,
        "requested_cycles": 1,
        "completed_cycles": 1,
        "cycle_success_rate": 1.0,
    }
    assert summary["timing"]["cycles"] == {
        "count": 1,
        "duration_s": {"p50": 2.0, "p95": 2.0, "max": 2.0, "total": 2.0},
    }
    assert summary["timing"]["phases"]["act_dig"] == {
        "count": 1,
        "duration_s": {"p50": 1.5, "p95": 1.5, "max": 1.5, "total": 1.5},
    }
    assert summary["measurements"]["payload_mass_kg"]["value"] is None
    assert (
        summary["measurements"]["payload_mass_kg"]["unavailable_reason"]
        == "not explicitly measured for this run"
    )
    assert summary["data"]["cameras"]["camera_front"] == {
        "available": False,
        "unavailable_reason": "no front camera quality evidence was registered",
    }
    assert summary["data"]["cameras"]["camera_dump"] == {
        "available": False,
        "unavailable_reason": "no dump camera quality evidence was registered",
    }


def test_phase_completion_may_omit_cycle_when_the_match_is_unambiguous(tmp_path):
    snapshot = _snapshot(
        events=(
            _event(
                0,
                "phase_started",
                1_000_000_000,
                phase="act_dig",
                cycle_index=2,
            ),
            _event(1, "phase_completed", 2_000_000_000, phase="act_dig"),
        ),
        final={"status": "success", "metrics": {}, "summary": "done"},
    )

    summary = evaluate_experiment_run(tmp_path / "run", loader=_Loader(snapshot))

    assert summary["timing"]["phases"]["act_dig"]["duration_s"]["p50"] == 1.0


def test_missing_declared_required_artifact_fails_closed(tmp_path):
    snapshot = _snapshot(
        final={"status": "success", "metrics": {}, "summary": "done"},
        evidence_requirements={
            "raw_episode": {"required": True, "min_count": 1},
            "act_runtime_log": {"required": False, "min_count": 0},
        },
    )

    summary = evaluate_experiment_run(tmp_path / "run-0001", loader=_Loader(snapshot))

    assert summary["passed"] is False
    assert summary["failure_reasons"] == [
        "required artifact role raw_episode expected at least 1 record(s); observed 0"
    ]
    assert summary["evidence"]["missing_required_roles"] == ["raw_episode"]


@pytest.mark.parametrize(
    ("role", "metadata"),
    (
        ("runtime_log", {"analyzer": "other", "mode": "shadow"}),
        ("mission_log", {"analyzer": "other"}),
    ),
)
def test_required_supported_artifact_needs_its_canonical_analyzer(
    tmp_path,
    role,
    metadata,
):
    artifact_path = tmp_path / f"{role}.log"
    artifact_path.write_text("registered but not canonical\n", encoding="utf-8")
    snapshot = _snapshot(
        artifacts=(_artifact(0, role, artifact_path, **metadata),),
        evidence_requirements={role: {"required": True, "min_count": 1}},
        final={"status": "success", "metrics": {}, "summary": "done"},
    )

    summary = evaluate_experiment_run(tmp_path / "run", loader=_Loader(snapshot))

    assert summary["passed"] is False
    assert summary["evidence"]["evaluated_role_counts"][role] == 0
    assert summary["evidence"]["unevaluated_required_roles"] == [role]
    assert any(
        f"required artifact role {role}" in reason
        and "canonically evaluated" in reason
        for reason in summary["failure_reasons"]
    )


@pytest.mark.parametrize(
    "role",
    (
        "act_deployment_manifest",
        "act_policy_checkpoint",
        "rl_onnx_model",
    ),
)
def test_required_policy_artifact_is_canonically_integrity_evaluated(
    tmp_path,
    role,
):
    artifact_path = tmp_path / role
    artifact_path.write_bytes(b"immutable policy evidence\n")
    snapshot = _snapshot(
        artifacts=(_artifact(0, role, artifact_path),),
        evidence_requirements={role: {"required": True, "min_count": 1}},
        final={"status": "success", "metrics": {}, "summary": "done"},
    )

    summary = evaluate_experiment_run(tmp_path / "run", loader=_Loader(snapshot))

    assert summary["passed"] is True
    assert summary["evidence"]["evaluated_roles"] == [role]
    assert summary["evidence"]["evaluated_role_counts"][role] == 1
    assert summary["evidence"]["integrity_verified"] == {
        "artifact_set": True,
        "roles": [role],
        "role_counts": {role: 1},
    }
    assert summary["data"]["episode_count"] == 0
    assert summary["runtime"]["act"]["available"] is False
    assert summary["handoffs"]["available"] is False


def test_unknown_required_artifact_role_is_rejected(tmp_path):
    role = "future_policy_bundle"
    artifact_path = tmp_path / role
    artifact_path.write_bytes(b"unknown evidence\n")
    snapshot = _snapshot(
        artifacts=(_artifact(0, role, artifact_path),),
        evidence_requirements={role: {"required": True, "min_count": 1}},
        final={"status": "success", "metrics": {}, "summary": "done"},
    )

    summary = evaluate_experiment_run(tmp_path / "run", loader=_Loader(snapshot))

    assert summary["passed"] is False
    assert summary["evidence"]["unevaluated_required_roles"] == [role]
    assert any(
        f"required artifact role {role}" in reason
        and "not supported by the canonical evaluation harness" in reason
        for reason in summary["failure_reasons"]
    )


@pytest.mark.parametrize("role", ("act_policy_checkpoint", "quality_report"))
def test_integrity_only_artifact_failure_is_not_counted_as_evaluation(
    tmp_path,
    role,
):
    artifact_path = tmp_path / "policy.safetensors"
    artifact_path.write_bytes(b"drifted policy\n")
    snapshot = _snapshot(
        artifacts=(_artifact(0, role, artifact_path),),
        evidence_requirements={role: {"required": True, "min_count": 1}},
        final={"status": "success", "metrics": {}, "summary": "done"},
        integrity_error=RuntimeError("policy fingerprint mismatch"),
    )

    summary = evaluate_experiment_run(tmp_path / "run", loader=_Loader(snapshot))

    assert summary["passed"] is False
    assert summary["evidence"]["evaluated_role_counts"][role] == 0
    assert summary["evidence"]["unevaluated_required_roles"] == [role]
    assert summary["evidence"]["integrity_verified"] == {
        "artifact_set": False,
        "roles": [],
        "role_counts": {},
    }
    assert summary["failure_reasons"][0] == (
        "artifact integrity verification failed: policy fingerprint mismatch"
    )


def test_safety_events_count_interventions_and_fail_on_abort_or_post_terminal_motion(
    tmp_path,
):
    events = (
        _event(0, "operator_intervention", 1_000_000_000, reason="manual_stop"),
        _event(1, "runtime_abort", 1_100_000_000, runtime="act"),
        _event(
            2,
            "post_terminal_nonzero_command",
            1_200_000_000,
            command_seq=42,
        ),
    )
    snapshot = _snapshot(
        events=events,
        final={"status": "success", "metrics": {}, "summary": "unsafe fixture"},
    )

    summary = evaluate_experiment_run(tmp_path / "run-0001", loader=_Loader(snapshot))

    assert summary["safety"] == {
        "intervention_count": 1,
        "runtime_abort_count": 1,
        "post_terminal_nonzero_count": 1,
    }
    assert summary["passed"] is False
    assert summary["failure_reasons"] == [
        "runtime abort event count is 1",
        "post-terminal nonzero command event count is 1",
    ]


def test_aggregate_is_deterministic_and_never_mixes_training_with_held_out_runs(
    tmp_path,
):
    training = _snapshot(
        run_id="run-training",
        final={
            "status": "success",
            "metrics": {
                "evaluation_scope": "training_internal",
                "requested_cycles": 2,
                "completed_cycles": 2,
            },
            "summary": "training-only check",
        },
    )
    held_out = _snapshot(
        run_id="run-held-out",
        final={
            "status": "success",
            "metrics": {
                "evaluation_scope": "held_out_experiment",
                "requested_cycles": 1,
                "completed_cycles": 1,
            },
            "summary": "formal trial",
        },
    )
    paths = (tmp_path / "run-training", tmp_path / "run-held-out")
    loader = _MappingLoader(
        {str(paths[0]): training, str(paths[1]): held_out}
    )

    with pytest.raises(
        EvaluationError,
        match="homogeneous aggregate cannot mix evaluation scopes",
    ):
        evaluate_experiment_runs(reversed(paths), loader=loader)

    aggregate = evaluate_experiment_runs((paths[1],), loader=loader)

    assert aggregate["schema_version"] == "evaluation_aggregate.v1"
    assert [item["run_id"] for item in aggregate["runs"]] == ["run-held-out"]
    assert [group["evaluation_scope"] for group in aggregate["groups"]] == [
        "held_out_experiment"
    ]
    assert [group["run_count"] for group in aggregate["groups"]] == [1]
    assert aggregate["scope_policy"] == (
        "one evaluation_scope per aggregate; training_internal and "
        "held_out_experiment are never mixed"
    )

    json_path = tmp_path / "evaluation.json"
    csv_path = tmp_path / "evaluation.csv"
    write_evaluation_outputs(aggregate, json_path=json_path, csv_path=csv_path)
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()
    write_evaluation_outputs(aggregate, json_path=json_path, csv_path=csv_path)
    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv
    csv_lines = first_csv.decode("utf-8").splitlines()
    assert "unavailable_reasons_json" in csv_lines[0]
    assert csv_lines[1].startswith("run-held-out,")


def test_evaluation_scope_is_required_for_every_finalized_run(tmp_path):
    snapshot = _snapshot(
        final={
            "status": "success",
            "metrics": {"evaluation_scope": None},
            "summary": "scope omitted",
        }
    )

    with pytest.raises(
        EvaluationError,
        match="final.metrics.evaluation_scope is required",
    ):
        evaluate_experiment_run(tmp_path / "run", loader=_Loader(snapshot))


def test_dirty_repository_fails_held_out_but_is_only_reported_for_training(
    tmp_path,
):
    repositories = {
        "excavator_il": {
            "source_path": str(tmp_path),
            "commit": "a" * 40,
            "dirty": True,
        }
    }
    held_out = _snapshot(
        repositories=repositories,
        final={"status": "success", "metrics": {}, "summary": "formal trial"},
    )
    training = _snapshot(
        run_kind="training",
        repositories=repositories,
        final={"status": "success", "metrics": {}, "summary": "pilot"},
    )

    held_out_summary = evaluate_experiment_run(
        tmp_path / "held-out",
        loader=_Loader(held_out),
    )
    training_summary = evaluate_experiment_run(
        tmp_path / "training",
        loader=_Loader(training),
    )

    assert held_out_summary["passed"] is False
    assert held_out_summary["reproducibility"]["dirty_repositories"] == [
        "excavator_il"
    ]
    assert "held_out_experiment" in held_out_summary["failure_reasons"][0]
    assert training_summary["passed"] is True
    assert training_summary["reproducibility"]["dirty_repository_count"] == 1


def test_artifact_fingerprint_drift_fails_closed_without_consuming_artifact(tmp_path):
    snapshot = _snapshot(
        artifacts=(_artifact(0, "raw_episode", tmp_path / "missing"),),
        final={"status": "success", "metrics": {}, "summary": "done"},
        integrity_error=RuntimeError("artifact artifact-0 fingerprint mismatch"),
    )

    summary = evaluate_experiment_run(tmp_path / "run-0001", loader=_Loader(snapshot))

    assert summary["passed"] is False
    assert summary["evidence"]["artifacts_verified"] is False
    assert summary["data"]["episode_count"] == 0
    assert summary["failure_reasons"] == [
        "artifact integrity verification failed: "
        "artifact artifact-0 fingerprint mismatch"
    ]


def test_run_kind_and_aggregate_mode_are_strict(tmp_path):
    invalid = _snapshot(
        run_kind="future_kind",
        final={"status": "success", "metrics": {}, "summary": "done"},
    )
    with pytest.raises(EvaluationError, match="start.run_kind must be one of"):
        evaluate_experiment_run(tmp_path / "invalid", loader=_Loader(invalid))

    hybrid = _snapshot(
        run_id="hybrid",
        final={
            "status": "success",
            "metrics": {"evaluation_scope": "held_out_experiment"},
            "summary": "done",
        },
    )
    training = _snapshot(
        run_id="training",
        run_kind="training",
        final={
            "status": "success",
            "metrics": {"evaluation_scope": "training_internal"},
            "summary": "done",
        },
    )
    paths = (tmp_path / "hybrid", tmp_path / "training")
    loader = _MappingLoader({str(paths[0]): hybrid, str(paths[1]): training})

    with pytest.raises(EvaluationError, match="homogeneous aggregate"):
        evaluate_experiment_runs(paths, loader=loader)
    with pytest.raises(EvaluationError, match="live_task aggregate"):
        evaluate_experiment_runs(paths, loader=loader, aggregate_mode="live_task")

    live = evaluate_experiment_runs(
        (paths[0],), loader=loader, aggregate_mode="live_task"
    )
    assert live["aggregate_mode"] == "live_task"


def test_snapshot_state_must_match_final_status(tmp_path):
    snapshot = _snapshot(
        state="failure",
        final={"status": "success", "metrics": {}, "summary": "inconsistent"},
    )

    with pytest.raises(EvaluationError, match="snapshot.state and final.status"):
        evaluate_experiment_run(tmp_path / "run", loader=_Loader(snapshot))


def test_collection_dataset_mode_rejects_non_collection_run(tmp_path):
    snapshot = _snapshot(
        final={"status": "success", "metrics": {}, "summary": "done"}
    )
    path = tmp_path / "hybrid"

    with pytest.raises(
        EvaluationError, match="collection_dataset aggregate accepts only"
    ):
        evaluate_experiment_runs(
            (path,), loader=_Loader(snapshot), aggregate_mode="collection_dataset"
        )

    training = _snapshot(
        run_id="collection-training",
        run_kind="collection_episode",
        final={
            "status": "success",
            "metrics": {"evaluation_scope": "training_internal"},
            "summary": "pilot collection",
        },
    )
    held_out = _snapshot(
        run_id="collection-held-out",
        run_kind="collection_episode",
        final={"status": "success", "metrics": {}, "summary": "formal data"},
    )
    collection_paths = (tmp_path / "training", tmp_path / "held-out")
    collection_loader = _MappingLoader(
        {
            str(collection_paths[0]): training,
            str(collection_paths[1]): held_out,
        }
    )
    with pytest.raises(
        EvaluationError,
        match="collection_dataset aggregate cannot mix evaluation scopes",
    ):
        evaluate_experiment_runs(
            collection_paths,
            loader=collection_loader,
            aggregate_mode="collection_dataset",
        )


def test_external_ratios_must_be_explicit_and_within_unit_interval(tmp_path):
    snapshot = _snapshot(
        final={
            "status": "success",
            "metrics": {
                "external_measurements": {
                    "fill_ratio": {
                        "value": 1.2,
                        "unit": "ratio",
                        "method": "image_annotation",
                        "source": "annotation-001",
                    }
                }
            },
            "summary": "invalid annotation",
        }
    )

    with pytest.raises(EvaluationError, match="fill_ratio.value must be at most 1"):
        evaluate_experiment_run(tmp_path / "run", loader=_Loader(snapshot))


def test_real_experiment_loader_and_cli_write_finalized_run_outputs(tmp_path):
    profile = tmp_path / "machine_profile.json"
    profile.write_text('{"action_order":["boom","stick","bucket","swing"]}\n')
    run = ExperimentRun.create(
        tmp_path / "evidence",
        run_id="run_eval_001",
        run_kind="hybrid_live",
        task_context=TaskContext(
            task_variant="load_truck",
            soil_reset_block_id="block_01",
            dig_point_id="dig_01",
            operator_id="zhaoshuai",
            material_id="dry_soil",
        ),
        policy_ids={"digging_policy": "act:checkpoint-200k"},
        host_topology={},
        repository_paths={},
        config_paths={},
        machine_profile_path=profile,
    )
    run.append_event("phase_started", {"phase": "act_dig", "cycle_index": 0})
    run.append_event("phase_completed", {"phase": "act_dig"})
    snapshot = run.finalize(
        "success",
        metrics={
            "evaluation_scope": "held_out_experiment",
            "requested_cycles": 1,
            "completed_cycles": 1,
        },
        summary="formal trial",
    )

    summary = evaluate_experiment_run(snapshot.run_dir)

    assert summary["run_id"] == "run_eval_001"
    assert summary["evidence"]["artifacts_verified"] is True
    output = tmp_path / "outputs"
    repository = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(repository / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "evaluate_experiment_runs.py"),
            str(snapshot.run_dir),
            "--output-dir",
            str(output),
            "--aggregate-mode",
            "live_task",
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "evaluation_aggregate.json").is_file()
    assert (output / "evaluation_aggregate.csv").is_file()
    assert json.loads(result.stdout)["run_count"] == 1


def test_real_experiment_run_integrity_evaluates_all_hybrid_policy_artifacts(
    tmp_path,
):
    profile = tmp_path / "machine_profile.json"
    profile.write_text('{"action_order":["boom","stick","bucket","swing"]}\n')
    roles = (
        "act_deployment_manifest",
        "act_policy_checkpoint",
        "rl_onnx_model",
    )
    run = ExperimentRun.create(
        tmp_path / "evidence",
        run_id="run_eval_policy_artifacts",
        run_kind="hybrid_live",
        task_context=TaskContext(
            task_variant="load_truck",
            soil_reset_block_id="block_01",
            dig_point_id="dig_01",
            operator_id="zhaoshuai",
            material_id="dry_soil",
        ),
        policy_ids={
            "digging_policy": "act:checkpoint-200k",
            "trajectory_controller": "rl:7496592",
        },
        host_topology={},
        repository_paths={},
        config_paths={},
        machine_profile_path=profile,
        evidence_requirements={
            role: {"required": True, "min_count": 1} for role in roles
        },
    )
    for sequence, role in enumerate(roles):
        policy_artifact = tmp_path / f"{role}.bin"
        policy_artifact.write_bytes(f"{role}\n".encode("utf-8"))
        run.register_artifact(
            f"artifact-{sequence}",
            policy_artifact,
            role=role,
        )
    snapshot = run.finalize(
        "success",
        metrics={"evaluation_scope": "held_out_experiment"},
        summary="policy artifacts captured",
    )

    summary = evaluate_experiment_run(snapshot.run_dir)

    assert summary["passed"] is True
    assert summary["evidence"]["integrity_verified"] == {
        "artifact_set": True,
        "roles": sorted(roles),
        "role_counts": {role: 1 for role in sorted(roles)},
    }

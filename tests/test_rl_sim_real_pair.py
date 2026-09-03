from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from excavator_il.experiment_run import ExperimentRun, TaskContext
from excavator_il.rl_sim_experiment_run import (
    RlSimExperimentRunRequest,
    record_rl_sim_experiment_run,
)
from excavator_il.rl_sim_real_pair import (
    evaluate_rl_sim_real_pair,
    write_rl_sim_real_pair_manifest,
)


TRACE_RUN_ID = "follow-run-real-001"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_trace(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _sample(
    sample_id: int,
    stamp_s: float,
    action: list[float],
    *,
    bucket_tip_ros_m: list[float] | None = None,
    reference_waypoint_ros_m: list[float] | None = None,
    waypoint_index: int = 0,
    waypoint_distance_m: float | None = None,
    episode_progress: float = 0.0,
) -> dict[str, object]:
    bucket_tip = bucket_tip_ros_m or [0.0, 0.0, 0.0]
    reference_waypoint = reference_waypoint_ros_m or [1.0, 0.0, 0.0]
    return {
        "schema_version": "excavator_rl_control_trace.v3",
        "record_type": "policy_sample",
        "sample_id": sample_id,
        "stamp_s": stamp_s,
        "action_order": ["boom", "stick", "bucket", "swing"],
        "action": action,
        "trace_semantics": "commanded_normalized_action",
        "bucket_tip_ros_m": bucket_tip,
        "reference_waypoint_ros_m": reference_waypoint,
        "waypoint_index": waypoint_index,
        "waypoint_distance_m": (
            math.dist(bucket_tip, reference_waypoint)
            if waypoint_distance_m is None
            else waypoint_distance_m
        ),
        "episode_progress": episode_progress,
        "result": "ACTIVE",
    }


def _terminal(stamp_s: float, elapsed_s: float, result: str) -> dict[str, object]:
    return {
        "schema_version": "excavator_rl_control_trace.v3",
        "record_type": "terminal",
        "stamp_s": stamp_s,
        "elapsed_s": elapsed_s,
        "trace_semantics": "commanded_normalized_action",
        "result": result,
    }


def _record_sim_run(
    tmp_path: Path,
    *,
    policy_id: str = "onnx_rl:scale_v3_deadzone_reward_03_p003",
    machine_profile: Path | None = None,
    onnx_bytes: bytes = b"shared-onnx",
    trace_records: list[dict[str, object]] | None = None,
) -> object:
    profile_path = machine_profile or _write_json(
        tmp_path / "machine_profile.sim.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    trajectory_suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {
            "suite_id": "suite-001",
            "sample_period_s": 0.1,
            "sample_ids": [0, 1, 2, 3],
        },
    )
    trajectory_suite_sha256 = hashlib.sha256(trajectory_suite.read_bytes()).hexdigest()
    selected_trace_records = trace_records or [
        _sample(0, 0.0, [0.0, 0.0, 0.0, 0.0], waypoint_distance_m=1.0),
        _sample(1, 0.1, [0.0, 1.0, 0.0, -1.0], bucket_tip_ros_m=[0.5, 0.0, 0.0], waypoint_distance_m=0.5, episode_progress=0.5),
        _sample(2, 0.2, [1.0, 0.0, 0.0, 0.0], bucket_tip_ros_m=[1.0, 0.0, 0.0], waypoint_distance_m=0.0, episode_progress=1.0),
        _terminal(0.3, 0.3, "COMPLETED"),
    ]
    trace_path = _write_trace(
        tmp_path / "simulation_trace.jsonl",
        [
            {
                **record,
                "trajectory_suite_sha256": record.get(
                    "trajectory_suite_sha256", trajectory_suite_sha256
                ),
            }
            for record in selected_trace_records
        ],
    )
    onnx_path = tmp_path / "controller.onnx"
    onnx_path.write_bytes(onnx_bytes)
    return record_rl_sim_experiment_run(
        RlSimExperimentRunRequest(
            experiment_run_root=tmp_path / "evidence",
            machine_profile_path=profile_path,
            trajectory_suite_path=trajectory_suite,
            trajectory_controller_onnx_path=onnx_path,
            trace_path=trace_path,
            policy_id=policy_id,
            evaluation_scope="held_out_experiment",
            task_variant="dig_transport_dump",
            operator_id="zhaoshuai",
            material_id="soil_default",
            run_id="rl_sim_parent",
        )
    )


def _record_real_run(
    tmp_path: Path,
    *,
    policy_id: str = "onnx_rl:scale_v3_deadzone_reward_03_p003",
    machine_profile: Path,
    onnx_path: Path,
    trace_records: list[dict[str, object]] | None = None,
    run_id: str = "real_parent",
    include_suite: bool = True,
    include_audit: bool = True,
    evaluation_scope: str = "held_out_experiment",
) -> object:
    trajectory_suite_sha256 = hashlib.sha256(
        (tmp_path / "trajectory_suite.json").read_bytes()
    ).hexdigest()
    selected_trace_records = trace_records or [
        _sample(0, 0.05, [0.0, 0.0, 0.0, 0.0], bucket_tip_ros_m=[0.1, 0.0, 0.0], waypoint_distance_m=0.9),
        _sample(1, 0.16, [0.0, 1.0, 0.0, 1.0], bucket_tip_ros_m=[0.4, 0.0, 0.0], reference_waypoint_ros_m=[1.1, 0.0, 0.0], waypoint_index=1, waypoint_distance_m=0.7, episode_progress=0.6),
        _sample(2, 0.28, [0.0, 0.0, 0.0, 0.0], bucket_tip_ros_m=[0.8, 0.0, 0.0], waypoint_distance_m=0.2, episode_progress=0.55),
        _terminal(0.35, 0.3, "COMPLETED"),
    ]
    trace_path = _write_trace(
        tmp_path / f"{run_id}.jsonl",
        [
            {
                **record,
                "trajectory_suite_sha256": record.get(
                    "trajectory_suite_sha256", trajectory_suite_sha256
                ),
            }
            for record in selected_trace_records
        ],
    )
    run = ExperimentRun.create(
        tmp_path / "evidence",
        run_id=run_id,
        run_kind="hybrid_live",
        task_context=TaskContext(
            task_variant="dig_transport_dump",
            soil_reset_block_id=None,
            dig_point_id="dig_01",
            operator_id="zhaoshuai",
            material_id="soil_default",
        ),
        policy_ids={"trajectory_controller": policy_id},
        host_topology={},
        repository_paths={},
        config_paths={},
        machine_profile_path=machine_profile,
        evidence_requirements={
            "rl_onnx_model": {"required": True, "min_count": 1},
            "trajectory_trace": {"required": True, "min_count": 1},
            **(
                {"rl_control_audit": {"required": True, "min_count": 1}}
                if include_audit
                else {}
            ),
            **(
                {"trajectory_suite": {"required": True, "min_count": 1}}
                if include_suite
                else {}
            ),
        },
    )
    run.register_artifact(
        "rl-model",
        onnx_path,
        role="rl_onnx_model",
        metadata={"policy": "trajectory_controller"},
    )
    if include_suite:
        run.register_artifact(
            "trajectory-suite",
            tmp_path / "trajectory_suite.json",
            role="trajectory_suite",
        )
    audit_record = None
    if include_audit:
        audit_path = _write_trace(
            tmp_path / f"{run_id}.audit.jsonl",
            [
                {
                    "schema_version": "orin_edge_control_audit.v1",
                    "mode": "control",
                    "status": "active",
                    "trajectory_controller_backend": "onnx_rl",
                    "trace_run_id": TRACE_RUN_ID,
                }
            ],
        )
        audit_record = run.register_artifact(
            "rl-control-audit",
            audit_path,
            role="rl_control_audit",
            metadata={
                "schema_version": "orin_edge_control_audit.v1",
                "trajectory_controller_backend": "onnx_rl",
                "trace_semantics": "commanded_normalized_action",
                "trace_run_id": TRACE_RUN_ID,
                "trajectory_suite_sha256": trajectory_suite_sha256,
            },
        )
    run.register_artifact(
        "trajectory-trace",
        trace_path,
        role="trajectory_trace",
        metadata={
            "trace_semantics": "commanded_normalized_action",
            "trace_run_id": TRACE_RUN_ID,
            "trajectory_suite_sha256": trajectory_suite_sha256,
            **(
                {"source_audit_sha256": audit_record["sha256"]}
                if audit_record is not None
                else {}
            ),
        },
    )
    return run.finalize(
        "success",
        metrics={"evaluation_scope": evaluation_scope},
        summary="real machine parent",
    )


def test_evaluate_rl_sim_real_pair_reports_alignment_and_axis_metrics(tmp_path: Path):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    onnx_path = tmp_path / "controller.onnx"
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=onnx_path,
    )
    manifest_path = write_rl_sim_real_pair_manifest(
        tmp_path / "pair_manifest.json",
        simulation_run_path=sim_snapshot.run_dir,
        real_run_path=real_snapshot.run_dir,
        pair_id="pair-001",
    )

    report = evaluate_rl_sim_real_pair(manifest_path)

    assert report["pair_id"] == "pair-001"
    assert report["trace_semantics"] == "commanded_normalized_action"
    assert report["binding"]["trajectory_suite_sample_period_s"] == 0.1
    assert report["binding"]["trajectory_trace_schema_version"] == (
        "excavator_rl_control_trace.v3"
    )
    assert report["trace_sha256"] == {
        "simulation": next(
            artifact["sha256"]
            for artifact in sim_snapshot.artifacts
            if artifact["role"] == "trajectory_trace"
        ),
        "real_machine": next(
            artifact["sha256"]
            for artifact in real_snapshot.artifacts
            if artifact["role"] == "trajectory_trace"
        ),
    }
    assert report["simulation_sample_count"] == 3
    assert report["real_machine_sample_count"] == 3
    assert report["aligned_sample_count"] == 3
    assert report["simulation_only_tail_count"] == 0
    assert report["simulation_only_tail_sample_ids"] == []
    assert report["real_machine_only_tail_count"] == 0
    assert report["real_machine_only_tail_sample_ids"] == []
    assert report["sample_coverage"] == {
        "simulation": {"consumed_count": 3, "suite_count": 4, "rate": 0.75},
        "real_machine": {"consumed_count": 3, "suite_count": 4, "rate": 0.75},
    }
    assert report["duration_s"] == {"simulation": 0.3, "real_machine": 0.3}
    assert report["nonzero_agreement_rate"] == pytest.approx(2 / 3)
    assert report["axes"]["boom"] == {
        "mae": pytest.approx(1 / 3),
        "rmse": pytest.approx(3 ** -0.5),
        "max_abs": 1.0,
        "sign_agreement_rate": pytest.approx(2 / 3),
    }
    assert report["axes"]["stick"] == {
        "mae": 0.0,
        "rmse": 0.0,
        "max_abs": 0.0,
        "sign_agreement_rate": 1.0,
    }
    assert report["axes"]["bucket"] == {
        "mae": 0.0,
        "rmse": 0.0,
        "max_abs": 0.0,
        "sign_agreement_rate": 1.0,
    }
    assert report["axes"]["swing"] == {
        "mae": pytest.approx(2 / 3),
        "rmse": pytest.approx(2 / 3 ** 0.5),
        "max_abs": 2.0,
        "sign_agreement_rate": pytest.approx(2 / 3),
    }
    assert report["tracking"]["bucket_tip_euclidean_error_m"] == {
        "mae": pytest.approx(0.4 / 3),
        "rmse": pytest.approx((0.06 / 3) ** 0.5),
        "max": 0.2,
    }
    assert report["tracking"]["reference_waypoint_euclidean_error_m"] == {
        "mae": pytest.approx(0.1 / 3),
        "rmse": pytest.approx((0.01 / 3) ** 0.5),
        "max": 0.1,
    }
    assert report["tracking"]["waypoint_index_agreement"] == {
        "count": 2,
        "rate": pytest.approx(2 / 3),
    }
    assert report["tracking"]["relative_sample_timing_error_s"] == {
        "mae": pytest.approx(0.04 / 3),
        "rmse": pytest.approx((0.001 / 3) ** 0.5),
        "max_abs": 0.03,
    }
    assert report["tracking"]["waypoint_distance_m"] == {
        "simulation": {"mean": 0.5, "p95": 0.95, "final": 0.0},
        "real_machine": {"mean": 0.6, "p95": 0.88, "final": 0.2},
    }
    assert report["tracking"]["terminal_result"] == {
        "agreement": True,
        "simulation": "COMPLETED",
        "real_machine": "COMPLETED",
        "completed_count": {"simulation": 1, "real_machine": 1},
        "timeout_count": {"simulation": 0, "real_machine": 0},
        "rejected_count": {"simulation": 0, "real_machine": 0},
        "interrupted_count": {"simulation": 0, "real_machine": 0},
    }


def test_pair_evaluator_aligns_common_prefix_and_reports_unaligned_tail(
    tmp_path: Path,
):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
        trace_records=[
            _sample(0, 0.05, [0.0, 1.0, 0.0, 1.0]),
            _sample(1, 0.15, [0.0, 0.0, 0.0, 0.0]),
            _terminal(0.25, 0.2, "COMPLETED"),
        ],
    )

    manifest_path = write_rl_sim_real_pair_manifest(
        tmp_path / "pair_manifest.json",
        simulation_run_path=sim_snapshot.run_dir,
        real_run_path=real_snapshot.run_dir,
        pair_id="pair-001",
    )
    report = evaluate_rl_sim_real_pair(manifest_path)

    assert report["simulation_sample_count"] == 3
    assert report["real_machine_sample_count"] == 2
    assert report["aligned_sample_count"] == 2
    assert report["simulation_only_tail_count"] == 1
    assert report["simulation_only_tail_sample_ids"] == [2]
    assert report["real_machine_only_tail_count"] == 0
    assert report["real_machine_only_tail_sample_ids"] == []


@pytest.mark.parametrize(
    ("modifier", "message"),
    [
        (
            lambda manifest, tmp_path: manifest["binding"].__setitem__(
                "trajectory_controller_policy_id", "onnx_rl:different"
            ),
            "trajectory controller policy binding does not match parent runs",
        ),
        (
            lambda manifest, tmp_path: manifest["binding"].__setitem__(
                "trajectory_controller_onnx_sha256", "0" * 64
            ),
            "trajectory controller ONNX binding does not match parent runs",
        ),
        (
            lambda manifest, tmp_path: manifest["binding"].__setitem__(
                "machine_profile_sha256", "1" * 64
            ),
            "machine profile binding does not match parent runs",
        ),
        (
            lambda manifest, tmp_path: manifest["binding"].__setitem__(
                "action_order", ["swing", "boom", "stick", "bucket"]
            ),
            "action order binding does not match parent runs",
        ),
        (
            lambda manifest, tmp_path: manifest["binding"].__setitem__(
                "trace_semantics", "normalized_action"
            ),
            "trajectory trace semantics do not match parent runs",
        ),
        (
            lambda manifest, tmp_path: manifest["binding"].__setitem__(
                "trajectory_trace_schema_version", "excavator_rl_control_trace.v2"
            ),
            "trajectory trace schema does not match parent runs",
        ),
        (
            lambda manifest, tmp_path: manifest["parent_runs"]["real_machine"].__setitem__(
                "manifest_sha256", hashlib.sha256(b"tampered").hexdigest()
            ),
            "parent manifest SHA does not match finalized run",
        ),
        (
            lambda manifest, tmp_path: manifest.__setitem__(
                "evaluation_scope", "training_internal"
            ),
            "evaluation scope does not match parent runs",
        ),
    ],
)
def test_evaluate_rl_sim_real_pair_rejects_drift(
    tmp_path: Path,
    modifier,
    message: str,
):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
    )
    manifest_path = write_rl_sim_real_pair_manifest(
        tmp_path / "pair_manifest.json",
        simulation_run_path=sim_snapshot.run_dir,
        real_run_path=real_snapshot.run_dir,
        pair_id="pair-001",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modifier(manifest, tmp_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        evaluate_rl_sim_real_pair(manifest_path)


def test_pair_manifest_rejects_real_parent_without_trajectory_suite(tmp_path: Path):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
        include_suite=False,
    )

    with pytest.raises(ValueError, match="exactly one trajectory_suite"):
        write_rl_sim_real_pair_manifest(
            tmp_path / "pair_manifest.json",
            simulation_run_path=sim_snapshot.run_dir,
            real_run_path=real_snapshot.run_dir,
            pair_id="pair-001",
        )


def test_pair_manifest_rejects_real_parent_without_raw_control_audit(tmp_path: Path):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
        include_audit=False,
    )

    with pytest.raises(ValueError, match="exactly one rl_control_audit"):
        write_rl_sim_real_pair_manifest(
            tmp_path / "pair_manifest.json",
            simulation_run_path=sim_snapshot.run_dir,
            real_run_path=real_snapshot.run_dir,
            pair_id="pair-001",
        )


def test_pair_manifest_rejects_real_trace_samples_outside_shared_suite(
    tmp_path: Path,
):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
        trace_records=[
            _sample(0, 0.0, [0.0, 0.0, 0.0, 0.0]),
            _sample(1, 0.1, [0.0, 0.0, 0.0, 0.0]),
            _sample(2, 0.2, [0.0, 0.0, 0.0, 0.0]),
            _sample(3, 0.3, [0.0, 0.0, 0.0, 0.0]),
            _sample(4, 0.4, [0.0, 0.0, 0.0, 0.0]),
            _terminal(0.5, 0.5, "COMPLETED"),
        ],
    )

    with pytest.raises(ValueError, match="outside its trajectory suite"):
        write_rl_sim_real_pair_manifest(
            tmp_path / "pair_manifest.json",
            simulation_run_path=sim_snapshot.run_dir,
            real_run_path=real_snapshot.run_dir,
            pair_id="pair-001",
        )


def test_pair_manifest_rejects_parent_evaluation_scope_mismatch(tmp_path: Path):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
        evaluation_scope="training_internal",
    )

    with pytest.raises(ValueError, match="evaluation scope does not match parent runs"):
        write_rl_sim_real_pair_manifest(
            tmp_path / "pair_manifest.json",
            simulation_run_path=sim_snapshot.run_dir,
            real_run_path=real_snapshot.run_dir,
            pair_id="pair-001",
        )


def test_pair_manifest_never_overwrites_existing_output(tmp_path: Path):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
    )
    output_path = tmp_path / "pair_manifest.json"
    output_path.write_text("keep-existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output already exists"):
        write_rl_sim_real_pair_manifest(
            output_path,
            simulation_run_path=sim_snapshot.run_dir,
            real_run_path=real_snapshot.run_dir,
            pair_id="pair-001",
        )

    assert output_path.read_text(encoding="utf-8") == "keep-existing\n"


@pytest.mark.parametrize("redirect_kind", ["absolute", "traversal"])
def test_evaluate_pair_rejects_trace_artifact_redirect_outside_parent_run(
    tmp_path: Path,
    redirect_kind: str,
):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
    )
    manifest_path = write_rl_sim_real_pair_manifest(
        tmp_path / "pair_manifest.json",
        simulation_run_path=sim_snapshot.run_dir,
        real_run_path=real_snapshot.run_dir,
        pair_id="pair-001",
    )
    external_trace = _write_trace(
        tmp_path / "external_trace.jsonl",
        [_sample(1, 0.1, [1.0, 1.0, 1.0, 1.0])],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["simulation"]["snapshot_path"] = (
        str(external_trace)
        if redirect_kind == "absolute"
        else os.path.relpath(external_trace, sim_snapshot.run_dir)
    )
    manifest["artifacts"]["simulation"]["sha256"] = hashlib.sha256(
        external_trace.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="trajectory trace artifact does not match finalized parent run",
    ):
        evaluate_rl_sim_real_pair(manifest_path)


def test_pair_manifest_rejects_different_real_trajectory_suite(tmp_path: Path):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    _write_json(
        tmp_path / "trajectory_suite.json",
        {
            "suite_id": "suite-other",
            "sample_period_s": 0.1,
            "sample_ids": [0, 1, 2, 3],
        },
    )
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
    )

    with pytest.raises(
        ValueError,
        match="trajectory suite binding does not match parent runs",
    ):
        write_rl_sim_real_pair_manifest(
            tmp_path / "pair_manifest.json",
            simulation_run_path=sim_snapshot.run_dir,
            real_run_path=real_snapshot.run_dir,
            pair_id="pair-001",
        )


def test_pair_manifest_rejects_trace_bound_to_different_suite_hash(
    tmp_path: Path,
):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    sim_snapshot = _record_sim_run(tmp_path, machine_profile=machine_profile)
    real_snapshot = _record_real_run(
        tmp_path,
        machine_profile=machine_profile,
        onnx_path=tmp_path / "controller.onnx",
        trace_records=[
            {
                **_sample(0, 0.1, [0.0, 0.0, 0.0, 0.0]),
                "trajectory_suite_sha256": "f" * 64,
            },
            {
                **_terminal(0.2, 0.2, "COMPLETED"),
                "trajectory_suite_sha256": "f" * 64,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="trajectory trace suite binding does not match parent run",
    ):
        write_rl_sim_real_pair_manifest(
            tmp_path / "pair_manifest.json",
            simulation_run_path=sim_snapshot.run_dir,
            real_run_path=real_snapshot.run_dir,
            pair_id="pair-001",
        )

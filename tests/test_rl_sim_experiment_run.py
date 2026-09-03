from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import excavator_il.rl_sim_experiment_run as rl_sim_module
from excavator_il.experiment_run import ExperimentRun, ExperimentRunValidationError
from excavator_il.rl_sim_experiment_run import (
    RlSimExperimentRunRequest,
    load_rl_control_trace,
    record_rl_sim_experiment_run,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_trace(path: Path, *, trajectory_suite_sha256: str) -> Path:
    records = (
        {
            "schema_version": "excavator_rl_control_trace.v3",
            "record_type": "policy_sample",
            "sample_id": 0,
            "stamp_s": 0.0,
            "action_order": ["boom", "stick", "bucket", "swing"],
            "action": [0.0, 0.0, 0.0, 0.0],
            "trace_semantics": "commanded_normalized_action",
            "trajectory_suite_sha256": trajectory_suite_sha256,
            "bucket_tip_ros_m": [0.8, 0.0, 0.0],
            "reference_waypoint_ros_m": [1.0, 0.0, 0.0],
            "waypoint_index": 0,
            "waypoint_distance_m": 0.2,
            "episode_progress": 0.0,
            "result": "ACTIVE",
        },
        {
            "schema_version": "excavator_rl_control_trace.v3",
            "record_type": "policy_sample",
            "sample_id": 1,
            "stamp_s": 0.1,
            "action_order": ["boom", "stick", "bucket", "swing"],
            "action": [0.1, 0.2, 0.0, -0.3],
            "trace_semantics": "commanded_normalized_action",
            "trajectory_suite_sha256": trajectory_suite_sha256,
            "bucket_tip_ros_m": [0.9, 0.0, 0.0],
            "reference_waypoint_ros_m": [1.0, 0.0, 0.0],
            "waypoint_index": 1,
            "waypoint_distance_m": 0.1,
            "episode_progress": 1.0,
            "result": "ACTIVE",
        },
        {
            "schema_version": "excavator_rl_control_trace.v3",
            "record_type": "terminal",
            "stamp_s": 0.2,
            "elapsed_s": 0.2,
            "trace_semantics": "commanded_normalized_action",
            "trajectory_suite_sha256": trajectory_suite_sha256,
            "result": "COMPLETED",
        },
    )
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_load_rl_control_trace_requires_real_tracking_evidence(tmp_path: Path):
    trace_path = tmp_path / "tracking_trace.jsonl"
    records = [
        {
                "schema_version": "excavator_rl_control_trace.v3",
                "record_type": "policy_sample",
                "sample_id": 0,
                "stamp_s": 1.25,
                "action_order": ["boom", "stick", "bucket", "swing"],
                "action": [0.1, -0.2, 0.3, -0.4],
                "trace_semantics": "commanded_normalized_action",
                "trajectory_suite_sha256": "a" * 64,
                "bucket_tip_ros_m": [0.8, 0.1, -0.2],
                "reference_waypoint_ros_m": [1.0, 0.2, -0.1],
                "waypoint_index": 2,
                "waypoint_distance_m": 0.244948974,
                "episode_progress": 0.75,
                "result": "ACTIVE",
        },
        {
            "schema_version": "excavator_rl_control_trace.v3",
            "record_type": "terminal",
            "stamp_s": 1.3,
            "elapsed_s": 1.3,
            "trace_semantics": "commanded_normalized_action",
            "trajectory_suite_sha256": "a" * 64,
            "result": "COMPLETED",
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    sample = load_rl_control_trace(trace_path)[0]

    assert sample.bucket_tip_ros_m == (0.8, 0.1, -0.2)
    assert sample.reference_waypoint_ros_m == (1.0, 0.2, -0.1)
    assert sample.waypoint_index == 2
    assert sample.waypoint_distance_m == pytest.approx(0.244948974)
    assert sample.episode_progress == 0.75
    assert sample.result == "ACTIVE"


def test_load_rl_control_trace_rejects_inconsistent_waypoint_distance(
    tmp_path: Path,
):
    trace_path = _write_trace(
        tmp_path / "tracking_trace.jsonl",
        trajectory_suite_sha256="a" * 64,
    )
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["waypoint_distance_m"] = 0.5
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="waypoint_distance_m must match bucket-tip to reference-waypoint distance",
    ):
        load_rl_control_trace(trace_path)


def test_load_rl_control_trace_requires_nonempty_contiguous_prefix(
    tmp_path: Path,
):
    trace_path = _write_trace(
        tmp_path / "tracking_trace.jsonl",
        trajectory_suite_sha256="a" * 64,
    )
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["sample_id"] = 2
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="sample_id values must form a contiguous prefix starting at 0",
    ):
        load_rl_control_trace(trace_path)


def test_load_trajectory_suite_requires_strict_10hz_contiguous_grid(
    tmp_path: Path,
):
    suite_path = _write_json(
        tmp_path / "trajectory_suite.json",
        {
            "suite_id": "suite-001",
            "sample_period_s": 0.1,
            "sample_ids": [0, 1, 2],
        },
    )

    suite, _ = rl_sim_module.load_trajectory_suite_snapshot(suite_path)

    assert suite == {
        "suite_id": "suite-001",
        "sample_period_s": 0.1,
        "sample_ids": (0, 1, 2),
    }


@pytest.mark.parametrize(
    ("sample_period_s", "sample_ids", "message"),
    [
        (0.2, [0, 1], "sample_period_s must be 0.1"),
        (0.1, [1, 2], "sample_ids must start at 0 and be contiguous"),
        (0.1, [0, 2], "sample_ids must start at 0 and be contiguous"),
    ],
)
def test_load_trajectory_suite_rejects_noncanonical_sampling_grid(
    tmp_path: Path,
    sample_period_s: float,
    sample_ids: list[int],
    message: str,
):
    suite_path = _write_json(
        tmp_path / "trajectory_suite.json",
        {
            "suite_id": "suite-001",
            "sample_period_s": sample_period_s,
            "sample_ids": sample_ids,
        },
    )

    with pytest.raises(ExperimentRunValidationError, match=message):
        rl_sim_module.load_trajectory_suite_snapshot(suite_path)


def test_record_rl_sim_experiment_run_publishes_strict_evaluation_parent(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    trajectory_suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0, 1]},
    )
    suite_sha256 = hashlib.sha256(trajectory_suite.read_bytes()).hexdigest()
    trace_path = _write_trace(
        tmp_path / "simulation_trace.jsonl",
        trajectory_suite_sha256=suite_sha256,
    )
    onnx_path = tmp_path / "trajectory_controller.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    training_status = _write_json(
        tmp_path / "training_status.json",
        {"run_id": "scale_v3_deadzone_reward_03_p003"},
    )

    snapshot = record_rl_sim_experiment_run(
        RlSimExperimentRunRequest(
            experiment_run_root=evidence_root,
            machine_profile_path=machine_profile,
            trajectory_suite_path=trajectory_suite,
            trajectory_controller_onnx_path=onnx_path,
            trace_path=trace_path,
            config_paths={"training_status": training_status},
            policy_id="onnx_rl:scale_v3_deadzone_reward_03_p003",
            evaluation_scope="held_out_experiment",
            task_variant="dig_transport_dump",
            operator_id="zhaoshuai",
            material_id="soil_default",
            run_id="rl_sim_suite_001",
        )
    )

    assert snapshot.run_id == "rl_sim_suite_001"
    assert snapshot.state == "success"
    assert snapshot.start["run_kind"] == "evaluation"
    assert snapshot.start["policy_ids"] == {
        "trajectory_controller": "onnx_rl:scale_v3_deadzone_reward_03_p003"
    }
    assert snapshot.final["metrics"] == {
        "evaluation_scope": "held_out_experiment",
        "machine_profile_sha256": snapshot.start["machine_profile"]["sha256"],
            "action_order": ("boom", "stick", "bucket", "swing"),
            "trace_semantics": "commanded_normalized_action",
        "trajectory_controller_onnx_sha256": next(
            artifact["sha256"]
            for artifact in snapshot.artifacts
            if artifact["role"] == "rl_onnx_model"
        ),
        "trajectory_suite_sha256": next(
            artifact["sha256"]
            for artifact in snapshot.artifacts
            if artifact["role"] == "trajectory_suite"
        ),
        "trace_sha256": next(
            artifact["sha256"]
            for artifact in snapshot.artifacts
            if artifact["role"] == "trajectory_trace"
        ),
    }
    assert [artifact["role"] for artifact in snapshot.artifacts] == [
        "rl_onnx_model",
        "trajectory_suite",
        "trajectory_trace",
    ]
    assert snapshot.manifest is not None


def test_record_rl_sim_experiment_run_rejects_suite_changed_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    trajectory_suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0, 1]},
    )
    trace_path = _write_trace(
        tmp_path / "simulation_trace.jsonl",
        trajectory_suite_sha256=hashlib.sha256(trajectory_suite.read_bytes()).hexdigest(),
    )
    onnx_path = tmp_path / "trajectory_controller.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    original_register = ExperimentRun.register_artifact

    def register_after_mutation(self, artifact_id, source_path, *, role, metadata=None):
        if role == "trajectory_suite":
            _write_json(
                trajectory_suite,
                {"suite_id": "suite-mutated", "sample_period_s": 0.1, "sample_ids": [0, 1]},
            )
        return original_register(
            self,
            artifact_id,
            source_path,
            role=role,
            metadata=metadata,
        )

    monkeypatch.setattr(ExperimentRun, "register_artifact", register_after_mutation)

    with pytest.raises(
        ExperimentRunValidationError,
        match="trajectory suite changed while recording evidence",
    ):
        record_rl_sim_experiment_run(
            RlSimExperimentRunRequest(
                experiment_run_root=tmp_path / "evidence",
                machine_profile_path=machine_profile,
                trajectory_suite_path=trajectory_suite,
                trajectory_controller_onnx_path=onnx_path,
                trace_path=trace_path,
                policy_id="onnx_rl:scale-v3",
                evaluation_scope="held_out_experiment",
                task_variant="dig_transport_dump",
                operator_id="zhaoshuai",
                material_id="soil_default",
            )
        )


def test_load_rl_control_trace_rejects_mixed_suite_bindings(tmp_path: Path):
    trace_path = _write_trace(
        tmp_path / "simulation_trace.jsonl",
        trajectory_suite_sha256="a" * 64,
    )
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["trajectory_suite_sha256"] = "b" * 64
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="must bind exactly one trajectory_suite_sha256",
    ):
        load_rl_control_trace(trace_path)


def test_load_rl_control_trace_rejects_unbound_v1_trace(tmp_path: Path):
    trace_path = _write_trace(
        tmp_path / "legacy_trace.jsonl",
        trajectory_suite_sha256="a" * 64,
    )
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        record["schema_version"] = "excavator_rl_control_trace.v1"
        record.pop("trajectory_suite_sha256")
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="trajectory_suite_sha256",
    ):
        load_rl_control_trace(trace_path)


def test_load_trajectory_suite_snapshot_parses_and_hashes_one_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    suite_path = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-a", "sample_period_s": 0.1, "sample_ids": [0, 1]},
    )
    first_payload = suite_path.read_bytes()
    second_payload = (
        json.dumps(
            {"suite_id": "suite-b", "sample_period_s": 0.1, "sample_ids": [0, 1]},
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    original_read_bytes = Path.read_bytes
    suite_read_count = 0

    def alternating_read_bytes(path: Path) -> bytes:
        nonlocal suite_read_count
        if path == suite_path:
            suite_read_count += 1
            return first_payload if suite_read_count == 1 else second_payload
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", alternating_read_bytes)

    suite, suite_sha256 = rl_sim_module.load_trajectory_suite_snapshot(suite_path)

    assert suite == {
        "suite_id": "suite-a",
        "sample_period_s": 0.1,
        "sample_ids": (0, 1),
    }
    assert suite_sha256 == hashlib.sha256(first_payload).hexdigest()
    assert suite_read_count == 1


def test_record_rl_sim_experiment_run_rejects_trace_bound_to_other_suite(
    tmp_path: Path,
):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    trajectory_suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0, 1]},
    )
    trace_path = _write_trace(
        tmp_path / "simulation_trace.jsonl",
        trajectory_suite_sha256="0" * 64,
    )
    onnx_path = tmp_path / "trajectory_controller.onnx"
    onnx_path.write_bytes(b"fake-onnx")

    with pytest.raises(
        ExperimentRunValidationError,
        match="trajectory_suite_sha256 does not match",
    ):
        record_rl_sim_experiment_run(
            RlSimExperimentRunRequest(
                experiment_run_root=tmp_path / "evidence",
                machine_profile_path=machine_profile,
                trajectory_suite_path=trajectory_suite,
                trajectory_controller_onnx_path=onnx_path,
                trace_path=trace_path,
                policy_id="onnx_rl:scale-v3",
                evaluation_scope="held_out_experiment",
                task_variant="dig_transport_dump",
                operator_id="zhaoshuai",
                material_id="soil_default",
            )
        )


def test_record_rl_sim_experiment_run_uses_one_suite_snapshot_for_hash_and_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    suite_a = {"suite_id": "suite-a", "sample_period_s": 0.1, "sample_ids": [0, 1]}
    suite_b = {"suite_id": "suite-b", "sample_period_s": 0.1, "sample_ids": [0, 1]}
    trajectory_suite = _write_json(tmp_path / "trajectory_suite.json", suite_a)
    suite_b_sha256 = hashlib.sha256(
        (json.dumps(suite_b, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    trace_path = _write_trace(
        tmp_path / "simulation_trace.jsonl",
        trajectory_suite_sha256=suite_b_sha256,
    )
    onnx_path = tmp_path / "trajectory_controller.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    original_snapshot = rl_sim_module.load_trajectory_suite_snapshot
    snapshot_called = False

    def snapshot_mutated_suite(path):
        nonlocal snapshot_called
        snapshot_called = True
        _write_json(trajectory_suite, suite_b)
        result = original_snapshot(path)
        _write_json(trajectory_suite, suite_a)
        return result

    monkeypatch.setattr(
        rl_sim_module,
        "load_trajectory_suite_snapshot",
        snapshot_mutated_suite,
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="trajectory suite changed while recording evidence",
    ):
        record_rl_sim_experiment_run(
            RlSimExperimentRunRequest(
                experiment_run_root=tmp_path / "evidence",
                machine_profile_path=machine_profile,
                trajectory_suite_path=trajectory_suite,
                trajectory_controller_onnx_path=onnx_path,
                trace_path=trace_path,
                policy_id="onnx_rl:scale-v3",
                evaluation_scope="held_out_experiment",
                task_variant="dig_transport_dump",
                operator_id="zhaoshuai",
                material_id="soil_default",
            )
        )
    assert snapshot_called


def test_rl_sim_request_rejects_non_onnx_rl_policy_identity(tmp_path: Path):
    with pytest.raises(
        ExperimentRunValidationError,
        match="policy_id must start with onnx_rl:",
    ):
        RlSimExperimentRunRequest(
            experiment_run_root=tmp_path / "evidence",
            machine_profile_path=tmp_path / "machine_profile.json",
            trajectory_suite_path=tmp_path / "trajectory_suite.json",
            trajectory_controller_onnx_path=tmp_path / "controller.onnx",
            trace_path=tmp_path / "trace.jsonl",
            policy_id="cartesian_p:not-an-rl-policy",
            evaluation_scope="held_out_experiment",
            task_variant="dig_transport_dump",
            operator_id="zhaoshuai",
            material_id="soil_default",
        )


def test_rl_sim_request_rejects_unknown_evaluation_scope(tmp_path: Path):
    with pytest.raises(
        ExperimentRunValidationError,
        match="evaluation_scope must be training_internal or held_out_experiment",
    ):
        RlSimExperimentRunRequest(
            experiment_run_root=tmp_path / "evidence",
            machine_profile_path=tmp_path / "machine_profile.json",
            trajectory_suite_path=tmp_path / "trajectory_suite.json",
            trajectory_controller_onnx_path=tmp_path / "controller.onnx",
            trace_path=tmp_path / "trace.jsonl",
            policy_id="onnx_rl:test-policy",
            evaluation_scope="informal_demo",
            task_variant="dig_transport_dump",
            operator_id="zhaoshuai",
            material_id="soil_default",
        )

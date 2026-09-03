from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import excavator_il.rl_real_control_trace as rl_real_module
from excavator_il.experiment_run import ExperimentRun, ExperimentRunValidationError
from excavator_il.rl_real_control_trace import (
    RlRealExperimentRunRequest,
    export_rl_real_control_trace,
    record_rl_real_experiment_run,
)
from excavator_il.rl_sim_experiment_run import load_rl_control_trace


def _write_jsonl(
    path: Path,
    records: list[dict[str, object]],
    *,
    append_terminal: bool = True,
) -> Path:
    completed_records = list(records)
    if (
        append_terminal
        and completed_records
        and not any(record.get("record_type") == "terminal" for record in completed_records)
        and completed_records[-1].get("schema_version")
        == "orin_edge_control_audit.v1"
    ):
        last = completed_records[-1]
        first_stamp = float(completed_records[0].get("runtime_monotonic_s", 0.0))
        terminal_stamp = float(last.get("runtime_monotonic_s", first_stamp)) + 0.1
        completed_records.append(
            _audit_terminal(
                terminal_stamp,
                terminal_stamp - first_stamp,
                trace_run_id=str(last["trace_run_id"]),
            )
        )
    policy_counts: dict[str, int] = {}
    normalized_records: list[dict[str, object]] = []
    for record in completed_records:
        normalized = dict(record)
        trace_run_id = normalized.get("trace_run_id")
        if (
            normalized.get("record_type") == "policy_sample"
            and isinstance(trace_run_id, str)
        ):
            policy_counts[trace_run_id] = policy_counts.get(trace_run_id, 0) + 1
        if (
            normalized.get("record_type") == "terminal"
            and isinstance(trace_run_id, str)
        ):
            count = policy_counts.get(trace_run_id, 0)
            normalized.setdefault("expected_policy_sample_count", count)
            normalized.setdefault("accepted_policy_sample_count", count)
            normalized.setdefault("dropped_policy_sample_count", 0)
        normalized_records.append(normalized)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in normalized_records
        ),
        encoding="utf-8",
    )
    return path


def _audit_record(
    sample_id: int,
    policy_action_seq: int,
    runtime_monotonic_s: float,
    action: list[float],
    *,
    trace_run_id: str = "follow-run-001",
    bucket_tip_ros_m: list[float] | None = None,
    reference_waypoint_ros_m: list[float] | None = None,
    waypoint_index: int = 0,
    waypoint_distance_m: float = 0.2,
    episode_progress: float = 0.5,
) -> dict[str, object]:
    return {
        "schema_version": "orin_edge_control_audit.v1",
        "record_type": "policy_sample",
        "mode": "control",
        "status": "active",
        "trajectory_controller_backend": "onnx_rl",
        "trace_semantics": "commanded_normalized_action",
        "sample_id": sample_id,
        "policy_action_seq": policy_action_seq,
        "runtime_monotonic_s": runtime_monotonic_s,
        "action_order": ["boom", "stick", "bucket", "swing"],
        "commanded_normalized_action": action,
        "trace_run_id": trace_run_id,
        "bucket_tip_ros_m": bucket_tip_ros_m or [0.8, 0.0, 0.0],
        "reference_waypoint_ros_m": reference_waypoint_ros_m or [1.0, 0.0, 0.0],
        "waypoint_index": waypoint_index,
        "waypoint_distance_m": waypoint_distance_m,
        "episode_progress": episode_progress,
        "result": "active",
    }


def _audit_terminal(
    runtime_monotonic_s: float,
    elapsed_s: float,
    *,
    trace_run_id: str = "follow-run-001",
    result: str = "completed",
) -> dict[str, object]:
    return {
        "schema_version": "orin_edge_control_audit.v1",
        "record_type": "terminal",
        "mode": "control",
        "status": "terminal",
        "trajectory_controller_backend": "onnx_rl",
        "trace_semantics": "commanded_normalized_action",
        "trace_run_id": trace_run_id,
        "runtime_monotonic_s": runtime_monotonic_s,
        "elapsed_s": elapsed_s,
        "result": result,
    }


def test_export_rl_real_control_trace_preserves_explicit_sample_identity(tmp_path: Path):
    audit_path = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [
            _audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4]),
            _audit_record(1, 101, 25.1, [0.0, 0.0, 0.0, 0.0]),
            _audit_terminal(25.2, 0.2),
        ],
    )
    output_path = tmp_path / "real_trace.jsonl"

    result = export_rl_real_control_trace(
        audit_path,
        output_path,
        trace_run_id="follow-run-001",
        trajectory_suite_sha256="a" * 64,
    )

    assert result.output_path == output_path
    assert result.sample_count == 2
    assert result.first_sample_id == 0
    assert result.last_sample_id == 1
    assert result.first_policy_action_seq == 100
    assert result.last_policy_action_seq == 101
    samples = load_rl_control_trace(output_path)
    assert [sample.sample_id for sample in samples] == [0, 1]
    assert [sample.trace_semantics for sample in samples] == [
        "commanded_normalized_action",
        "commanded_normalized_action",
    ]
    assert samples[0].action == (0.1, -0.2, 0.3, -0.4)
    assert samples[0].bucket_tip_ros_m == (0.8, 0.0, 0.0)
    assert samples[0].result == "ACTIVE"
    assert {sample.trajectory_suite_sha256 for sample in samples} == {"a" * 64}


def test_export_rl_real_control_trace_rejects_unrecognized_unselected_record(
    tmp_path: Path,
):
    audit_path = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [
            {
                "schema_version": "unknown.v1",
                "record_type": "policy_sample",
                "trace_run_id": "follow-run-other",
            },
            _audit_record(0, 100, 25.0, [0.0, 0.0, 0.0, 0.0]),
            _audit_terminal(25.1, 0.1),
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="schema_version must be orin_edge_control_audit.v1",
    ):
        export_rl_real_control_trace(
            audit_path,
            tmp_path / "real_trace.jsonl",
            trace_run_id="follow-run-001",
            trajectory_suite_sha256="a" * 64,
        )


def test_record_rl_real_experiment_run_binds_suite_model_audit_and_trace(tmp_path: Path):
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_text(
        json.dumps({"action_order": ["boom", "stick", "bucket", "swing"]}) + "\n",
        encoding="utf-8",
    )
    suite = tmp_path / "trajectory_suite.json"
    suite.write_text(
        json.dumps(
            {
                "suite_id": "suite-001",
                "sample_period_s": 0.1,
                "sample_ids": [0, 1, 2],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "controller.onnx"
    model.write_bytes(b"onnx-controller")
    audit = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [
            _audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4]),
            _audit_record(1, 101, 25.1, [0.0, 0.0, 0.0, 0.0]),
        ],
    )

    snapshot = record_rl_real_experiment_run(
        RlRealExperimentRunRequest(
            experiment_run_root=tmp_path / "evidence",
            machine_profile_path=machine_profile,
            trajectory_suite_path=suite,
            trajectory_controller_onnx_path=model,
            control_audit_path=audit,
            trace_output_path=tmp_path / "real_trace.jsonl",
            trace_run_id="follow-run-001",
            trajectory_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
            policy_id="onnx_rl:scale-v3",
            evaluation_scope="held_out_experiment",
            task_variant="dig_transport_dump",
            operator_id="zhaoshuai",
            material_id="soil_default",
            run_id="rl_real_suite_001",
        )
    )

    assert snapshot.run_id == "rl_real_suite_001"
    assert snapshot.state == "success"
    assert snapshot.start["run_kind"] == "hybrid_live"
    assert [artifact["role"] for artifact in snapshot.artifacts] == [
        "rl_onnx_model",
        "trajectory_suite",
        "rl_control_audit",
        "trajectory_trace",
    ]
    assert snapshot.final["metrics"]["trace_semantics"] == (
        "commanded_normalized_action"
    )
    assert snapshot.final["metrics"]["sample_count"] == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode", "shadow", "mode must be control"),
        (
            "trajectory_controller_backend",
            "cartesian_p",
            "trajectory_controller_backend must be onnx_rl",
        ),
        (
            "trace_semantics",
            "normalized_action",
            "trace_semantics must be commanded_normalized_action",
        ),
        ("policy_action_seq", None, "policy_action_seq must be a non-negative int"),
        (
            "action_order",
            ["swing", "boom", "stick", "bucket"],
            "action_order must be",
        ),
    ],
)
def test_export_rl_real_control_trace_rejects_non_rl_or_ambiguous_records(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
):
    record = _audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4])
    if value is None:
        record.pop(field)
    else:
        record[field] = value
    audit_path = _write_jsonl(tmp_path / "edge_control.jsonl", [record])

    with pytest.raises(ExperimentRunValidationError, match=message):
        export_rl_real_control_trace(
            audit_path,
            tmp_path / "trace.jsonl",
            trace_run_id="follow-run-001",
            trajectory_suite_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        (
            "sample_id",
            [0, 0],
            "sample_id values must form a contiguous prefix starting at 0",
        ),
        (
            "policy_action_seq",
            [100, 99],
            "policy_action_seq must be unique and strictly increasing",
        ),
    ],
)
def test_export_rl_real_control_trace_rejects_reused_or_regressing_identity(
    tmp_path: Path,
    field: str,
    values: list[int],
    message: str,
):
    records = [
        _audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4]),
        _audit_record(1, 101, 25.1, [0.0, 0.0, 0.0, 0.0]),
    ]
    records[0][field], records[1][field] = values
    audit_path = _write_jsonl(tmp_path / "edge_control.jsonl", records)

    with pytest.raises(ExperimentRunValidationError, match=message):
        export_rl_real_control_trace(
            audit_path,
            tmp_path / "trace.jsonl",
            trace_run_id="follow-run-001",
            trajectory_suite_sha256="a" * 64,
        )


def test_record_rl_real_experiment_run_rejects_samples_outside_suite(tmp_path: Path):
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_text(
        json.dumps({"action_order": ["boom", "stick", "bucket", "swing"]}) + "\n",
        encoding="utf-8",
    )
    suite = tmp_path / "trajectory_suite.json"
    suite.write_text(
        json.dumps(
            {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]}
        )
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "controller.onnx"
    model.write_bytes(b"onnx-controller")
    audit = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [
            _audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4]),
            _audit_record(1, 101, 25.1, [0.0, 0.0, 0.0, 0.0]),
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="sample_id values outside the trajectory suite",
    ):
        record_rl_real_experiment_run(
            RlRealExperimentRunRequest(
                experiment_run_root=tmp_path / "evidence",
                machine_profile_path=machine_profile,
                trajectory_suite_path=suite,
                trajectory_controller_onnx_path=model,
                control_audit_path=audit,
                trace_output_path=tmp_path / "real_trace.jsonl",
                trace_run_id="follow-run-001",
                trajectory_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
                policy_id="onnx_rl:scale-v3",
                evaluation_scope="held_out_experiment",
                task_variant="dig_transport_dump",
                operator_id="zhaoshuai",
                material_id="soil_default",
            )
        )


def test_record_rl_real_experiment_run_rejects_wrong_explicit_suite_hash(
    tmp_path: Path,
):
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_text(
        json.dumps({"action_order": ["boom", "stick", "bucket", "swing"]}) + "\n",
        encoding="utf-8",
    )
    suite = tmp_path / "trajectory_suite.json"
    suite.write_text(
        json.dumps(
            {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]}
        )
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "controller.onnx"
    model.write_bytes(b"onnx-controller")
    audit = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [_audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4])],
    )
    trace_output = tmp_path / "real_trace.jsonl"

    with pytest.raises(
        ExperimentRunValidationError,
        match="explicit trajectory_suite_sha256 does not match",
    ):
        record_rl_real_experiment_run(
            RlRealExperimentRunRequest(
                experiment_run_root=tmp_path / "evidence",
                machine_profile_path=machine_profile,
                trajectory_suite_path=suite,
                trajectory_controller_onnx_path=model,
                control_audit_path=audit,
                trace_output_path=trace_output,
                trace_run_id="follow-run-001",
                trajectory_suite_sha256="0" * 64,
                policy_id="onnx_rl:scale-v3",
                evaluation_scope="held_out_experiment",
                task_variant="dig_transport_dump",
                operator_id="zhaoshuai",
                material_id="soil_default",
            )
        )

    assert not trace_output.exists()


def test_record_rl_real_experiment_run_uses_one_suite_snapshot_for_hash_and_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_text(
        json.dumps({"action_order": ["boom", "stick", "bucket", "swing"]}) + "\n",
        encoding="utf-8",
    )
    suite_a = {"suite_id": "suite-a", "sample_period_s": 0.1, "sample_ids": [0]}
    suite_b = {"suite_id": "suite-b", "sample_period_s": 0.1, "sample_ids": [0]}
    suite = tmp_path / "trajectory_suite.json"
    suite.write_text(json.dumps(suite_a, sort_keys=True) + "\n", encoding="utf-8")
    suite_b_sha256 = hashlib.sha256(
        (json.dumps(suite_b, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    model = tmp_path / "controller.onnx"
    model.write_bytes(b"onnx-controller")
    audit = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [_audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4])],
    )
    original_snapshot = rl_real_module.load_trajectory_suite_snapshot
    snapshot_called = False

    def snapshot_mutated_suite(path):
        nonlocal snapshot_called
        snapshot_called = True
        suite.write_text(json.dumps(suite_b, sort_keys=True) + "\n", encoding="utf-8")
        result = original_snapshot(path)
        suite.write_text(json.dumps(suite_a, sort_keys=True) + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        rl_real_module,
        "load_trajectory_suite_snapshot",
        snapshot_mutated_suite,
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="trajectory suite changed while recording evidence",
    ):
        record_rl_real_experiment_run(
            RlRealExperimentRunRequest(
                experiment_run_root=tmp_path / "evidence",
                machine_profile_path=machine_profile,
                trajectory_suite_path=suite,
                trajectory_controller_onnx_path=model,
                control_audit_path=audit,
                trace_output_path=tmp_path / "real_trace.jsonl",
                trace_run_id="follow-run-001",
                trajectory_suite_sha256=suite_b_sha256,
                policy_id="onnx_rl:scale-v3",
                evaluation_scope="held_out_experiment",
                task_variant="dig_transport_dump",
                operator_id="zhaoshuai",
                material_id="soil_default",
            )
        )
    assert snapshot_called


def test_record_rl_real_experiment_run_rejects_audit_changed_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_text(
        json.dumps({"action_order": ["boom", "stick", "bucket", "swing"]}) + "\n",
        encoding="utf-8",
    )
    suite = tmp_path / "trajectory_suite.json"
    suite.write_text(
        json.dumps(
            {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]}
        )
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "controller.onnx"
    model.write_bytes(b"onnx-controller")
    audit = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [_audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4])],
    )
    original_register = ExperimentRun.register_artifact

    def register_after_mutation(self, artifact_id, source_path, *, role, metadata=None):
        if role == "rl_control_audit":
            audit.write_text(
                json.dumps(
                    _audit_record(0, 100, 25.0, [0.9, 0.9, 0.9, 0.9]),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
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
        match="control audit changed while recording evidence",
    ):
        record_rl_real_experiment_run(
            RlRealExperimentRunRequest(
                experiment_run_root=tmp_path / "evidence",
                machine_profile_path=machine_profile,
                trajectory_suite_path=suite,
                trajectory_controller_onnx_path=model,
                control_audit_path=audit,
                trace_output_path=tmp_path / "real_trace.jsonl",
                trace_run_id="follow-run-001",
                trajectory_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
                policy_id="onnx_rl:scale-v3",
                evaluation_scope="held_out_experiment",
                task_variant="dig_transport_dump",
                operator_id="zhaoshuai",
                material_id="soil_default",
            )
        )


def test_export_rl_real_control_trace_never_overwrites_source_audit(tmp_path: Path):
    audit_path = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [_audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4])],
    )
    original = audit_path.read_bytes()

    with pytest.raises(
        ExperimentRunValidationError,
        match="output must not overwrite the source audit",
    ):
        export_rl_real_control_trace(
            audit_path,
            audit_path,
            trace_run_id="follow-run-001",
            trajectory_suite_sha256="a" * 64,
        )

    assert audit_path.read_bytes() == original


def test_export_rl_real_control_trace_never_overwrites_existing_output(tmp_path: Path):
    audit_path = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [_audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4])],
    )
    output_path = tmp_path / "real_trace.jsonl"
    output_path.write_text("keep-existing\n", encoding="utf-8")

    with pytest.raises(
        ExperimentRunValidationError,
        match="output already exists",
    ):
        export_rl_real_control_trace(
            audit_path,
            output_path,
            trace_run_id="follow-run-001",
            trajectory_suite_sha256="a" * 64,
        )

    assert output_path.read_text(encoding="utf-8") == "keep-existing\n"


def test_export_rl_real_control_trace_selects_one_explicit_append_log_segment(
    tmp_path: Path,
):
    target_terminal = _audit_terminal(
        25.2,
        0.2,
        trace_run_id="follow-run-target",
    )
    audit_path = _write_jsonl(
        tmp_path / "shared_edge_control.jsonl",
        [
            _audit_record(
                0,
                50,
                24.0,
                [0.9, 0.9, 0.9, 0.9],
                trace_run_id="follow-run-other",
            ),
            _audit_terminal(
                24.1,
                0.1,
                trace_run_id="follow-run-other",
            ),
            _audit_record(
                0,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                trace_run_id="follow-run-target",
            ),
            _audit_record(
                1,
                101,
                25.1,
                [0.0, 0.0, 0.0, 0.0],
                trace_run_id="follow-run-target",
            ),
            target_terminal,
        ],
    )

    result = export_rl_real_control_trace(
        audit_path,
        tmp_path / "target_trace.jsonl",
        trace_run_id="follow-run-target",
        trajectory_suite_sha256="a" * 64,
    )

    assert result.trace_run_id == "follow-run-target"
    assert result.sample_count == 2
    assert [
        sample.action
        for sample in load_rl_control_trace(result.output_path)
    ] == [
        (0.1, -0.2, 0.3, -0.4),
        (0.0, 0.0, 0.0, 0.0),
    ]


def test_export_rl_real_control_trace_rejects_non_active_target_record(
    tmp_path: Path,
):
    rejected = _audit_record(
        0,
        100,
        25.0,
        [0.1, -0.2, 0.3, -0.4],
        trace_run_id="follow-run-target",
    )
    rejected["status"] = "command_failed"
    audit_path = _write_jsonl(tmp_path / "shared_edge_control.jsonl", [rejected])

    with pytest.raises(
        ExperimentRunValidationError,
        match="status must be active",
    ):
        export_rl_real_control_trace(
            audit_path,
            tmp_path / "target_trace.jsonl",
            trace_run_id="follow-run-target",
            trajectory_suite_sha256="a" * 64,
        )


def test_export_rl_real_control_trace_rejects_selected_audit_error(
    tmp_path: Path,
):
    audit_path = _write_jsonl(
        tmp_path / "shared_edge_control.jsonl",
        [
            _audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4]),
            {
                "schema_version": "orin_edge_control_audit.v1",
                "record_type": "audit_error",
                "trace_run_id": "follow-run-001",
                "error": "decision mismatch",
            },
            _audit_terminal(25.2, 0.2),
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="selected Orin audit segment contains audit_error",
    ):
        export_rl_real_control_trace(
            audit_path,
            tmp_path / "target_trace.jsonl",
            trace_run_id="follow-run-001",
            trajectory_suite_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("terminal_counts", "terminal_fields", "error"),
    [
        ((2, 1, 1), {}, "dropped policy samples"),
        ((2, 2, 0), {}, "does not match persisted policy samples"),
        ((1, 1, 0), {"sample_id": 0}, "must not contain policy sample field"),
    ],
)
def test_export_rejects_terminal_policy_sample_count_mismatch(
    tmp_path: Path, terminal_counts: tuple[int, int, int],
    terminal_fields: dict[str, object], error: str,
):
    expected, accepted, dropped = terminal_counts
    terminal = _audit_terminal(25.1, 0.1)
    terminal.update(
        {
            "expected_policy_sample_count": expected,
            "accepted_policy_sample_count": accepted,
            "dropped_policy_sample_count": dropped,
            **terminal_fields,
        }
    )
    audit_path = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [
            _audit_record(0, 100, 25.0, [0.1, -0.2, 0.3, -0.4]),
            terminal,
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match=error,
    ):
        export_rl_real_control_trace(
            audit_path,
            tmp_path / "target_trace.jsonl",
            trace_run_id="follow-run-001",
            trajectory_suite_sha256="a" * 64,
        )


def test_export_rl_real_control_trace_rejects_discontiguous_reused_segment(
    tmp_path: Path,
):
    audit_path = _write_jsonl(
        tmp_path / "shared_edge_control.jsonl",
        [
            _audit_record(
                0,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                trace_run_id="follow-run-target",
            ),
            _audit_record(
                0,
                200,
                30.0,
                [0.0, 0.0, 0.0, 0.0],
                trace_run_id="follow-run-other",
            ),
            _audit_record(
                1,
                101,
                25.1,
                [0.0, 0.0, 0.0, 0.0],
                trace_run_id="follow-run-target",
            ),
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="trace_run_id must identify one contiguous audit segment",
    ):
        export_rl_real_control_trace(
            audit_path,
            tmp_path / "target_trace.jsonl",
            trace_run_id="follow-run-target",
            trajectory_suite_sha256="a" * 64,
        )

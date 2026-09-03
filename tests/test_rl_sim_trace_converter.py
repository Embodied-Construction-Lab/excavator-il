from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import excavator_il.rl_sim_trace_converter as converter_module
from excavator_il.experiment_run import ExperimentRunValidationError
from excavator_il.rl_sim_trace_converter import export_rl_sim_control_trace
from excavator_il.rl_sim_experiment_run import (
    load_rl_control_trace,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


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
        == "rl_excavator_sim_control_audit.v1"
    ):
        last = completed_records[-1]
        first_stamp = float(completed_records[0].get("runtime_monotonic_s", 0.0))
        terminal_stamp = float(last.get("runtime_monotonic_s", first_stamp)) + 0.1
        completed_records.append(
            _simulation_terminal(
                terminal_stamp,
                terminal_stamp - first_stamp,
                suite_sha256=str(last["trajectory_suite_sha256"]),
                trace_run_id=str(last["trace_run_id"]),
            )
        )
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in completed_records
        ),
        encoding="utf-8",
    )
    return path


def _simulation_record(
    sample_id: int,
    policy_action_seq: int,
    stamp_s: float,
    action: list[float],
    *,
    suite_sha256: str,
    trace_run_id: str = "unity-follow-001",
    bucket_tip_ros_m: list[float] | None = None,
    reference_waypoint_ros_m: list[float] | None = None,
    waypoint_index: int = 0,
    waypoint_distance_m: float = 0.2,
    episode_progress: float = 0.5,
    result: str = "active",
) -> dict[str, object]:
    return {
        "schema_version": "rl_excavator_sim_control_audit.v1",
        "record_type": "policy_sample",
        "mode": "control",
        "status": "active",
        "trajectory_controller_backend": "onnx_rl",
        "trajectory_suite_sha256": suite_sha256,
        "trace_semantics": "commanded_normalized_action",
        "trace_run_id": trace_run_id,
        "sample_id": sample_id,
        "policy_action_seq": policy_action_seq,
        "runtime_monotonic_s": stamp_s,
        "action_order": ["boom", "stick", "bucket", "swing"],
        "commanded_normalized_action": action,
        "bucket_tip_ros_m": bucket_tip_ros_m or [0.8, 0.0, 0.0],
        "reference_waypoint_ros_m": reference_waypoint_ros_m or [1.0, 0.0, 0.0],
        "waypoint_index": waypoint_index,
        "waypoint_distance_m": waypoint_distance_m,
        "episode_progress": episode_progress,
        "result": result,
    }


def _simulation_terminal(
    stamp_s: float,
    elapsed_s: float,
    *,
    suite_sha256: str,
    trace_run_id: str = "unity-follow-001",
    result: str = "completed",
    consumed_sample_count: int = 1,
    suite_sample_count: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": "rl_excavator_sim_control_audit.v1",
        "record_type": "terminal",
        "mode": "control",
        "status": "terminal",
        "trajectory_controller_backend": "onnx_rl",
        "trajectory_suite_sha256": suite_sha256,
        "trace_semantics": "commanded_normalized_action",
        "trace_run_id": trace_run_id,
        "runtime_monotonic_s": stamp_s,
        "elapsed_s": elapsed_s,
        "consumed_sample_count": consumed_sample_count,
        "suite_sample_count": suite_sample_count,
        "result": result,
    }


def test_export_rl_sim_control_trace_preserves_explicit_suite_sample_identity(
    tmp_path: Path,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0, 1, 2]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    audit = _write_jsonl(
        tmp_path / "unity_control_audit.jsonl",
        [
            _simulation_record(
                0,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                suite_sha256=suite_sha256,
            ),
            _simulation_record(
                1,
                101,
                25.1,
                [0.0, 0.0, 0.0, 0.0],
                suite_sha256=suite_sha256,
            ),
            _simulation_terminal(
                25.2,
                0.2,
                suite_sha256=suite_sha256,
                consumed_sample_count=2,
                suite_sample_count=3,
            ),
        ],
    )
    output = tmp_path / "simulation_trace.jsonl"

    result = export_rl_sim_control_trace(
        audit,
        output,
        trajectory_suite_path=suite,
        trace_run_id="unity-follow-001",
    )

    assert result.output_path == output
    assert result.trace_run_id == "unity-follow-001"
    assert result.sample_count == 2
    assert result.first_sample_id == 0
    assert result.last_sample_id == 1
    assert result.first_policy_action_seq == 100
    assert result.last_policy_action_seq == 101
    assert result.trajectory_suite_sha256 == suite_sha256
    samples = load_rl_control_trace(output)
    assert [sample.sample_id for sample in samples] == [0, 1]
    assert samples[0].action == (0.1, -0.2, 0.3, -0.4)
    assert samples[0].bucket_tip_ros_m == (0.8, 0.0, 0.0)
    assert samples[0].reference_waypoint_ros_m == (1.0, 0.0, 0.0)
    assert samples[0].waypoint_index == 0
    assert samples[0].waypoint_distance_m == 0.2
    assert samples[0].episode_progress == 0.5
    assert samples[0].result == "ACTIVE"
    assert {sample.trajectory_suite_sha256 for sample in samples} == {suite_sha256}


def test_export_rl_sim_control_trace_rejects_unrecognized_unselected_record(
    tmp_path: Path,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    audit = _write_jsonl(
        tmp_path / "unity_control_audit.jsonl",
        [
            {
                "schema_version": "unknown.v1",
                "record_type": "policy_sample",
                "trace_run_id": "unity-follow-other",
            },
            _simulation_record(
                0,
                100,
                25.0,
                [0.0, 0.0, 0.0, 0.0],
                suite_sha256=suite_sha256,
            ),
            _simulation_terminal(25.1, 0.1, suite_sha256=suite_sha256),
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="schema_version must be rl_excavator_sim_control_audit.v1",
    ):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )


def test_export_rl_sim_control_trace_rejects_legacy_fixed_update_csv(
    tmp_path: Path,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]},
    )
    legacy_csv = tmp_path / "legacy_open_loop.csv"
    legacy_csv.write_text(
        "# rl_excavator_open_loop_velocity_export\n"
        "sample_index,timestamp_s,boom_action_cmd,stick_action_cmd,"
        "bucket_action_cmd,swing_action_cmd\n"
        "0,0.0,0.1,-0.2,0.3,-0.4\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match=(
            "legacy OpenLoopVelocityRecorder CSV cannot be converted: "
            "explicit trace_run_id, policy_action_seq and frozen trajectory-suite "
            "binding are missing"
        ),
    ):
        export_rl_sim_control_trace(
            legacy_csv,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )


def test_export_rl_sim_control_trace_rejects_discontiguous_reused_segment(
    tmp_path: Path,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0, 1]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    audit = _write_jsonl(
        tmp_path / "shared_unity_audit.jsonl",
        [
            _simulation_record(
                0,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                suite_sha256=suite_sha256,
                trace_run_id="unity-follow-target",
            ),
            _simulation_record(
                0,
                200,
                30.0,
                [0.9, 0.9, 0.9, 0.9],
                suite_sha256=suite_sha256,
                trace_run_id="unity-follow-other",
            ),
            _simulation_record(
                1,
                101,
                25.1,
                [0.0, 0.0, 0.0, 0.0],
                suite_sha256=suite_sha256,
                trace_run_id="unity-follow-target",
            ),
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="trace_run_id must identify one contiguous simulator audit segment",
    ):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-target",
        )


def test_export_rl_sim_control_trace_never_overwrites_existing_output(
    tmp_path: Path,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    audit = _write_jsonl(
        tmp_path / "unity_control_audit.jsonl",
        [
            _simulation_record(
                0,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                suite_sha256=suite_sha256,
            )
        ],
    )
    output = tmp_path / "simulation_trace.jsonl"
    output.write_text("keep-existing\n", encoding="utf-8")

    with pytest.raises(
        ExperimentRunValidationError,
        match="simulator trajectory trace output already exists",
    ):
        export_rl_sim_control_trace(
            audit,
            output,
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )

    assert output.read_text(encoding="utf-8") == "keep-existing\n"


def test_export_rl_sim_control_trace_requires_explicit_trace_run_id(
    tmp_path: Path,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    audit = _write_jsonl(
        tmp_path / "unity_control_audit.jsonl",
        [
            _simulation_record(
                0,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                suite_sha256=suite_sha256,
            )
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="simulator control audit contains no trace_run_id=unknown-run",
    ):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unknown-run",
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
        (
            "runtime_monotonic_s",
            [25.0, 24.9],
            "runtime_monotonic_s must be strictly increasing",
        ),
    ],
)
def test_export_rl_sim_control_trace_rejects_regressing_identity_or_time(
    tmp_path: Path,
    field: str,
    values: list[int | float],
    message: str,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0, 1]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    records = [
        _simulation_record(
            0,
            100,
            25.0,
            [0.1, -0.2, 0.3, -0.4],
            suite_sha256=suite_sha256,
        ),
        _simulation_record(
            1,
            101,
            25.1,
            [0.0, 0.0, 0.0, 0.0],
            suite_sha256=suite_sha256,
        ),
    ]
    records[0] = {**records[0], field: values[0]}
    records[1] = {**records[1], field: values[1]}
    audit = _write_jsonl(tmp_path / "unity_control_audit.jsonl", records)

    with pytest.raises(ExperimentRunValidationError, match=message):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "trajectory_controller_backend",
            "cartesian_p",
            "trajectory_controller_backend must be onnx_rl",
        ),
        (
            "trace_semantics",
            "raw_policy_action",
            "trace_semantics must be commanded_normalized_action",
        ),
        (
            "action_order",
            ["swing", "boom", "stick", "bucket"],
            "action_order must be",
        ),
        (
            "policy_action_seq",
            None,
            "policy_action_seq must be a non-negative int",
        ),
        (
            "commanded_normalized_action",
            [0.0, 0.0, 0.0],
            "commanded_normalized_action must contain four values",
        ),
        (
            "commanded_normalized_action",
            [0.0, 0.0, 0.0, 1.01],
            "commanded_normalized_action must be within",
        ),
    ],
)
def test_export_rl_sim_control_trace_rejects_ambiguous_or_invalid_records(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    record = _simulation_record(
        0,
        100,
        25.0,
        [0.1, -0.2, 0.3, -0.4],
        suite_sha256=suite_sha256,
    )
    record = (
        {key: item for key, item in record.items() if key != field}
        if value is None
        else {**record, field: value}
    )
    audit = _write_jsonl(tmp_path / "unity_control_audit.jsonl", [record])

    with pytest.raises(ExperimentRunValidationError, match=message):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )


@pytest.mark.parametrize(
    "field",
    [
        "bucket_tip_ros_m",
        "reference_waypoint_ros_m",
        "waypoint_index",
        "waypoint_distance_m",
        "episode_progress",
        "result",
    ],
)
def test_export_rl_sim_control_trace_requires_real_tracking_fields(
    tmp_path: Path,
    field: str,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    record = _simulation_record(
        0,
        100,
        25.0,
        [0.1, -0.2, 0.3, -0.4],
        suite_sha256=suite_sha256,
    )
    record.pop(field)
    audit = _write_jsonl(tmp_path / "unity_control_audit.jsonl", [record])

    with pytest.raises(ExperimentRunValidationError, match=field):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )


def test_export_rl_sim_control_trace_rejects_audit_error_in_selected_segment(
    tmp_path: Path,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    audit = _write_jsonl(
        tmp_path / "unity_control_audit.jsonl",
        [
            _simulation_record(
                0,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                suite_sha256=suite_sha256,
            ),
            {
                "schema_version": "rl_excavator_sim_control_audit.v1",
                "record_type": "audit_error",
                "trace_run_id": "unity-follow-001",
                "error": "decision mismatch",
            },
            _simulation_terminal(25.2, 0.2, suite_sha256=suite_sha256),
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="selected simulator audit segment contains audit_error",
    ):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )


@pytest.mark.parametrize(
    "records_factory",
    [
        lambda suite_sha256: [
            _simulation_record(
                0,
                100,
                25.0,
                [0.0, 0.0, 0.0, 0.0],
                suite_sha256=suite_sha256,
            )
        ],
        lambda suite_sha256: [
            _simulation_record(
                0,
                100,
                25.0,
                [0.0, 0.0, 0.0, 0.0],
                suite_sha256=suite_sha256,
            ),
            _simulation_terminal(25.1, 0.1, suite_sha256=suite_sha256),
            _simulation_terminal(25.2, 0.2, suite_sha256=suite_sha256),
        ],
    ],
)
def test_export_rl_sim_control_trace_requires_one_final_terminal(
    tmp_path: Path,
    records_factory,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    audit = _write_jsonl(
        tmp_path / "unity_control_audit.jsonl",
        records_factory(suite_sha256),
        append_terminal=False,
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="exactly one final terminal record",
    ):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("consumed_sample_count", 2, "consumed_sample_count must equal 1"),
        ("suite_sample_count", 1, "suite_sample_count must equal 2"),
        ("consumed_sample_count", 0, "consumed_sample_count must be a positive int"),
        ("suite_sample_count", True, "suite_sample_count must be a positive int"),
    ],
)
def test_export_rl_sim_control_trace_rejects_false_terminal_sample_counts(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0, 1]},
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    terminal = _simulation_terminal(
        25.1,
        0.1,
        suite_sha256=suite_sha256,
        consumed_sample_count=1,
        suite_sample_count=2,
    )
    terminal[field] = value
    audit = _write_jsonl(
        tmp_path / "unity_control_audit.jsonl",
        [
            _simulation_record(
                0,
                100,
                25.0,
                [0.0, 0.0, 0.0, 0.0],
                suite_sha256=suite_sha256,
            ),
            terminal,
        ],
    )

    with pytest.raises(ExperimentRunValidationError, match=message):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )


def test_export_rl_sim_control_trace_rejects_wrong_suite_binding_and_sample(
    tmp_path: Path,
):
    suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {"suite_id": "suite-001", "sample_period_s": 0.1, "sample_ids": [0]},
    )
    wrong_suite_hash = "0" * 64
    audit = _write_jsonl(
        tmp_path / "unity_control_audit.jsonl",
        [
            _simulation_record(
                0,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                suite_sha256=wrong_suite_hash,
            )
        ],
    )

    with pytest.raises(
        ExperimentRunValidationError,
        match="trajectory_suite_sha256 must be",
    ):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )

    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    audit = _write_jsonl(
        tmp_path / "unity_control_audit_outside.jsonl",
        [
            _simulation_record(
                2,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                suite_sha256=suite_sha256,
            )
        ],
    )
    with pytest.raises(
        ExperimentRunValidationError,
        match="sample_id values outside the trajectory suite",
    ):
        export_rl_sim_control_trace(
            audit,
            tmp_path / "simulation_trace_outside.jsonl",
            trajectory_suite_path=suite,
            trace_run_id="unity-follow-001",
        )


def test_export_rl_sim_control_trace_hashes_and_parses_one_suite_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    suite_a = {"suite_id": "suite-a", "sample_period_s": 0.1, "sample_ids": [0]}
    suite_b = {"suite_id": "suite-b", "sample_period_s": 0.1, "sample_ids": [0]}
    suite = _write_json(tmp_path / "trajectory_suite.json", suite_a)
    original_snapshot = converter_module.load_trajectory_suite_snapshot
    snapshot_called = False

    def snapshot_then_restore(path):
        nonlocal snapshot_called
        snapshot_called = True
        _write_json(suite, suite_b)
        result = original_snapshot(path)
        _write_json(suite, suite_a)
        return result

    monkeypatch.setattr(
        converter_module,
        "load_trajectory_suite_snapshot",
        snapshot_then_restore,
    )
    suite_b_sha256 = hashlib.sha256(
        (json.dumps(suite_b, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    audit = _write_jsonl(
        tmp_path / "unity_control_audit.jsonl",
        [
            _simulation_record(
                0,
                100,
                25.0,
                [0.1, -0.2, 0.3, -0.4],
                suite_sha256=suite_b_sha256,
            )
        ],
    )

    output = tmp_path / "simulation_trace.jsonl"
    result = export_rl_sim_control_trace(
        audit,
        output,
        trajectory_suite_path=suite,
        trace_run_id="unity-follow-001",
    )

    assert snapshot_called
    assert result.trajectory_suite_sha256 == suite_b_sha256
    samples = load_rl_control_trace(output)
    assert [sample.sample_id for sample in samples] == [0]
    assert {sample.trajectory_suite_sha256 for sample in samples} == {
        suite_b_sha256
    }

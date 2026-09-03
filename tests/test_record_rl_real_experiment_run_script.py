import json
import hashlib
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/record_rl_real_experiment_run.py"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return path


def _audit_record(
    sample_id: int,
    policy_action_seq: int,
    runtime_monotonic_s: float,
    action: list[float],
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
        "trace_run_id": "follow-run-001",
        "bucket_tip_ros_m": [0.8 + 0.1 * sample_id, 0.0, 0.0],
        "reference_waypoint_ros_m": [1.0, 0.0, 0.0],
        "waypoint_index": sample_id,
        "waypoint_distance_m": abs(0.2 - 0.1 * sample_id),
        "episode_progress": sample_id / 2.0,
        "result": "active",
    }


def test_script_records_real_parent_run_and_exports_trace(tmp_path: Path):
    machine_profile = _write_json(
        tmp_path / "machine_profile.json",
        {"action_order": ["boom", "stick", "bucket", "swing"]},
    )
    trajectory_suite = _write_json(
        tmp_path / "trajectory_suite.json",
        {
            "suite_id": "suite-001",
            "sample_period_s": 0.1,
            "sample_ids": [0, 1, 2],
        },
    )
    onnx = tmp_path / "controller.onnx"
    onnx.write_bytes(b"onnx-controller")
    audit = _write_jsonl(
        tmp_path / "edge_control.jsonl",
        [
            _audit_record(0, 10, 1.0, [0.1, 0.0, -0.2, 0.3]),
            _audit_record(1, 11, 1.1, [0.0, 0.0, 0.0, 0.0]),
            {
                "schema_version": "orin_edge_control_audit.v1",
                "record_type": "terminal",
                "mode": "control",
                "status": "terminal",
                "trajectory_controller_backend": "onnx_rl",
                "trace_semantics": "commanded_normalized_action",
                "trace_run_id": "follow-run-001",
                    "runtime_monotonic_s": 1.2,
                    "elapsed_s": 0.2,
                    "expected_policy_sample_count": 2,
                    "accepted_policy_sample_count": 2,
                    "dropped_policy_sample_count": 0,
                    "result": "completed",
            },
        ],
    )
    evidence_root = tmp_path / "evidence"
    trace_output = tmp_path / "trace/real_trace.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--evidence-root",
            str(evidence_root),
            "--machine-profile",
            str(machine_profile),
            "--trajectory-suite",
            str(trajectory_suite),
            "--trajectory-suite-sha256",
            hashlib.sha256(trajectory_suite.read_bytes()).hexdigest(),
            "--trajectory-controller-onnx",
            str(onnx),
            "--control-audit",
            str(audit),
            "--trace-output",
            str(trace_output),
            "--trace-run-id",
            "follow-run-001",
            "--policy-id",
            "onnx_rl:test-policy",
            "--evaluation-scope",
            "held_out_experiment",
            "--task-variant",
            "dig_transport_dump",
            "--operator-id",
            "zhaoshuai",
            "--material-id",
            "soil_default",
            "--run-id",
            "rl_real_suite_001",
        ],
        cwd=REPOSITORY,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(REPOSITORY / "src"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["run_id"] == "rl_real_suite_001"
    run_dir = Path(report["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["start"]["policy_ids"] == {
        "trajectory_controller": "onnx_rl:test-policy"
    }
    artifacts = [
        json.loads(line)
        for line in (run_dir / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [artifact["role"] for artifact in artifacts] == [
        "rl_onnx_model",
        "trajectory_suite",
        "rl_control_audit",
        "trajectory_trace",
    ]
    trace_records = [
        json.loads(line)
        for line in trace_output.read_text(encoding="utf-8").splitlines()
    ]
    assert [
        record["sample_id"]
        for record in trace_records
        if record["record_type"] == "policy_sample"
    ] == [0, 1]
    assert trace_records[-1]["result"] == "COMPLETED"

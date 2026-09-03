from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/convert_rl_sim_trace.py"


def test_script_converts_explicit_unity_audit_segment(tmp_path: Path):
    suite = tmp_path / "trajectory_suite.json"
    suite.write_text(
        json.dumps(
            {
                "suite_id": "suite-001",
                "sample_period_s": 0.1,
                "sample_ids": [0, 1],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    suite_sha256 = hashlib.sha256(suite.read_bytes()).hexdigest()
    audit = tmp_path / "unity_control_audit.jsonl"
    records = [
        {
            "schema_version": "rl_excavator_sim_control_audit.v1",
            "record_type": "policy_sample",
            "mode": "control",
            "status": "active",
            "trajectory_controller_backend": "onnx_rl",
            "trajectory_suite_sha256": suite_sha256,
            "trace_semantics": "commanded_normalized_action",
            "trace_run_id": "unity-follow-001",
            "sample_id": sample_id,
            "policy_action_seq": 100 + sample_id,
            "runtime_monotonic_s": 25.0 + 0.1 * sample_id,
            "action_order": ["boom", "stick", "bucket", "swing"],
            "commanded_normalized_action": [0.1, -0.2, 0.3, -0.4],
            "bucket_tip_ros_m": [0.8 + 0.1 * sample_id, 0.0, 0.0],
            "reference_waypoint_ros_m": [1.0, 0.0, 0.0],
            "waypoint_index": sample_id,
            "waypoint_distance_m": 0.2 - 0.1 * sample_id,
            "episode_progress": float(sample_id),
            "result": "active",
        }
        for sample_id in (0, 1)
    ]
    records.append(
        {
            "schema_version": "rl_excavator_sim_control_audit.v1",
            "record_type": "terminal",
            "mode": "control",
            "status": "terminal",
            "trajectory_controller_backend": "onnx_rl",
            "trajectory_suite_sha256": suite_sha256,
            "trace_semantics": "commanded_normalized_action",
            "trace_run_id": "unity-follow-001",
            "runtime_monotonic_s": 25.2,
            "elapsed_s": 0.2,
            "consumed_sample_count": 2,
            "suite_sample_count": 2,
            "result": "completed",
        }
    )
    audit.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "simulation_trace.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--control-audit",
            str(audit),
            "--trajectory-suite",
            str(suite),
            "--trace-output",
            str(output),
            "--trace-run-id",
            "unity-follow-001",
        ],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == {
        "first_policy_action_seq": 100,
        "first_sample_id": 0,
        "last_policy_action_seq": 101,
        "last_sample_id": 1,
        "sample_count": 2,
        "terminal_result": "COMPLETED",
        "trace_output": str(output),
        "trace_run_id": "unity-follow-001",
        "trace_semantics": "commanded_normalized_action",
        "trajectory_suite_sha256": suite_sha256,
    }

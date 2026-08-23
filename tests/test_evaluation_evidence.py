from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from excavator_il.evaluation_harness import evaluate_experiment_run
from excavator_il.experiment_run import ExperimentRun, TaskContext


class _Loader:
    def __init__(self, snapshot: SimpleNamespace) -> None:
        self._snapshot = snapshot

    def load(self, _run_path):
        return self._snapshot


def _snapshot(
    *,
    artifacts: tuple[dict[str, object], ...] = (),
    evidence_requirements: dict[str, object] | None = None,
    run_kind: str = "hybrid_live",
) -> SimpleNamespace:
    run_dir = (
        Path(str(artifacts[0]["source_path"])).parent
        if artifacts
        else None
    )
    return SimpleNamespace(
        run_id="run-evidence-001",
        run_dir=run_dir,
        state="success",
        start={
            "schema_version": "experiment_run.v1",
            "run_id": "run-evidence-001",
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
            "repositories": {},
            "evidence_requirements": evidence_requirements or {},
        },
        events=(),
        artifacts=artifacts,
        final={
            "status": "success",
            "metrics": {"evaluation_scope": "held_out_experiment"},
            "summary": "evidence fixture",
        },
        manifest={},
        verify_artifacts=lambda: None,
    )


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


def _act_step(state_ns: int, decision_ns: int) -> dict[str, object]:
    return {
        "schema_version": "excavator_act_runtime_step.v1",
        "state_monotonic_ns": state_ns,
        "camera_monotonic_ns": state_ns - 10_000_000,
        "decision_monotonic_ns": decision_ns,
        "predicted_action": [0.1, -0.2, 0.3, 0.0],
        "commanded_action": [0.0, 0.0, 0.0, 0.0],
        "reason": "shadow_mode",
        "serial_write_attempted": False,
        "requested_serial_axes": None,
        "effective_serial_axes": None,
        "final_gate_reason": None,
        "command_seq": None,
        "serial_write_performed": False,
        "dropped_state_count": 0,
    }


def _handoff_sample(
    *,
    generation: int,
    from_source: str,
    from_mode: str,
    to_source: str,
    to_mode: str,
    latency_ms: float,
) -> dict[str, object]:
    terminal_ack_ns = generation * 10_000_000_000 + 1_000_000_000
    target_ack_ns = terminal_ack_ns + 10_000_000
    first_write_ns = target_ack_ns + 20_000_000
    first_ack_ns = terminal_ack_ns + int(latency_ms * 1_000_000)
    return {
        "schema_version": "resident_handoff_sample.v1",
        "runtime_id": "runtime-001",
        "generation": generation,
        "from_source": from_source,
        "from_mode": from_mode,
        "to_source": to_source,
        "to_mode": to_mode,
        "terminal_zero_command_seq": generation * 3,
        "terminal_zero_ack_monotonic_ns": terminal_ack_ns,
        "target_zero_command_seq": generation * 3 + 1,
        "target_zero_ack_monotonic_ns": target_ack_ns,
        "first_nonzero_command_seq": generation * 3 + 2,
        "first_nonzero_action": [0.25, 0.0, -0.5, 0.0],
        "first_nonzero_write_monotonic_ns": first_write_ns,
        "first_nonzero_ack_monotonic_ns": first_ack_ns,
        "zero_claim_ms": 10.0,
        "policy_ready_wait_ms": 20.0,
        "first_command_ack_ms": (first_ack_ns - first_write_ns) / 1_000_000,
        "latency_ms": latency_ms,
    }


def test_real_collection_run_integrity_evaluates_required_quality_report(
    tmp_path,
    rgb_episode_factory,
):
    episode = rgb_episode_factory(step_count=3, dual_camera=True)
    quality_report = episode / "quality_report.json"
    quality_report.write_text(
        json.dumps(
            {
                "episode_id": episode.name,
                "passed": True,
                "camera_streams": {
                    role: {
                        "frame_count": 3,
                        "estimated_rate_hz": 10.0,
                        "age_ms": {"p50": 5.0, "p95": 8.0, "max": 9.0},
                        "sequence_gap_count": 0,
                        "queue_drop_count": 0,
                    }
                    for role in ("front", "dump")
                },
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "machine_profile.json"
    profile.write_text('{"action_order":["boom","stick","bucket","swing"]}\n')
    run = ExperimentRun.create(
        tmp_path / "evidence",
        run_id="collection_episode_0001",
        run_kind="collection_episode",
        task_context=TaskContext(
            task_variant="dig_transport_dump",
            soil_reset_block_id="block_01",
            dig_point_id="dig_01",
            operator_id="zhaoshuai",
            material_id="dry_soil",
        ),
        policy_ids={"collection_policy": "human_demonstration:v1"},
        host_topology={},
        repository_paths={},
        config_paths={},
        machine_profile_path=profile,
        evidence_requirements={
            "raw_episode": {"required": True, "min_count": 1},
            "quality_report": {"required": True, "min_count": 1},
        },
    )
    run.register_artifact("raw-episode", episode, role="raw_episode")
    run.register_artifact(
        "quality-report",
        quality_report,
        role="quality_report",
    )
    snapshot = run.finalize(
        "success",
        metrics={"evaluation_scope": "training_internal"},
        summary="formal dual-camera demonstration",
    )

    summary = evaluate_experiment_run(snapshot.run_dir)

    assert summary["passed"] is True
    assert summary["evidence"]["unevaluated_required_roles"] == []
    assert summary["evidence"]["evaluated_roles"] == [
        "quality_report",
        "raw_episode",
    ]
    assert summary["evidence"]["integrity_verified"] == {
        "artifact_set": True,
        "roles": ["quality_report"],
        "role_counts": {"quality_report": 1},
    }
    assert summary["data"]["episode_count"] == 1
    assert summary["data"]["cameras"]["camera_front"]["rate_hz"]["p50"] == 10.0
    assert summary["data"]["cameras"]["camera_dump"]["rate_hz"]["p50"] == 10.0


def test_raw_episode_artifact_reuses_validator_and_reports_camera_quality(
    tmp_path,
    rgb_episode_factory,
):
    episode = rgb_episode_factory(step_count=3)
    (episode / "quality_report.json").write_text(
        json.dumps(
            {
                "joystick_timeout_count": 0,
                "stream_timing": {
                    "camera_front": {
                        "count": 3,
                        "estimated_rate_hz": 30.1,
                        "mean_period_ms": 33.2,
                        "p95_period_ms": 34.0,
                        "max_period_ms": 35.0,
                    }
                },
                "camera_age_ms": {"p50": 4.0, "p95": 5.0, "max": 5.0},
                "camera_queue_drop_count": 0,
            }
        ),
        encoding="utf-8",
    )
    snapshot = _snapshot(
        artifacts=(_artifact(0, "raw_episode", episode),),
        evidence_requirements={
            "raw_episode": {"required": True, "min_count": 1}
        },
    )

    summary = evaluate_experiment_run(episode, loader=_Loader(snapshot))

    assert summary["passed"] is True
    assert summary["data"]["episode_count"] == 1
    assert summary["data"]["training_frame_count"] == 3
    assert summary["data"]["cameras"]["camera_front"] == {
        "frame_count": 3,
        "rate_hz": {"p50": 30.1, "p95": 30.1, "min": 30.1, "max": 30.1},
        "age_ms": {"p50": 4.0, "p95": 5.0, "max": 5.0},
        "sequence_gap_count": 0,
        "queue_drop_count": 0,
    }
    assert summary["evidence"]["evaluated_roles"] == ["raw_episode"]


def test_valid_episode_without_optional_quality_report_keeps_counts(
    rgb_episode_factory,
):
    episode = rgb_episode_factory(step_count=2)
    snapshot = _snapshot(
        run_kind="collection_episode",
        artifacts=(_artifact(0, "raw_episode", episode),),
    )

    summary = evaluate_experiment_run(episode, loader=_Loader(snapshot))

    assert summary["passed"] is True
    assert summary["data"]["episode_count"] == 1
    assert summary["data"]["training_frame_count"] == 2
    assert summary["data"]["cameras"]["camera_front"]["available"] is False


def test_dual_camera_quality_uses_canonical_front_and_dump_streams(
    rgb_episode_factory,
):
    episode = rgb_episode_factory(step_count=4, dual_camera=True)
    (episode / "quality_report.json").write_text(
        json.dumps(
            {
                "camera_streams": {
                    "front": {
                        "frame_count": 4,
                        "estimated_rate_hz": 30.0,
                        "age_ms": {"p50": 5.0, "p95": 8.0, "max": 9.0},
                        "sequence_gap_count": 1,
                        "queue_drop_count": 2,
                    },
                    "dump": {
                        "frame_count": 4,
                        "estimated_rate_hz": 29.5,
                        "age_ms": {"p50": 6.0, "p95": 9.0, "max": 10.0},
                        "sequence_gap_count": 0,
                        "queue_drop_count": 1,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    snapshot = _snapshot(
        run_kind="collection_episode",
        artifacts=(_artifact(0, "raw_episode", episode),),
    )

    summary = evaluate_experiment_run(episode, loader=_Loader(snapshot))

    assert summary["data"]["training_frame_count"] == 4
    assert summary["data"]["cameras"]["camera_front"]["sequence_gap_count"] == 1
    assert summary["data"]["cameras"]["camera_dump"] == {
        "frame_count": 4,
        "rate_hz": {"p50": 29.5, "p95": 29.5, "min": 29.5, "max": 29.5},
        "age_ms": {"p50": 6.0, "p95": 9.0, "max": 10.0},
        "sequence_gap_count": 0,
        "queue_drop_count": 1,
    }


def test_act_runtime_artifact_reports_inference_and_deadline_evidence(tmp_path):
    log = tmp_path / "act-shadow.jsonl"
    log.write_text(
        "".join(
            json.dumps(event) + "\n"
            for event in (
                _act_step(1_000_000_000, 1_020_000_000),
                _act_step(1_100_000_000, 1_120_000_000),
            )
        ),
        encoding="utf-8",
    )
    snapshot = _snapshot(
        artifacts=(
            _artifact(
                0,
                "runtime_log",
                log,
                analyzer="act_runtime",
                mode="shadow",
            ),
        ),
        evidence_requirements={
            "runtime_log": {"required": True, "min_count": 1}
        },
    )

    summary = evaluate_experiment_run(log, loader=_Loader(snapshot))

    assert summary["runtime"]["act"]["passed"] is True
    assert summary["runtime"]["act"]["step_count"] == 2
    assert summary["runtime"]["inference"] == {
        "available": True,
        "estimated_rate_hz": 10.0,
        "max_state_to_decision_ms": 20.0,
        "max_camera_age_ms": 10.0,
    }
    assert summary["runtime"]["deadline"] == {
        "available": True,
        "dropped_state_count": 0,
        "deadline_miss_count": None,
        "unavailable_reason": (
            "ACT Runtime evidence does not record a per-step deadline-miss counter"
        ),
    }
    assert summary["evidence"]["evaluated_roles"] == ["runtime_log"]


def test_resident_owner_log_reuses_handoff_analyzer_for_both_directions(
    tmp_path,
):
    log = tmp_path / "resident-owner.log"
    samples = (
        _handoff_sample(
            generation=1,
            from_source="rl_follow",
            from_mode="velocity_reference",
            to_source="act_dig",
            to_mode="manual_action",
            latency_ms=45.0,
        ),
        _handoff_sample(
            generation=2,
            from_source="act_dig",
            from_mode="manual_action",
            to_source="rl_follow",
            to_mode="velocity_reference",
            latency_ms=65.0,
        ),
    )
    log.write_text(
        "".join(
            "INFO RESIDENT_HANDOFF_SAMPLE " + json.dumps(sample) + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    snapshot = _snapshot(
        artifacts=(
            _artifact(
                0,
                "mission_log",
                log,
                analyzer="resident_handoff",
            ),
        ),
    )

    summary = evaluate_experiment_run(log, loader=_Loader(snapshot))

    assert summary["passed"] is False
    assert summary["handoffs"]["available"] is True
    assert summary["handoffs"]["sample_count"] == 2
    assert summary["handoffs"]["directions"] == {
        "act_dig/manual_action->rl_follow/velocity_reference": {
            "count": 1,
            "p50_ms": 65.0,
            "p95_ms": 65.0,
            "max_ms": 65.0,
            "passed": False,
        },
        "rl_follow/velocity_reference->act_dig/manual_action": {
            "count": 1,
            "p50_ms": 45.0,
            "p95_ms": 45.0,
            "max_ms": 45.0,
            "passed": False,
        },
    }
    assert summary["handoffs"]["benchmark_passed"] is False
    assert summary["failure_reasons"] == [
        "resident handoff benchmark: " + reason
        for reason in summary["handoffs"]["benchmark_failure_reasons"]
    ]
    assert summary["evidence"]["evaluated_roles"] == ["mission_log"]

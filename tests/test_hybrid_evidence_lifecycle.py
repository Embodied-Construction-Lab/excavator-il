import multiprocessing
import time

import pytest

from excavator_il.hybrid_experiment_run import (
    HybridMissionEvidenceLifecycle,
    HybridMissionRunRequest,
)
from excavator_il.hybrid_mission import REQUIRED_HYBRID_MOTION_AUTHORIZATION
from excavator_il.hybrid_mission_session import HybridMissionSupervisor


def _wait_for_stage(supervisor, stage, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = supervisor.snapshot()
        if snapshot.stage == stage:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"stage did not become {stage}: {supervisor.snapshot()}")


def _scripted_segment(
    _config_path,
    target_id,
    start_segment,
    automatic,
    _authorization,
    events,
    commands,
    _cycle_count=1,
    _dig_target_ids=(),
):
    assert target_id == "dig_02"
    assert automatic is False
    next_by_segment = {
        "rl_to_dig": ("awaiting_act_dig", "act_dig"),
        "act_dig": ("awaiting_rl_to_dump", "rl_to_dump_and_dump"),
        "rl_to_dump_and_dump": ("awaiting_rl_return", "rl_return_to_dig"),
        "rl_return_to_dig": ("completed", ""),
    }
    current = start_segment
    while True:
        events.put({"kind": "stage", "stage": f"running_{current}"})
        stage, next_segment = next_by_segment[current]
        if stage == "completed":
            events.put(
                {"kind": "terminal", "stage": stage, "next_segment": ""}
            )
            return
        events.put(
            {"kind": "waiting", "stage": stage, "next_segment": next_segment}
        )
        command = commands.get(timeout=2.0)
        if command["kind"] == "stop":
            events.put(
                {"kind": "terminal", "stage": "cancelled", "next_segment": ""}
            )
            return
        assert command["kind"] == "advance"
        assert command["segment"] == next_segment
        current = next_segment


def _target_reporting_worker(
    _config_path,
    _target_id,
    _start_segment,
    _automatic,
    _authorization,
    events,
    _commands,
    _cycle_count=1,
    _dig_target_ids=(),
):
    events.put(
        {
            "kind": "stage",
            "stage": "running_act_dig",
            "dig_target_id": "dig_03",
        }
    )
    events.put(
        {
            "kind": "terminal",
            "stage": "completed",
            "next_segment": "",
            "completed_cycles": 1,
        }
    )


def _two_cycle_evidence_worker(
    _config_path,
    _target_id,
    _start_segment,
    _automatic,
    _authorization,
    events,
    _commands,
    _cycle_count=1,
    _dig_target_ids=(),
):
    for segment, target_id in (
        ("rl_to_dig", "dig_01"),
        ("act_dig", "dig_01"),
        ("rl_to_dump_and_dump", "dig_01"),
        ("rl_return_to_dig", "dig_02"),
    ):
        events.put(
            {
                "kind": "stage",
                "stage": f"running_{segment}",
                "dig_target_id": target_id,
            }
        )
    events.put({"kind": "progress", "completed_cycles": 1})
    for segment in ("act_dig", "rl_to_dump_and_dump", "rl_return_to_dig"):
        events.put(
            {
                "kind": "stage",
                "stage": f"running_{segment}",
                "dig_target_id": "dig_02",
            }
        )
    events.put({"kind": "progress", "completed_cycles": 2})
    events.put(
        {
            "kind": "terminal",
            "stage": "completed",
            "next_segment": "",
            "completed_cycles": 2,
        }
    )


def _worker_exits_without_terminal(
    _config_path,
    _target_id,
    _start_segment,
    _automatic,
    _authorization,
    events,
    _commands,
    _cycle_count=1,
    _dig_target_ids=(),
):
    events.put(
        {
            "kind": "stage",
            "stage": "running_rl_to_dig",
            "dig_target_id": "dig_01",
        }
    )


def _worker_completes_after_phase(
    _config_path,
    _target_id,
    _start_segment,
    _automatic,
    _authorization,
    events,
    _commands,
    _cycle_count=1,
    _dig_target_ids=(),
):
    events.put(
        {
            "kind": "stage",
            "stage": "running_rl_to_dig",
            "dig_target_id": "dig_01",
        }
    )
    time.sleep(0.5)
    events.put(
        {
            "kind": "terminal",
            "stage": "completed",
            "next_segment": "",
            "completed_cycles": 1,
        }
    )


class _RecordingExperimentRun:
    def __init__(self, run_id="hybrid_run_001"):
        self.run_id = run_id
        self.events = []
        self.finalizations = []

    def append_event(self, event_type, payload=None):
        self.events.append((event_type, dict(payload or {})))

    def finalize(self, status, *, metrics=None, summary=None):
        self.finalizations.append((status, dict(metrics or {}), summary))


def test_evidence_lifecycle_retains_failed_finalization_for_retry():
    class _FlakyRun(_RecordingExperimentRun):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def finalize(self, status, *, metrics=None, summary=None):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("evidence volume unavailable")
            super().finalize(status, metrics=metrics, summary=summary)

    recorder = _FlakyRun()
    lifecycle = HybridMissionEvidenceLifecycle(
        recorder,
        cycle_targets=("dig_01",),
    )
    lifecycle.start_mission(
        automatic=False,
        requested_cycles=1,
        dig_target_id="dig_01",
    )

    lifecycle.finish(
        stage="completed",
        error="",
        requested_cycles=1,
        completed_cycles=1,
        automatic=False,
    )

    assert lifecycle.finalization_pending is True
    assert "evidence volume unavailable" in lifecycle.error
    assert [event for event, _payload in recorder.events].count("mission_completed") == 1

    lifecycle.retry_finalize()

    assert lifecycle.finalization_pending is False
    assert lifecycle.error == ""
    assert recorder.attempts == 2
    assert [event for event, _payload in recorder.events].count("mission_completed") == 1


def test_supervisor_creates_one_evidence_run_and_surfaces_stable_run_id(tmp_path):
    recorder = _RecordingExperimentRun()
    requests = []

    def create_run(request):
        requests.append(request)
        return recorder

    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_target_reporting_worker,
        process_context=multiprocessing.get_context("spawn"),
        evidence_run_factory=create_run,
    )
    try:
        supervisor.start("dig_01", automatic=False, motion_authorization=None)
        completed = _wait_for_stage(supervisor, "completed")

        assert requests == [
            HybridMissionRunRequest(
                config_path=(tmp_path / "hybrid.json").resolve(),
                dig_target_id="dig_01",
                automatic=False,
                requested_cycles=1,
            )
        ]
        assert completed.run_id == "hybrid_run_001"
        assert supervisor.snapshot().run_id == "hybrid_run_001"
    finally:
        supervisor.close()


def test_supervisor_does_not_launch_when_initial_evidence_cannot_be_written(tmp_path):
    class _BrokenRun(_RecordingExperimentRun):
        def append_event(self, _event_type, _payload=None):
            raise OSError("evidence volume unavailable")

    recorder = _BrokenRun()
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_target_reporting_worker,
        process_context=multiprocessing.get_context("spawn"),
        evidence_run_factory=lambda _request: recorder,
    )

    with pytest.raises(RuntimeError, match="initial hybrid Mission evidence"):
        supervisor.start("dig_01", automatic=False, motion_authorization=None)

    failed = supervisor.snapshot()
    assert failed.stage == "failed"
    assert "evidence volume unavailable" in failed.evidence_error
    assert recorder.finalizations[0][0] == "failure"
    supervisor.close()


def test_supervisor_aborts_active_mission_when_evidence_append_fails(tmp_path):
    class _BrokenPhaseRun(_RecordingExperimentRun):
        def append_event(self, event_type, payload=None):
            if event_type == "phase_started":
                raise OSError("evidence volume unavailable")
            super().append_event(event_type, payload)

    recorder = _BrokenPhaseRun()
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_worker_completes_after_phase,
        process_context=multiprocessing.get_context("spawn"),
        evidence_run_factory=lambda _request: recorder,
    )
    try:
        supervisor.start("dig_01", automatic=False, motion_authorization=None)
        failed = _wait_for_stage(supervisor, "failed")

        assert "evidence recording failed" in failed.error
        assert "evidence volume unavailable" in failed.evidence_error
        assert recorder.finalizations[0][0] == "failure"
    finally:
        supervisor.close()


def test_supervisor_records_segment_cycle_and_success_terminal_evidence(tmp_path):
    recorder = _RecordingExperimentRun()
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        dig_target_ids=("dig_01", "dig_02"),
        worker_target=_two_cycle_evidence_worker,
        process_context=multiprocessing.get_context("spawn"),
        evidence_run_factory=lambda _request: recorder,
    )
    try:
        supervisor.start(
            "dig_01",
            automatic=True,
            motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
            cycle_count=2,
        )
        completed = _wait_for_stage(supervisor, "completed")

        event_types = [event_type for event_type, _payload in recorder.events]
        assert event_types == [
            "mission_started",
            "cycle_started",
            "phase_started",
            "phase_completed",
            "phase_started",
            "phase_completed",
            "phase_started",
            "phase_completed",
            "phase_started",
            "phase_completed",
            "cycle_completed",
            "cycle_started",
            "phase_started",
            "phase_completed",
            "phase_started",
            "phase_completed",
            "phase_started",
            "phase_completed",
            "cycle_completed",
            "mission_completed",
        ]
        assert all(
            payload["run_id"] == completed.run_id
            for _event_type, payload in recorder.events
        )
        assert recorder.events[1][1] == {
            "run_id": "hybrid_run_001",
            "cycle_index": 0,
            "dig_target_id": "dig_01",
        }
        assert recorder.events[11][1] == {
            "run_id": "hybrid_run_001",
            "cycle_index": 1,
            "dig_target_id": "dig_02",
        }
        assert recorder.finalizations == [
            (
                "success",
                {
                    "requested_cycles": 2,
                    "completed_cycles": 2,
                    "automatic": True,
                    "terminal_stage": "completed",
                },
                "hybrid Mission completed",
            )
        ]
    finally:
        supervisor.close()


def test_supervisor_finalizes_cancelled_mission_as_failed_evidence(tmp_path):
    recorder = _RecordingExperimentRun()
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_scripted_segment,
        process_context=multiprocessing.get_context("spawn"),
        evidence_run_factory=lambda _request: recorder,
    )
    try:
        supervisor.start("dig_02", automatic=False, motion_authorization=None)
        _wait_for_stage(supervisor, "awaiting_act_dig")

        supervisor.stop()
        cancelled = _wait_for_stage(supervisor, "cancelled")

        assert recorder.events[-2] == (
            "cycle_completed",
            {
                "run_id": cancelled.run_id,
                "cycle_index": 0,
                "dig_target_id": "dig_02",
                "outcome": "cancelled",
                "completed_cycles": 0,
            },
        )
        assert recorder.events[-1][0] == "mission_cancelled"
        assert recorder.finalizations == [
            (
                "failure",
                {
                    "requested_cycles": 1,
                    "completed_cycles": 0,
                    "automatic": False,
                    "terminal_stage": "cancelled",
                },
                "hybrid Mission cancelled",
            )
        ]
    finally:
        supervisor.close()


def test_worker_exit_without_terminal_closes_failed_evidence_intervals(tmp_path):
    recorder = _RecordingExperimentRun()
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_worker_exits_without_terminal,
        process_context=multiprocessing.get_context("spawn"),
        evidence_run_factory=lambda _request: recorder,
    )
    try:
        supervisor.start("dig_01", automatic=False, motion_authorization=None)
        failed = _wait_for_stage(supervisor, "failed")

        assert [event_type for event_type, _payload in recorder.events][-3:] == [
            "phase_completed",
            "cycle_completed",
            "mission_failed",
        ]
        assert recorder.events[-3][1]["outcome"] == "failure"
        assert recorder.events[-2][1]["outcome"] == "failure"
        assert recorder.finalizations[0][0] == "failure"
        assert "worker exited" in failed.error
    finally:
        supervisor.close()


def test_supervisor_can_retry_failed_evidence_publication(tmp_path):
    class _FlakyRun(_RecordingExperimentRun):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def finalize(self, status, *, metrics=None, summary=None):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("evidence volume unavailable")
            super().finalize(status, metrics=metrics, summary=summary)

    recorder = _FlakyRun()
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_target_reporting_worker,
        process_context=multiprocessing.get_context("spawn"),
        evidence_run_factory=lambda _request: recorder,
    )
    try:
        supervisor.start("dig_01", automatic=False, motion_authorization=None)
        completed = _wait_for_stage(supervisor, "completed")
        assert "evidence volume unavailable" in completed.evidence_error

        with pytest.raises(RuntimeError, match="finalization is pending"):
            supervisor.start("dig_01", automatic=False, motion_authorization=None)

        supervisor.retry_evidence_finalization()

        assert supervisor.snapshot().evidence_error == ""
        assert recorder.attempts == 2
    finally:
        supervisor.close()

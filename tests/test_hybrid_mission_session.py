import multiprocessing
import queue
import threading
import time

import pytest

from excavator_il.hybrid_mission import REQUIRED_HYBRID_MOTION_AUTHORIZATION
from excavator_il.hybrid_mission_session import (
    HybridMissionSupervisor,
    run_hybrid_mission_worker,
)


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


def test_segmented_hybrid_supervisor_advances_only_in_contract_order(tmp_path):
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_scripted_segment,
        process_context=multiprocessing.get_context("spawn"),
    )
    try:
        supervisor.start("dig_02", automatic=False, motion_authorization=None)
        first = _wait_for_stage(supervisor, "awaiting_act_dig")
        assert first.next_segment == "act_dig"
        worker_pid = supervisor._process.pid

        with pytest.raises(ValueError, match="authorization"):
            supervisor.advance(motion_authorization="wrong")
        supervisor.advance(
            motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION
        )
        _wait_for_stage(supervisor, "awaiting_rl_to_dump")
        assert supervisor._process.pid == worker_pid
        supervisor.advance(motion_authorization=None)
        _wait_for_stage(supervisor, "awaiting_rl_return")
        supervisor.advance(motion_authorization=None)
        completed = _wait_for_stage(supervisor, "completed")

        assert completed.completed_cycles == 1
        assert completed.next_segment == ""
    finally:
        supervisor.close()


def test_automatic_hybrid_start_requires_exact_authorization(tmp_path):
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_scripted_segment,
        process_context=multiprocessing.get_context("spawn"),
    )
    with pytest.raises(ValueError, match="authorization"):
        supervisor.start("dig_01", automatic=True, motion_authorization=None)
    supervisor.close()


def test_segmented_hybrid_supervisor_can_cancel_while_prewarm_is_waiting(tmp_path):
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_scripted_segment,
        process_context=multiprocessing.get_context("spawn"),
    )
    try:
        supervisor.start("dig_02", automatic=False, motion_authorization=None)
        _wait_for_stage(supervisor, "awaiting_act_dig")

        supervisor.stop()

        cancelled = _wait_for_stage(supervisor, "cancelled")
        assert cancelled.next_segment == ""
    finally:
        supervisor.close()


def test_hybrid_worker_runs_all_segments_and_emits_completion(monkeypatch):
    events = queue.Queue()
    commands = queue.Queue()
    calls = []

    class _Config:
        act_max_steps = 130

    class _Operations:
        def __init__(self, _config, output):
            self.output = output

        def run_rl_to_dig(self, target):
            calls.append(("dig", target))

        def run_act_dig(self, steps):
            calls.append(("act", steps))

        def run_rl_to_dump_and_dump(self):
            calls.append(("dump",))

        def run_rl_return_to_dig(self, target):
            calls.append(("return", target))

        def safe_stop(self):
            calls.append(("stop",))

    monkeypatch.setattr(
        "excavator_il.hybrid_mission_session.HybridMissionConfig.load",
        lambda _path: _Config(),
    )
    monkeypatch.setattr(
        "excavator_il.hybrid_mission_session.SystemHybridMissionOperations",
        _Operations,
    )

    run_hybrid_mission_worker(
        "hybrid.json",
        "dig_03",
        "rl_to_dig",
        True,
        REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        events,
        commands,
    )

    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    assert calls == [
        ("dig", "dig_03"),
        ("act", 130),
        ("dump",),
        ("return", "dig_03"),
    ]
    assert emitted[-1] == {
        "kind": "terminal",
        "stage": "completed",
        "next_segment": "",
    }


def test_segmented_worker_keeps_one_operations_instance_between_clicks(monkeypatch):
    events = queue.Queue()
    commands = queue.Queue()
    calls = []
    instances = []

    class _Config:
        act_max_steps = 130

    class _Operations:
        def __init__(self, _config, output):
            instances.append(self)
            self.output = output

        def run_rl_to_dig(self, target):
            calls.append(("dig", target))

        def run_act_dig(self, steps):
            calls.append(("act", steps))

        def run_rl_to_dump_and_dump(self):
            calls.append(("dump",))

        def run_rl_return_to_dig(self, target):
            calls.append(("return", target))

        def safe_stop(self):
            calls.append(("stop",))

    monkeypatch.setattr(
        "excavator_il.hybrid_mission_session.HybridMissionConfig.load",
        lambda _path: _Config(),
    )
    monkeypatch.setattr(
        "excavator_il.hybrid_mission_session.SystemHybridMissionOperations",
        _Operations,
    )
    worker = threading.Thread(
        target=run_hybrid_mission_worker,
        args=(
            "hybrid.json",
            "dig_01",
            "rl_to_dig",
            False,
            None,
            events,
            commands,
        ),
    )
    worker.start()

    for segment, authorization in (
        ("act_dig", REQUIRED_HYBRID_MOTION_AUTHORIZATION),
        ("rl_to_dump_and_dump", None),
        ("rl_return_to_dig", None),
    ):
        while True:
            event = events.get(timeout=1.0)
            if event.get("kind") == "waiting":
                break
        commands.put(
            {
                "kind": "advance",
                "segment": segment,
                "motion_authorization": authorization,
            }
        )

    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert len(instances) == 1
    assert calls == [
        ("dig", "dig_01"),
        ("act", 130),
        ("dump",),
        ("return", "dig_01"),
    ]

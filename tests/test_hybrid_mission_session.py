import multiprocessing
import queue
import threading
import time

import pytest

from excavator_il.hybrid_mission import REQUIRED_HYBRID_MOTION_AUTHORIZATION
from excavator_il.hybrid_mission_session import (
    HybridMissionSupervisor,
    _watch_parent_identity,
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


def test_parent_identity_watcher_interrupts_when_supervisor_disappears():
    observed_parent_pids = iter((42, 42, 1))
    sleeps = []
    interrupts = []

    _watch_parent_identity(
        42,
        get_parent_pid=lambda: next(observed_parent_pids),
        sleep=lambda seconds: sleeps.append(seconds),
        interrupt=lambda: interrupts.append("SIGINT"),
    )

    assert sleeps == [0.1, 0.1]
    assert interrupts == ["SIGINT"]


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


def _failed_worker_that_lingers_during_cleanup(
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
            "kind": "terminal",
            "stage": "failed",
            "next_segment": "",
            "error": "Follow failed: MOTION_GATE_CLOSED",
        }
    )
    time.sleep(5.0)


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
        assert completed.run_completed_cycles == 1
        assert completed.next_segment == ""
    finally:
        supervisor.close()


def test_hybrid_snapshot_reports_the_current_cycle_target(tmp_path):
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        worker_target=_target_reporting_worker,
        process_context=multiprocessing.get_context("spawn"),
    )
    try:
        supervisor.start("dig_01", automatic=False, motion_authorization=None)

        completed = _wait_for_stage(supervisor, "completed")

        assert completed.dig_target_id == "dig_03"
    finally:
        supervisor.close()


def test_failed_snapshot_keeps_safety_stop_available_while_worker_cleans_up(
    tmp_path,
):
    supervisor = HybridMissionSupervisor(
        config_path=tmp_path / "hybrid.json",
        worker_target=_failed_worker_that_lingers_during_cleanup,
        process_context=multiprocessing.get_context("spawn"),
    )
    try:
        supervisor.start("dig_01", automatic=False, motion_authorization=None)

        failed = _wait_for_stage(supervisor, "failed")

        assert failed.can_stop is True
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

        def prewarm_next_act(self, steps):
            calls.append(("prewarm", steps))

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
        ("stop",),
    ]
    assert emitted[-1] == {
        "kind": "terminal",
        "stage": "completed",
        "next_segment": "",
        "completed_cycles": 1,
    }


def test_keyboard_interrupt_reports_failure_when_safe_stop_cannot_be_confirmed(
    monkeypatch,
):
    events = queue.Queue()

    class _Config:
        act_max_steps = 130

    class _Operations:
        def __init__(self, _config, output):
            del output

        def run_rl_to_dig(self, _target):
            raise KeyboardInterrupt

        def safe_stop(self):
            raise RuntimeError("terminal zero ACK timeout")

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
        "dig_01",
        "rl_to_dig",
        False,
        None,
        events,
        queue.Queue(),
    )

    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    terminal = emitted[-1]
    assert terminal["kind"] == "terminal"
    assert terminal["stage"] == "failed"
    assert "terminal zero ACK timeout" in terminal["error"]


@pytest.mark.parametrize(
    ("selected_target", "cycle_count", "expected_calls", "expected_act_targets"),
    [
        (
            "dig_01",
            3,
            [
                ("dig", "dig_01"),
                ("act", 130),
                ("dump",),
                ("prewarm", 130),
                ("return", "dig_02"),
                ("act", 130),
                ("dump",),
                ("prewarm", 130),
                ("return", "dig_03"),
                ("act", 130),
                ("dump",),
                ("return", "dig_03"),
            ],
            ["dig_01", "dig_02", "dig_03"],
        ),
        (
            "dig_02",
            4,
            [
                ("dig", "dig_02"),
                ("act", 130),
                ("dump",),
                ("prewarm", 130),
                ("return", "dig_03"),
                ("act", 130),
                ("dump",),
                ("prewarm", 130),
                ("return", "dig_01"),
                ("act", 130),
                ("dump",),
                ("prewarm", 130),
                ("return", "dig_02"),
                ("act", 130),
                ("dump",),
                ("return", "dig_02"),
            ],
            ["dig_02", "dig_03", "dig_01", "dig_02"],
        ),
        (
            "dig_03",
            1,
            [
                ("dig", "dig_03"),
                ("act", 130),
                ("dump",),
                ("return", "dig_03"),
            ],
            ["dig_03"],
        ),
    ],
)
def test_automatic_worker_cycles_targets_from_selected_point_without_restarting_adapter(
    monkeypatch,
    selected_target,
    cycle_count,
    expected_calls,
    expected_act_targets,
):
    events = queue.Queue()
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

        def prewarm_next_act(self, steps):
            calls.append(("prewarm", steps))

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
        selected_target,
        "rl_to_dig",
        True,
        REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        events,
        queue.Queue(),
        cycle_count=cycle_count,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
    )

    assert len(instances) == 1
    assert calls == [*expected_calls, ("stop",)]
    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    assert [
        event["dig_target_id"]
        for event in emitted
        if event.get("stage") == "running_act_dig"
    ] == expected_act_targets
    assert [
        event["completed_cycles"]
        for event in emitted
        if event["kind"] == "progress"
    ] == list(range(1, cycle_count + 1))
    assert emitted[-1] == {
        "kind": "terminal",
        "stage": "completed",
        "next_segment": "",
        "completed_cycles": cycle_count,
    }


def test_automatic_worker_accepts_nine_cycle_truck_loading(monkeypatch):
    events = queue.Queue()
    act_targets = []
    current_target = {"value": ""}

    class _Config:
        act_max_steps = 130

    class _Operations:
        def __init__(self, _config, output):
            self.output = output

        def run_rl_to_dig(self, target):
            current_target["value"] = target

        def run_act_dig(self, _steps):
            act_targets.append(current_target["value"])

        def run_rl_to_dump_and_dump(self):
            pass

        def run_rl_return_to_dig(self, target):
            current_target["value"] = target

        def prewarm_next_act(self, _steps):
            pass

        def safe_stop(self):
            pass

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
        "dig_01",
        "rl_to_dig",
        True,
        REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        events,
        queue.Queue(),
        cycle_count=9,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
    )

    assert act_targets == ["dig_01", "dig_02", "dig_03"] * 3


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
        ("stop",),
    ]


def test_worker_selects_resident_backend_from_authoritative_config(monkeypatch):
    events = queue.Queue()
    calls = []

    class _Config:
        act_max_steps = 130
        runtime_backend = "resident"

    class _ResidentOperations:
        def __init__(self, _config, output):
            calls.append(("constructed", output is not None))

        def run_rl_to_dig(self, target):
            calls.append(("dig", target))

        def run_act_dig(self, steps):
            calls.append(("act", steps))

        def run_rl_to_dump_and_dump(self):
            calls.append(("dump",))

        def run_rl_return_to_dig(self, target):
            calls.append(("return", target))

        def prewarm_next_act(self, steps):
            calls.append(("resident_ready", steps))

        def safe_stop(self):
            calls.append(("stop",))

    class _LegacyOperations:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("legacy backend must not be constructed")

    monkeypatch.setattr(
        "excavator_il.hybrid_mission_session.HybridMissionConfig.load",
        lambda _path: _Config(),
    )
    monkeypatch.setattr(
        "excavator_il.hybrid_mission_session.SystemHybridMissionOperations",
        _LegacyOperations,
    )
    monkeypatch.setattr(
        "excavator_il.hybrid_mission_session.SystemResidentHybridMissionOperations",
        _ResidentOperations,
    )

    run_hybrid_mission_worker(
        "hybrid.json",
        "dig_01",
        "rl_to_dig",
        True,
        REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        events,
        queue.Queue(),
    )

    assert calls == [
        ("constructed", True),
        ("dig", "dig_01"),
        ("act", 130),
        ("dump",),
        ("return", "dig_01"),
        ("stop",),
    ]

import multiprocessing
import queue
import time

import pytest

from excavator_il import guided_episode
from excavator_il.collection_ui_session import (
    GuidedCollectionSupervisor,
    run_guided_collection_worker,
)


def _scripted_manual_collection(
    _config_path,
    positioning_mode,
    dig_target_id,
    commands,
    events,
    task_variant=None,
    soil_reset_block_id=None,
    dig_point_id=None,
):
    assert positioning_mode == "manual"
    assert dig_target_id is None
    assert (task_variant, soil_reset_block_id, dig_point_id) == (
        "dig_only",
        "block_04",
        "dig_02",
    )
    events.put({"kind": "stage", "stage": "manual_positioning"})
    assert commands.get(timeout=1.0) == {"command": "complete_manual_positioning"}
    events.put({"kind": "stage", "stage": "review"})
    assert commands.get(timeout=1.0) == {
        "command": "submit_outcome",
        "outcome": "success",
    }
    events.put(
        {
            "kind": "terminal",
            "stage": "completed",
            "episode_path": "/data/episode_0001",
        }
    )


def _cancellable_collection(
    _config_path,
    _positioning_mode,
    _dig_target_id,
    _commands,
    events,
    *_protocol,
):
    events.put({"kind": "stage", "stage": "recording"})
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        events.put({"kind": "terminal", "stage": "cancelled"})


def _scripted_standalone_teleop(
    _config_path,
    positioning_mode,
    dig_target_id,
    _commands,
    events,
    *_protocol,
):
    assert positioning_mode == "teleop"
    assert dig_target_id is None
    events.put({"kind": "stage", "stage": "teleoperation"})
    events.put({"kind": "terminal", "stage": "completed"})


def _wait_for_stage(supervisor, stage, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = supervisor.snapshot()
        if snapshot.stage == stage:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(
        f"stage did not become {stage!r}: {supervisor.snapshot()}"
    )


def test_guided_collection_supervisor_exposes_one_stateful_control_interface(tmp_path):
    supervisor = GuidedCollectionSupervisor(
        config_path=tmp_path / "guided.json",
        worker_target=_scripted_manual_collection,
        process_context=multiprocessing.get_context("spawn"),
    )

    try:
        assert supervisor.snapshot().stage == "idle"

        supervisor.start(
            "manual",
            task_variant="dig_only",
            soil_reset_block_id="block_04",
            dig_point_id="dig_02",
        )
        _wait_for_stage(supervisor, "manual_positioning")
        supervisor.complete_manual_positioning()
        _wait_for_stage(supervisor, "review")
        supervisor.submit_outcome("success")
        completed = _wait_for_stage(supervisor, "completed")

        assert completed.episode_path == "/data/episode_0001"
        assert completed.positioning_mode == "manual"
        assert completed.task_variant == "dig_only"
        assert completed.soil_reset_block_id == "block_04"
        assert completed.dig_point_id == "dig_02"
        assert completed.completed_count == 1
        assert completed.error == ""

        supervisor.start(
            "manual",
            task_variant="dig_only",
            soil_reset_block_id="block_04",
            dig_point_id="dig_02",
        )
        _wait_for_stage(supervisor, "manual_positioning")
        supervisor.complete_manual_positioning()
        _wait_for_stage(supervisor, "review")
        supervisor.submit_outcome("success")
        second = _wait_for_stage(supervisor, "completed")
        assert second.completed_count == 2
    finally:
        supervisor.close()


def test_guided_collection_supervisor_interrupts_the_owned_worker_on_stop(tmp_path):
    supervisor = GuidedCollectionSupervisor(
        config_path=tmp_path / "guided.json",
        worker_target=_cancellable_collection,
        process_context=multiprocessing.get_context("spawn"),
    )

    try:
        supervisor.start("direct")
        _wait_for_stage(supervisor, "recording")
        supervisor.stop()

        cancelled = _wait_for_stage(supervisor, "cancelled")
        assert cancelled.error == ""
    finally:
        supervisor.close()


def test_guided_collection_supervisor_accepts_teleop_without_counting_episode(tmp_path):
    supervisor = GuidedCollectionSupervisor(
        config_path=tmp_path / "guided.json",
        worker_target=_scripted_standalone_teleop,
        process_context=multiprocessing.get_context("spawn"),
    )

    try:
        supervisor.start("teleop")
        completed = _wait_for_stage(supervisor, "completed")
        assert completed.positioning_mode == "teleop"
        assert completed.episode_path == ""
        assert completed.completed_count == 0
    finally:
        supervisor.close()


class _StartFailureProcess:
    def start(self):
        raise OSError("process creation failed")


class _StartFailureContext:
    def __init__(self):
        self._queues = multiprocessing.get_context("spawn")

    def Queue(self):
        return self._queues.Queue()

    def Process(self, **_kwargs):
        return _StartFailureProcess()


def test_guided_collection_supervisor_recovers_from_process_start_failure(tmp_path):
    supervisor = GuidedCollectionSupervisor(
        config_path=tmp_path / "guided.json",
        process_context=_StartFailureContext(),
    )

    with pytest.raises(RuntimeError, match="cannot start guided collection"):
        supervisor.start("direct")

    failed = supervisor.snapshot()
    assert failed.stage == "failed"
    assert "process creation failed" in failed.error
    supervisor.close()


def test_guided_collection_worker_adapts_structured_ui_commands(monkeypatch):
    commands = queue.Queue()
    events = queue.Queue()
    commands.put({"command": "complete_manual_positioning"})
    commands.put({"command": "submit_outcome", "outcome": "success"})
    config = object()

    monkeypatch.setattr(guided_episode.GuidedEpisodeConfig, "load", lambda _path: config)
    monkeypatch.setattr(
        guided_episode,
        "SystemGuidedEpisodeOperations",
        lambda loaded, output: (loaded, output),
    )

    def fake_run(
        loaded,
        _operations,
        *,
        positioning_mode,
        input_fn,
        output,
        stage_callback,
        rl_target_id,
    ):
        assert loaded is config
        assert positioning_mode is guided_episode.PositioningMode.MANUAL
        assert rl_target_id is None
        stage_callback(guided_episode.GuidedEpisodeStage.MANUAL_POSITIONING)
        assert input_fn("manual") == "c"
        stage_callback(guided_episode.GuidedEpisodeStage.REVIEW)
        assert input_fn("review") == "s"
        output("Episode sealed")
        return "/data/episode_0002"

    monkeypatch.setattr(guided_episode, "run_guided_episode", fake_run)

    run_guided_collection_worker("guided.json", "manual", None, commands, events)

    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    assert {"kind": "log", "message": "Episode sealed"} in emitted
    assert emitted[-1] == {
        "kind": "terminal",
        "stage": "completed",
        "episode_path": "/data/episode_0002",
    }


def test_guided_collection_worker_passes_selected_rl_target(monkeypatch):
    commands = queue.Queue()
    events = queue.Queue()
    config = object()
    calls = []

    monkeypatch.setattr(guided_episode.GuidedEpisodeConfig, "load", lambda _path: config)
    monkeypatch.setattr(
        guided_episode,
        "SystemGuidedEpisodeOperations",
        lambda loaded, output: (loaded, output),
    )
    monkeypatch.setattr(
        guided_episode,
        "run_guided_episode",
        lambda _config, _operations, **kwargs: calls.append(kwargs)
        or "/data/episode_0003",
    )

    run_guided_collection_worker(
        "guided.json",
        "rl",
        "dig_03",
        commands,
        events,
        "dig_transport_dump",
        "block_09",
        "dig_03",
    )

    assert calls[0]["rl_target_id"] == "dig_03"
    assert calls[0]["task_variant"] == "dig_transport_dump"
    assert calls[0]["soil_reset_block_id"] == "block_09"
    assert calls[0]["dig_point_id"] == "dig_03"


def test_guided_collection_supervisor_rejects_partial_protocol_metadata(tmp_path):
    supervisor = GuidedCollectionSupervisor(config_path=tmp_path / "guided.json")

    with pytest.raises(ValueError, match="must be provided together"):
        supervisor.start("direct", task_variant="dig_only")

    assert supervisor.snapshot().stage == "idle"
    supervisor.close()


def test_guided_collection_worker_runs_teleop_without_episode_lifecycle(monkeypatch):
    commands = queue.Queue()
    events = queue.Queue()
    config = object()
    calls = []

    monkeypatch.setattr(guided_episode.GuidedEpisodeConfig, "load", lambda _path: config)
    monkeypatch.setattr(
        guided_episode,
        "SystemGuidedEpisodeOperations",
        lambda loaded, output: (loaded, output),
    )

    def fake_teleop(loaded, _operations, *, wait_fn, output, stage_callback):
        assert loaded is config
        assert callable(wait_fn)
        stage_callback(guided_episode.GuidedEpisodeStage.TELEOPERATION)
        output("standalone teleop ready")
        calls.append("teleop")

    monkeypatch.setattr(guided_episode, "run_standalone_teleop", fake_teleop)
    monkeypatch.setattr(
        guided_episode,
        "run_guided_episode",
        lambda *_args, **_kwargs: pytest.fail("Episode workflow must not run"),
    )

    run_guided_collection_worker("guided.json", "teleop", None, commands, events)

    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    assert calls == ["teleop"]
    assert {"kind": "stage", "stage": "teleoperation"} in emitted
    assert {"kind": "log", "message": "standalone teleop ready"} in emitted
    assert emitted[-1] == {"kind": "terminal", "stage": "completed"}

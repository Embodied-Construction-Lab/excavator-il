"""Process-isolated supervisor for segmented or repeated hybrid Missions."""

from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import threading
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .hybrid_mission import (
    REQUIRED_HYBRID_MOTION_AUTHORIZATION,
    HybridMissionConfig,
    HybridMissionSegment,
    execute_hybrid_segment,
    next_hybrid_segment,
)
from .hybrid_mission_system import SystemHybridMissionOperations


_FINAL_STAGES = frozenset({"completed", "failed", "cancelled"})
_WAITING_STAGES = {
    HybridMissionSegment.ACT_DIG: "awaiting_act_dig",
    HybridMissionSegment.RL_TO_DUMP_AND_DUMP: "awaiting_rl_to_dump",
    HybridMissionSegment.RL_RETURN_TO_DIG: "awaiting_rl_return",
}


@dataclass(frozen=True)
class HybridMissionSnapshot:
    stage: str = "idle"
    dig_target_id: str = ""
    automatic: bool = False
    next_segment: str = ""
    error: str = ""
    logs: tuple[str, ...] = ()
    # Lifetime total remains available for experiment bookkeeping.  The UI uses
    # run_completed_cycles so a new 4-scoop run always starts at 0 / 4.
    completed_cycles: int = 0
    run_completed_cycles: int = 0
    requested_cycles: int = 1


def run_hybrid_mission_worker(
    config_path: str,
    dig_target_id: str,
    start_segment: str,
    automatic: bool,
    motion_authorization: str | None,
    events: Any,
    commands: Any,
    cycle_count: int = 1,
    dig_target_ids: tuple[str, ...] = (),
) -> None:
    if not isinstance(cycle_count, int) or isinstance(cycle_count, bool):
        raise ValueError("cycle_count must be an integer")
    if not 1 <= cycle_count <= 5:
        raise ValueError("cycle_count must be within [1, 5]")
    target_cycle = _rotated_target_cycle(dig_target_id, dig_target_ids)
    config = HybridMissionConfig.load(config_path)
    operations = SystemHybridMissionOperations(
        config,
        output=lambda message: events.put({"kind": "log", "message": str(message)}),
    )
    segment = HybridMissionSegment(start_segment)
    current_authorization = motion_authorization
    completed_cycles = 0
    current_target_id = dig_target_id
    try:
        while True:
            segment_target_id = current_target_id
            prewarm_next_act = (
                automatic
                and segment is HybridMissionSegment.RL_RETURN_TO_DIG
                and completed_cycles + 1 < cycle_count
            )
            if prewarm_next_act:
                segment_target_id = target_cycle[
                    (completed_cycles + 1) % len(target_cycle)
                ]
            events.put(
                {
                    "kind": "stage",
                    "stage": f"running_{segment.value}",
                    "dig_target_id": segment_target_id,
                }
            )
            if prewarm_next_act:
                operations.prewarm_next_act(config.act_max_steps)
            execute_hybrid_segment(
                operations,
                segment=segment,
                dig_target_id=segment_target_id,
                act_max_steps=config.act_max_steps,
                motion_authorization=current_authorization,
            )
            events.put(
                {
                    "kind": "log",
                    "message": f"混合 Mission 分段完成：{segment.value}",
                }
            )
            next_segment = next_hybrid_segment(segment)
            if next_segment is None:
                completed_cycles += 1
                events.put(
                    {
                        "kind": "progress",
                        "completed_cycles": completed_cycles,
                    }
                )
                events.put(
                    {
                        "kind": "log",
                        "message": f"装车循环完成：{completed_cycles}/{cycle_count}",
                    }
                )
                if automatic and completed_cycles < cycle_count:
                    # Return-to-dig is already the next cycle's handoff pose.
                    current_target_id = segment_target_id
                    segment = HybridMissionSegment.ACT_DIG
                    continue
                events.put(
                    {
                        "kind": "terminal",
                        "stage": "completed",
                        "next_segment": "",
                        "completed_cycles": completed_cycles,
                    }
                )
                return
            if automatic:
                segment = next_segment
                continue
            events.put(
                {
                    "kind": "waiting",
                    "stage": _WAITING_STAGES[next_segment],
                    "next_segment": next_segment.value,
                }
            )
            command = commands.get()
            if isinstance(command, dict) and command.get("kind") == "stop":
                operations.safe_stop()
                events.put(
                    {
                        "kind": "terminal",
                        "stage": "cancelled",
                        "next_segment": "",
                    }
                )
                return
            if not isinstance(command, dict) or command.get("kind") != "advance":
                raise RuntimeError("invalid hybrid Mission worker command")
            requested = HybridMissionSegment(command.get("segment"))
            if requested is not next_segment:
                raise RuntimeError(
                    f"hybrid Mission expected {next_segment.value}, got {requested.value}"
                )
            current_authorization = command.get("motion_authorization")
            segment = requested
    except KeyboardInterrupt:
        try:
            operations.safe_stop()
        except Exception as exc:
            events.put({"kind": "log", "message": f"停止清理失败：{exc}"})
        events.put(
            {"kind": "terminal", "stage": "cancelled", "next_segment": ""}
        )
        return
    except BaseException as exc:
        cleanup_error = ""
        try:
            operations.safe_stop()
        except Exception as cleanup_exc:
            cleanup_error = f"; cleanup={cleanup_exc}"
        events.put(
            {
                "kind": "terminal",
                "stage": "failed",
                "next_segment": "",
                "error": f"{type(exc).__name__}: {exc}{cleanup_error}",
            }
        )
        return


def _rotated_target_cycle(
    selected_target_id: str,
    available_target_ids: tuple[str, ...],
) -> tuple[str, ...]:
    targets = _validated_target_ids(available_target_ids)
    if not targets:
        return (selected_target_id,)
    try:
        start = targets.index(selected_target_id)
    except ValueError as exc:
        raise ValueError("selected dig_target_id is not in dig_target_ids") from exc
    return targets[start:] + targets[:start]


def _validated_target_ids(target_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(target_ids, tuple):
        raise ValueError("dig_target_ids must be a tuple")
    if any(
        not isinstance(target_id, str) or not target_id.strip()
        for target_id in target_ids
    ):
        raise ValueError("dig_target_ids must contain non-empty strings")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("dig_target_ids must be unique")
    return target_ids


class HybridMissionSupervisor:
    """Expose the closed loop as ordered, interruptible operator segments."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        dig_target_ids: tuple[str, ...] = (),
        worker_target: Callable[..., None] = run_hybrid_mission_worker,
        process_context: Any | None = None,
        log_capacity: int = 400,
    ) -> None:
        if log_capacity <= 0:
            raise ValueError("log_capacity must be positive")
        self._config_path = Path(config_path).expanduser().resolve()
        self._dig_target_ids = _validated_target_ids(dig_target_ids)
        self._worker_target = worker_target
        self._context = process_context or multiprocessing.get_context("spawn")
        self._log_capacity = log_capacity
        self._lock = threading.RLock()
        self._logs: deque[str] = deque(maxlen=log_capacity)
        self._state = HybridMissionSnapshot()
        self._completed_cycles = 0
        self._events: Any | None = None
        self._commands: Any | None = None
        self._process: Any | None = None
        self._monitor: threading.Thread | None = None

    def snapshot(self) -> HybridMissionSnapshot:
        with self._lock:
            return replace(self._state, logs=tuple(self._logs))

    def start(
        self,
        dig_target_id: str,
        *,
        automatic: bool,
        motion_authorization: str | None,
        cycle_count: int = 1,
    ) -> None:
        if not isinstance(dig_target_id, str) or not dig_target_id.strip():
            raise ValueError("dig_target_id must be non-empty")
        if not isinstance(automatic, bool):
            raise ValueError("automatic must be boolean")
        if not isinstance(cycle_count, int) or isinstance(cycle_count, bool):
            raise ValueError("cycle_count must be an integer")
        if not 1 <= cycle_count <= 5:
            raise ValueError("cycle_count must be within [1, 5]")
        if not automatic and cycle_count != 1:
            raise ValueError("segmented Mission supports exactly one cycle")
        if automatic and motion_authorization != REQUIRED_HYBRID_MOTION_AUTHORIZATION:
            raise ValueError("automatic Mission requires exact motion authorization")
        with self._lock:
            if self._state.stage not in {"idle", *_FINAL_STAGES}:
                raise RuntimeError("a hybrid Mission is already active")
            self._logs.clear()
            self._state = HybridMissionSnapshot(
                stage="starting",
                dig_target_id=dig_target_id,
                automatic=automatic,
                next_segment=HybridMissionSegment.RL_TO_DIG.value,
                completed_cycles=self._completed_cycles,
                run_completed_cycles=0,
                requested_cycles=cycle_count,
            )
        self._launch(
            segment=HybridMissionSegment.RL_TO_DIG,
            automatic=automatic,
            motion_authorization=motion_authorization,
            cycle_count=cycle_count,
        )

    def advance(self, *, motion_authorization: str | None) -> None:
        with self._lock:
            next_value = self._state.next_segment
            if self._state.stage not in _WAITING_STAGES.values() or not next_value:
                raise RuntimeError("hybrid Mission is not waiting for the next segment")
            segment = HybridMissionSegment(next_value)
            if (
                segment is HybridMissionSegment.ACT_DIG
                and motion_authorization != REQUIRED_HYBRID_MOTION_AUTHORIZATION
            ):
                raise ValueError("ACT segment requires exact motion authorization")
            commands = self._commands
            process = self._process
            if commands is None or process is None or not process.is_alive():
                raise RuntimeError("hybrid Mission worker is not available")
            self._state = replace(
                self._state,
                stage="starting",
                automatic=False,
                next_segment=segment.value,
                error="",
            )
            commands.put(
                {
                    "kind": "advance",
                    "segment": segment.value,
                    "motion_authorization": motion_authorization,
                }
            )

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.is_alive():
                was_waiting = self._state.stage in _WAITING_STAGES.values()
                self._state = replace(self._state, stage="stopping")
                if was_waiting and self._commands is not None:
                    self._commands.put({"kind": "stop"})
                    return
                os.kill(process.pid, signal.SIGINT)
                return
            raise RuntimeError("no hybrid Mission is active")

    def close(self) -> None:
        with self._lock:
            process = self._process
            monitor = self._monitor
        if process is not None and process.is_alive():
            try:
                self.stop()
            except (OSError, RuntimeError):
                pass
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=15.0)
        if process is not None and process.is_alive():
            process.join(timeout=5.0)
        with self._lock:
            self._release_finished_locked()

    def _launch(
        self,
        *,
        segment: HybridMissionSegment,
        automatic: bool,
        motion_authorization: str | None,
        cycle_count: int,
    ) -> None:
        with self._lock:
            prior_monitor = self._monitor
        if prior_monitor is not None and prior_monitor is not threading.current_thread():
            prior_monitor.join(timeout=2.0)
        with self._lock:
            self._release_finished_locked()
            events = self._context.Queue()
            commands = self._context.Queue()
            self._events = events
            self._commands = commands
            self._state = replace(
                self._state,
                stage="starting",
                automatic=automatic,
                next_segment=segment.value,
                error="",
            )
            process = self._context.Process(
                target=self._worker_target,
                args=(
                    str(self._config_path),
                    self._state.dig_target_id,
                    segment.value,
                    automatic,
                    motion_authorization,
                    events,
                    commands,
                    cycle_count,
                    self._dig_target_ids,
                ),
                name=f"hybrid-mission-{segment.value}",
            )
            try:
                process.start()
            except Exception as exc:
                self._state = replace(
                    self._state,
                    stage="failed",
                    next_segment="",
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._close_events_locked()
                raise RuntimeError(f"cannot start hybrid Mission worker: {exc}") from exc
            self._process = process
            monitor = threading.Thread(
                target=self._monitor_events,
                args=(process, events),
                name="hybrid-mission-monitor",
                daemon=True,
            )
            self._monitor = monitor
            monitor.start()

    def _monitor_events(self, process: Any, events: Any) -> None:
        terminal_received = False
        while not terminal_received:
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                if process.is_alive():
                    continue
                break
            terminal_received = self._apply_event(event)
        process.join(timeout=1.0)
        with self._lock:
            if self._process is process and not terminal_received:
                self._state = replace(
                    self._state,
                    stage="failed",
                    next_segment="",
                    error=f"hybrid Mission worker exited with code {process.exitcode}",
                )

    def _apply_event(self, event: object) -> bool:
        if not isinstance(event, dict):
            return False
        with self._lock:
            if event.get("kind") == "log":
                message = event.get("message")
                if isinstance(message, str) and message:
                    self._logs.append(message)
                return False
            if event.get("kind") == "stage":
                stage = event.get("stage")
                if isinstance(stage, str) and stage:
                    target_id = event.get("dig_target_id")
                    if (
                        isinstance(target_id, str)
                        and target_id.strip()
                        and (
                            not self._dig_target_ids
                            or target_id in self._dig_target_ids
                        )
                    ):
                        self._state = replace(
                            self._state,
                            stage=stage,
                            dig_target_id=target_id,
                        )
                    else:
                        self._state = replace(self._state, stage=stage)
                return False
            if event.get("kind") == "waiting":
                stage = event.get("stage")
                next_segment = event.get("next_segment")
                if stage not in _WAITING_STAGES.values():
                    return False
                if not isinstance(next_segment, str) or not next_segment:
                    return False
                self._state = replace(
                    self._state,
                    stage=stage,
                    next_segment=next_segment,
                    error="",
                )
                return False
            if event.get("kind") == "progress":
                completed_cycles = event.get("completed_cycles")
                if (
                    not isinstance(completed_cycles, int)
                    or isinstance(completed_cycles, bool)
                    or not 0 <= completed_cycles <= self._state.requested_cycles
                ):
                    return False
                self._state = replace(
                    self._state,
                    run_completed_cycles=completed_cycles,
                )
                return False
            if event.get("kind") != "terminal":
                return False
            stage = event.get("stage")
            allowed = {*_FINAL_STAGES, *_WAITING_STAGES.values()}
            if stage not in allowed:
                stage = "failed"
            next_segment = event.get("next_segment", "")
            error = event.get("error", "")
            if stage == "completed":
                completed_cycles = event.get("completed_cycles", 1)
                if (
                    not isinstance(completed_cycles, int)
                    or isinstance(completed_cycles, bool)
                    or completed_cycles < 1
                ):
                    completed_cycles = 1
                self._completed_cycles += completed_cycles
            self._state = replace(
                self._state,
                stage=stage,
                next_segment=next_segment if isinstance(next_segment, str) else "",
                error=error if isinstance(error, str) else "",
                completed_cycles=self._completed_cycles,
                run_completed_cycles=(
                    completed_cycles
                    if stage == "completed"
                    else self._state.run_completed_cycles
                ),
            )
            return True

    def _release_finished_locked(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            raise RuntimeError("hybrid Mission worker is still active")
        if process is not None:
            process.join(timeout=0.1)
        self._close_events_locked()
        self._process = None
        self._monitor = None

    def _close_events_locked(self) -> None:
        if self._events is not None:
            self._events.close()
            self._events.join_thread()
        self._events = None
        if self._commands is not None:
            self._commands.close()
            self._commands.join_thread()
        self._commands = None

"""Process-isolated control interface for one guided Demonstration Episode."""

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


_TERMINAL_STAGES = frozenset({"completed", "failed", "cancelled"})
_POSITIONING_MODES = frozenset({"rl", "manual", "direct", "teleop"})
_OUTCOMES = frozenset({"success", "failure", "retake"})


@dataclass(frozen=True)
class CollectionSessionSnapshot:
    stage: str = "idle"
    positioning_mode: str = ""
    dig_target_id: str = ""
    task_variant: str = ""
    soil_reset_block_id: str = ""
    dig_point_id: str = ""
    episode_path: str = ""
    error: str = ""
    logs: tuple[str, ...] = ()
    completed_count: int = 0


def run_guided_collection_worker(
    config_path: str,
    positioning_mode: str,
    dig_target_id: str | None,
    commands: Any,
    events: Any,
    task_variant: str | None = None,
    soil_reset_block_id: str | None = None,
    dig_point_id: str | None = None,
) -> None:
    """Run the existing guided workflow behind structured process messages."""
    from .guided_episode import (
        GuidedEpisodeConfig,
        GuidedEpisodeStage,
        PositioningMode,
        SystemGuidedEpisodeOperations,
        run_guided_episode,
        run_standalone_teleop,
    )

    current_stage = GuidedEpisodeStage.PREFLIGHT

    def emit_stage(stage: GuidedEpisodeStage) -> None:
        nonlocal current_stage
        current_stage = stage
        events.put({"kind": "stage", "stage": stage.value})

    def emit_log(message: str) -> None:
        events.put({"kind": "log", "message": str(message)})

    def wait_for_operator(_prompt: str) -> str:
        command = commands.get()
        if current_stage is GuidedEpisodeStage.MANUAL_POSITIONING:
            if command == {"command": "complete_manual_positioning"}:
                return "c"
            raise RuntimeError("manual positioning requires completion command")
        if current_stage is GuidedEpisodeStage.REVIEW:
            if not isinstance(command, dict) or command.get("command") != "submit_outcome":
                raise RuntimeError("Episode review requires an outcome command")
            outcome = command.get("outcome")
            token = {"success": "s", "failure": "f", "retake": "r"}.get(outcome)
            if token is None:
                raise RuntimeError("invalid Episode review outcome")
            return token
        raise RuntimeError(f"operator input is not valid during {current_stage.value}")

    try:
        config = GuidedEpisodeConfig.load(config_path)
        operations = SystemGuidedEpisodeOperations(config, output=emit_log)
        if positioning_mode == "teleop":
            run_standalone_teleop(
                config,
                operations,
                wait_fn=signal.pause,
                output=emit_log,
                stage_callback=emit_stage,
            )
            episode_path = ""
        else:
            if task_variant is not None:
                emit_log(
                    "[episode-context] "
                    f"task_variant={task_variant} "
                    f"soil_reset_block_id={soil_reset_block_id} "
                    f"dig_point_id={dig_point_id}"
                )
            run_options: dict[str, Any] = {
                "positioning_mode": PositioningMode(positioning_mode),
                "input_fn": wait_for_operator,
                "output": emit_log,
                "stage_callback": emit_stage,
                "rl_target_id": dig_target_id,
            }
            if task_variant is not None:
                run_options.update(
                    task_variant=task_variant,
                    soil_reset_block_id=soil_reset_block_id,
                    dig_point_id=dig_point_id,
                )
            episode_path = run_guided_episode(config, operations, **run_options)
    except KeyboardInterrupt:
        events.put({"kind": "terminal", "stage": "cancelled"})
    except BaseException as exc:
        events.put(
            {
                "kind": "terminal",
                "stage": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        event = {"kind": "terminal", "stage": "completed"}
        if episode_path:
            event["episode_path"] = episode_path
        events.put(event)


class GuidedCollectionSupervisor:
    """Own one guided collection child and expose only valid operator actions."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        worker_target: Callable[..., None] = run_guided_collection_worker,
        process_context: Any | None = None,
        log_capacity: int = 300,
    ) -> None:
        if log_capacity <= 0:
            raise ValueError("log_capacity must be positive")
        self._config_path = Path(config_path).expanduser().resolve()
        self._worker_target = worker_target
        self._context = process_context or multiprocessing.get_context("spawn")
        self._log_capacity = log_capacity
        self._lock = threading.RLock()
        self._state = CollectionSessionSnapshot()
        self._completed_count = 0
        self._logs: deque[str] = deque(maxlen=log_capacity)
        self._commands: Any | None = None
        self._events: Any | None = None
        self._process: Any | None = None
        self._monitor: threading.Thread | None = None

    def snapshot(self) -> CollectionSessionSnapshot:
        with self._lock:
            return replace(self._state, logs=tuple(self._logs))

    def clear_logs(self) -> None:
        """Clear the operator-visible log without changing session state."""

        with self._lock:
            self._logs.clear()

    def start(
        self,
        positioning_mode: str,
        dig_target_id: str | None = None,
        *,
        task_variant: str | None = None,
        soil_reset_block_id: str | None = None,
        dig_point_id: str | None = None,
    ) -> None:
        if positioning_mode not in _POSITIONING_MODES:
            raise ValueError("positioning_mode must be rl, manual, direct or teleop")
        from .collector.config import validate_collection_protocol

        protocol = validate_collection_protocol(
            task_variant=task_variant,
            soil_reset_block_id=soil_reset_block_id,
            dig_point_id=dig_point_id,
        )
        if positioning_mode == "teleop" and protocol:
            raise ValueError("teleop does not create an Episode or accept collection protocol")
        with self._lock:
            if self._state.stage not in {"idle", *_TERMINAL_STAGES}:
                raise RuntimeError("a guided collection session is already active")
            prior_monitor = self._monitor
        if prior_monitor is not None and prior_monitor is not threading.current_thread():
            prior_monitor.join(timeout=2.0)
        with self._lock:
            if self._state.stage not in {"idle", *_TERMINAL_STAGES}:
                raise RuntimeError("a guided collection session is already active")
            self._release_finished_process_locked()
            self._logs.clear()
            self._commands = self._context.Queue()
            self._events = self._context.Queue()
            self._state = CollectionSessionSnapshot(
                stage="starting",
                positioning_mode=positioning_mode,
                dig_target_id=dig_target_id or "",
                task_variant=task_variant or "",
                soil_reset_block_id=soil_reset_block_id or "",
                dig_point_id=dig_point_id or "",
                completed_count=self._completed_count,
            )
            process = self._context.Process(
                target=self._worker_target,
                args=(
                    str(self._config_path),
                    positioning_mode,
                    dig_target_id,
                    self._commands,
                    self._events,
                    task_variant,
                    soil_reset_block_id,
                    dig_point_id,
                ),
                name="guided-collection-worker",
            )
            try:
                process.start()
            except Exception as exc:
                self._state = replace(
                    self._state,
                    stage="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._close_channels_locked()
                raise RuntimeError(
                    f"cannot start guided collection worker: {exc}"
                ) from exc
            self._process = process
            monitor = threading.Thread(
                target=self._monitor_events,
                args=(process, self._events),
                name="guided-collection-monitor",
                daemon=True,
            )
            self._monitor = monitor
            monitor.start()

    def complete_manual_positioning(self) -> None:
        self._send_command(
            required_stage="manual_positioning",
            command={"command": "complete_manual_positioning"},
        )

    def submit_outcome(self, outcome: str) -> None:
        if outcome not in _OUTCOMES:
            raise ValueError("outcome must be success, failure or retake")
        self._send_command(
            required_stage="review",
            command={"command": "submit_outcome", "outcome": outcome},
        )

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None or not process.is_alive():
                raise RuntimeError("no guided collection session is active")
            self._state = replace(self._state, stage="stopping")
            os.kill(process.pid, signal.SIGINT)

    def close(self) -> None:
        with self._lock:
            process = self._process
            monitor = self._monitor
            terminal_received = self._state.stage in _TERMINAL_STAGES
        if process is not None and process.is_alive():
            if not terminal_received:
                try:
                    self.stop()
                except (OSError, RuntimeError):
                    pass
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=10.0)
        if process is not None and process.is_alive():
            process.join(timeout=10.0)
        with self._lock:
            self._release_finished_process_locked()

    def _send_command(self, *, required_stage: str, command: dict[str, str]) -> None:
        with self._lock:
            if self._state.stage != required_stage or self._commands is None:
                raise RuntimeError(
                    f"operator action requires stage {required_stage}, "
                    f"current stage is {self._state.stage}"
                )
            self._commands.put(command)

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
            if self._process is not process:
                return
            if self._state.stage not in _TERMINAL_STAGES:
                self._state = replace(
                    self._state,
                    stage="failed",
                    error=f"guided collection worker exited with code {process.exitcode}",
                )

    def _apply_event(self, event: object) -> bool:
        if not isinstance(event, dict):
            return False
        kind = event.get("kind")
        with self._lock:
            if kind == "log":
                message = event.get("message")
                if isinstance(message, str) and message:
                    self._logs.append(message)
                return False
            if kind == "stage":
                stage = event.get("stage")
                if isinstance(stage, str) and stage:
                    self._state = replace(self._state, stage=stage)
                return False
            if kind == "terminal":
                stage = event.get("stage")
                if stage not in _TERMINAL_STAGES:
                    stage = "failed"
                episode_path = event.get("episode_path", "")
                error = event.get("error", "")
                if (
                    stage == "completed"
                    and isinstance(episode_path, str)
                    and episode_path
                ):
                    self._completed_count += 1
                self._state = replace(
                    self._state,
                    stage=stage,
                    episode_path=(
                        episode_path if isinstance(episode_path, str) else ""
                    ),
                    error=error if isinstance(error, str) else "",
                    completed_count=self._completed_count,
                )
                return True
        return False

    def _release_finished_process_locked(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            raise RuntimeError("guided collection worker is still active")
        if process is not None:
            process.join(timeout=0.1)
        self._close_channels_locked()
        self._process = None
        self._monitor = None

    def _close_channels_locked(self) -> None:
        for channel in (self._commands, self._events):
            if channel is not None:
                channel.close()
                channel.join_thread()
        self._commands = None
        self._events = None

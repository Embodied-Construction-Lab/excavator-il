"""STM32 command and fresh-state control for the online ACT Runtime."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .act_runtime import RuntimeMode
from .stm32_protocol import Stm32ManualCommandEncoder, Stm32TelemetryFrame


@dataclass(frozen=True)
class CommandWriteResult:
    requested_axes: tuple[float, ...]
    effective_axes: tuple[float, ...]
    command_seq: int | None
    final_gate_reason: str
    write_performed: bool


class LatestStateQueue:
    """One-slot queue so GPU latency cannot replay a telemetry backlog."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._dropped_count = 0
        self._lock = threading.Lock()
        self._dropped_since_get = 0

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def put(self, frame: Any, *, external_gap_count: int = 0) -> None:
        if external_gap_count < 0:
            raise ValueError("external gap count must be non-negative")
        if external_gap_count:
            with self._lock:
                self._dropped_count += external_gap_count
                self._dropped_since_get += external_gap_count
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            self._dropped_count += 1
            self._dropped_since_get += 1
        self._queue.put_nowait(frame)

    def get(self, *, timeout_s: float) -> tuple[Any, int]:
        frame = self._queue.get(timeout=timeout_s)
        with self._lock:
            dropped = self._dropped_since_get
            self._dropped_since_get = 0
        return frame, dropped


class SensorSequenceTracker:
    """Detect missing or reset 10 Hz sensor states, including uint32 wrap."""

    def __init__(self) -> None:
        self._previous: int | None = None

    def observe(self, sensor_seq: int) -> int:
        sequence = int(sensor_seq) & 0xFFFFFFFF
        if self._previous is None:
            self._previous = sequence
            return 0
        expected = (self._previous + 1) & 0xFFFFFFFF
        self._previous = sequence
        return 0 if sequence == expected else 1


class Stm32CommandChannel:
    """Enforce the shadow no-write invariant at the physical serial boundary."""

    def __init__(
        self,
        *,
        serial_port: Any,
        encoder: Stm32ManualCommandEncoder,
        mode: RuntimeMode,
        max_state_age_ms: float = 100.0,
        record_command: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_state_age_ms <= 0:
            raise ValueError("runtime state timeout must be positive")
        self._serial = serial_port
        self._encoder = encoder
        self._mode = RuntimeMode(mode)
        self._max_state_age_ns = int(max_state_age_ms * 1_000_000)
        self._synchronized = False
        self._terminally_disarmed = False
        self._state_monotonic_ns: int | None = None
        self._state_timeout_zero_sent = False
        self._state_generation = 0
        self._state_safe = False
        self._motion_epoch = 0
        self._record_command = record_command
        self._lock = threading.Lock()

    @property
    def mode(self) -> RuntimeMode:
        return self._mode

    @property
    def synchronized(self) -> bool:
        return self._synchronized

    def synchronize(self, frame: Stm32TelemetryFrame) -> int:
        with self._lock:
            self._synchronized = True
            return self._encoder.synchronize(frame)

    @property
    def motion_epoch(self) -> int:
        with self._lock:
            return self._motion_epoch

    def _interrupt_motion_locked(self) -> None:
        self._motion_epoch += 1

    def _write_locked(
        self, axes: tuple[float, ...], monotonic_ns: int, *, reason: str
    ) -> int | None:
        if self._mode is RuntimeMode.SHADOW:
            return None
        if not self._synchronized:
            raise RuntimeError("STM32 command sequence is not synchronized")
        command_sequence = self._encoder.next_sequence
        payload = self._encoder.encode(axes=axes, monotonic_ns=monotonic_ns)
        written = self._serial.write(payload)
        if written != len(payload):
            raise OSError(f"short serial write: {written}/{len(payload)} bytes")
        self._serial.flush()
        if self._record_command is not None:
            self._record_command(
                {
                    "schema_version": "excavator_act_runtime_command.v1",
                    "command_monotonic_ns": monotonic_ns,
                    "command_seq": command_sequence,
                    "serial_axes": list(axes),
                    "reason": reason,
                    "serial_write_performed": True,
                }
            )
        return command_sequence

    def write_axes(
        self,
        axes: tuple[float, ...],
        *,
        monotonic_ns: int,
        state_generation: int | None = None,
        motion_epoch: int | None = None,
    ) -> CommandWriteResult:
        with self._lock:
            requested = axes
            if self._terminally_disarmed:
                return CommandWriteResult(
                    requested, (0.0,) * 6, None, "terminally_disarmed", False
                )
            nonzero = any(abs(value) > 1e-12 for value in axes)
            expected_motion_epoch = (
                self._motion_epoch if motion_epoch is None else motion_epoch
            )
            gate_reason = "accepted"
            if nonzero and expected_motion_epoch != self._motion_epoch:
                axes = (0.0,) * 6
                gate_reason = "motion_interrupted"
            state_fresh = (
                self._state_monotonic_ns is None
                or (
                    monotonic_ns >= self._state_monotonic_ns
                    and monotonic_ns - self._state_monotonic_ns
                    <= self._max_state_age_ns
                )
            )
            if nonzero and (
                state_generation is None
                or state_generation != self._state_generation
                or not state_fresh
                or not self._state_safe
                or self._state_timeout_zero_sent
            ):
                axes = (0.0,) * 6
                gate_reason = "state_not_fresh_or_current"
            sequence = self._write_locked(axes, monotonic_ns, reason=gate_reason)
            return CommandWriteResult(
                requested_axes=requested,
                effective_axes=axes,
                command_seq=sequence,
                final_gate_reason=(
                    "shadow_no_write"
                    if self._mode is RuntimeMode.SHADOW
                    else gate_reason
                ),
                write_performed=self._mode is RuntimeMode.MOTION,
            )

    def safe_zero(self, *, monotonic_ns: int, reason: str) -> int | None:
        if not reason:
            raise ValueError("safe-zero reason must be non-empty")
        with self._lock:
            return self._write_locked((0.0,) * 6, monotonic_ns, reason=reason)

    def update_state(self, frame: Stm32TelemetryFrame) -> int:
        with self._lock:
            was_safe = self._state_safe
            self._state_monotonic_ns = frame.receive_monotonic_ns
            values = frame.values
            self._state_safe = (
                all(
                    int(values.get(field, 0)) == 1
                    for field in ("control_enabled", "rs485_ok", "dwj_ok", "imu_ok")
                )
                and int(values.get("estop", 1)) == 0
                and int(values.get("fault_flags", 1)) == 0
            )
            safety_changed = self._state_safe != was_safe
            if frame.sensor_is_new or safety_changed:
                self._state_generation += 1
            self._state_timeout_zero_sent = False
            if was_safe and not self._state_safe and not self._terminally_disarmed:
                self._interrupt_motion_locked()
                self._write_locked(
                    (0.0,) * 6,
                    frame.receive_monotonic_ns,
                    reason="unsafe_telemetry",
                )
            return self._state_generation

    def enforce_state_timeout(self, *, monotonic_ns: int) -> bool:
        with self._lock:
            expired = (
                self._state_monotonic_ns is not None
                and monotonic_ns - self._state_monotonic_ns > self._max_state_age_ns
            )
            if not expired or self._state_timeout_zero_sent:
                return False
            self._state_timeout_zero_sent = True
            self._interrupt_motion_locked()
            self._write_locked((0.0,) * 6, monotonic_ns, reason="state_timeout")
            return True

    def terminal_disarm(self, *, monotonic_ns: int, reason: str) -> None:
        if not reason:
            raise ValueError("terminal disarm reason must be non-empty")
        with self._lock:
            if self._terminally_disarmed:
                return
            self._terminally_disarmed = True
            self._interrupt_motion_locked()
            if self._synchronized:
                self._write_locked((0.0,) * 6, monotonic_ns, reason=reason)

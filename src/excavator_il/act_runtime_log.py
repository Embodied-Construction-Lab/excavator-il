"""Offline acceptance checks for ACT Runtime JSONL evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


STEP_SCHEMA = "excavator_act_runtime_step.v1"
COMMAND_SCHEMA = "excavator_act_runtime_command.v1"


@dataclass(frozen=True)
class ActRuntimeLogReport:
    passed: bool
    mode: str
    step_count: int
    command_event_count: int
    serial_write_count: int
    nonzero_serial_write_count: int
    dropped_state_count: int
    estimated_step_rate_hz: float
    max_state_to_decision_ms: float
    max_camera_age_ms: float
    failure_reasons: tuple[str, ...]


def _timestamp(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _vector(value: Any, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain {length} values")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise ValueError(f"{field} must contain only numeric values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) and -1.000001 <= item <= 1.000001 for item in result):
        raise ValueError(f"{field} must be finite and within [-1,1]")
    return result


def _same(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        abs(a - b) <= 1e-9 for a, b in zip(left, right, strict=True)
    )


def inspect_act_runtime_log(
    path: str | Path,
    *,
    mode: str,
    max_state_to_decision_ms: float = 100.0,
    max_camera_age_ms: float = 120.0,
    min_step_rate_hz: float = 8.0,
    max_step_rate_hz: float = 12.0,
) -> ActRuntimeLogReport:
    """Validate one completed Shadow or motion Runtime log without hardware."""

    if mode not in ("shadow", "motion"):
        raise ValueError("ACT Runtime log mode must be shadow or motion")
    if not all(
        math.isfinite(limit) and limit > 0
        for limit in (
            max_state_to_decision_ms,
            max_camera_age_ms,
            min_step_rate_hz,
            max_step_rate_hz,
        )
    ):
        raise ValueError("ACT Runtime log limits must be finite and positive")
    if min_step_rate_hz >= max_step_rate_hz:
        raise ValueError("ACT Runtime minimum step rate must be below maximum")

    log_path = Path(path)
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read ACT Runtime log {log_path}: {exc}") from exc

    failures: list[str] = []
    state_stamps: list[int] = []
    state_to_decision_ms: list[float] = []
    camera_ages_ms: list[float] = []
    step_count = 0
    command_event_count = 0
    serial_write_count = 0
    nonzero_serial_write_count = 0
    dropped_state_count = 0
    commands: list[tuple[int, tuple[float, ...], str, int]] = []
    step_writes: list[tuple[int, tuple[float, ...], str]] = []

    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event must be an object")
            schema = event.get("schema_version")
            if schema == STEP_SCHEMA:
                step_count += 1
                dropped_state_count += _timestamp(
                    event.get("dropped_state_count"), "dropped_state_count"
                )
                state_ns = _timestamp(event.get("state_monotonic_ns"), "state_monotonic_ns")
                camera_ns = _timestamp(
                    event.get("camera_monotonic_ns"), "camera_monotonic_ns"
                )
                decision_ns = _timestamp(
                    event.get("decision_monotonic_ns"), "decision_monotonic_ns"
                )
                predicted = _vector(
                    event.get("predicted_action"), "predicted_action", 4
                )
                commanded = _vector(
                    event.get("commanded_action"), "commanded_action", 4
                )
                if not camera_ns <= state_ns <= decision_ns:
                    raise ValueError("camera/state/decision timestamps are not causal")
                state_age_ms = (decision_ns - state_ns) / 1_000_000.0
                camera_age_ms = (state_ns - camera_ns) / 1_000_000.0
                state_stamps.append(state_ns)
                state_to_decision_ms.append(state_age_ms)
                camera_ages_ms.append(camera_age_ms)
                if state_age_ms > max_state_to_decision_ms:
                    failures.append(f"line {line_number}: state-to-decision age exceeded")
                if camera_age_ms > max_camera_age_ms:
                    failures.append(f"line {line_number}: camera age exceeded")
                if mode == "shadow" and (
                    commanded != (0.0,) * 4
                    or event.get("serial_write_attempted") is not False
                    or event.get("serial_write_performed") is not False
                    or event.get("requested_serial_axes") is not None
                    or event.get("effective_serial_axes") is not None
                    or event.get("final_gate_reason") is not None
                    or event.get("command_seq") is not None
                ):
                    failures.append(f"line {line_number}: shadow attempted motion")
                if mode == "motion":
                    requested = _vector(
                        event.get("requested_serial_axes"),
                        "requested_serial_axes",
                        6,
                    )
                    effective = _vector(
                        event.get("effective_serial_axes"),
                        "effective_serial_axes",
                        6,
                    )
                    final_gate_reason = event.get("final_gate_reason")
                    if not isinstance(final_gate_reason, str) or not final_gate_reason:
                        raise ValueError("motion final gate reason must be non-empty text")
                    if event.get("serial_write_attempted") is not True:
                        raise ValueError("motion step did not attempt a serial write")
                    write_performed = event.get("serial_write_performed")
                    if write_performed is True:
                        command_seq = _timestamp(
                            event.get("command_seq"), "command_seq"
                        )
                        step_writes.append(
                            (command_seq, effective, final_gate_reason)
                        )
                    elif (
                        write_performed is not False
                        or event.get("command_seq") is not None
                        or final_gate_reason != "terminally_disarmed"
                        or any(abs(value) > 1e-9 for value in effective)
                    ):
                        raise ValueError(
                            "motion step without a serial write must be terminally disarmed"
                        )
                    if event.get("reason") == "motion_allowed":
                        expected_axes = (
                            predicted[3],
                            predicted[1],
                            0.0,
                            predicted[2],
                            predicted[0],
                            0.0,
                        )
                        if not _same(commanded, predicted):
                            raise ValueError("motion commanded action differs from prediction")
                        if not _same(requested, expected_axes):
                            raise ValueError(
                                "motion serial axes violate [X1,Y1,Z1,X2,Y2,Z2] mapping"
                            )
                        expected_effective = (
                            expected_axes
                            if final_gate_reason == "accepted"
                            else (0.0,) * 6
                        )
                        if not _same(effective, expected_effective):
                            raise ValueError(
                                "motion final gate did not produce the expected axes"
                            )
                    elif any(abs(value) > 1e-9 for value in effective):
                        raise ValueError("non-motion decision wrote nonzero effective axes")
            elif schema == COMMAND_SCHEMA:
                command_event_count += 1
                command_ns = _timestamp(
                    event.get("command_monotonic_ns"), "command_monotonic_ns"
                )
                command_seq = _timestamp(event.get("command_seq"), "command_seq")
                axes = _vector(event.get("serial_axes"), "serial_axes", 6)
                reason = event.get("reason")
                if not isinstance(reason, str) or not reason:
                    raise ValueError("command reason must be non-empty text")
                if event.get("serial_write_performed") is not True:
                    raise ValueError("command event must describe a performed serial write")
                serial_write_count += 1
                if any(abs(value) > 1e-9 for value in axes):
                    nonzero_serial_write_count += 1
                commands.append((command_seq, axes, reason, command_ns))
                if mode == "shadow":
                    failures.append(f"line {line_number}: shadow contains a command event")
            else:
                raise ValueError(f"unsupported schema: {schema!r}")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(f"line {line_number}: {exc}")

    if step_count == 0:
        failures.append("log contains no ACT Runtime steps")
    if any(
        current <= previous
        for previous, current in zip(state_stamps, state_stamps[1:])
    ):
        failures.append("ACT Runtime state timestamps are not strictly increasing")
    if mode == "motion":
        if not commands:
            failures.append("motion log contains no command events")
        else:
            _, first_axes, first_reason, _ = commands[0]
            if first_reason != "act_runtime_startup" or any(
                abs(value) > 1e-9 for value in first_axes
            ):
                failures.append("motion log does not start with startup zero")
            _, last_axes, last_reason, _ = commands[-1]
            if last_reason != "act_runtime_shutdown" or any(
                abs(value) > 1e-9 for value in last_axes
            ):
                failures.append("motion log does not end with shutdown zero")
            for previous, current in zip(commands, commands[1:]):
                if current[0] != (previous[0] + 1) & 0xFFFFFFFF:
                    failures.append("motion command sequence is not strictly continuous")
                    break
            commands_by_sequence = {
                sequence: (axes, reason)
                for sequence, axes, reason, _ in commands
            }
            for sequence, effective, final_gate_reason in step_writes:
                command = commands_by_sequence.get(sequence)
                if command is None:
                    failures.append("motion step references a missing command event")
                    continue
                command_axes, command_reason = command
                if not _same(command_axes, effective) or command_reason != final_gate_reason:
                    failures.append(
                        "motion command event differs from the recorded step result"
                    )
    estimated_rate_hz = 0.0
    if len(state_stamps) >= 2 and state_stamps[-1] > state_stamps[0]:
        estimated_rate_hz = (len(state_stamps) - 1) * 1_000_000_000.0 / (
            state_stamps[-1] - state_stamps[0]
        )
        if not min_step_rate_hz <= estimated_rate_hz <= max_step_rate_hz:
            failures.append(
                f"ACT Runtime step rate {estimated_rate_hz:.3f} Hz is outside "
                f"[{min_step_rate_hz:.3f}, {max_step_rate_hz:.3f}]"
            )
    elif step_count > 0:
        failures.append("ACT Runtime log has too few ordered steps to estimate step rate")
    return ActRuntimeLogReport(
        passed=not failures,
        mode=mode,
        step_count=step_count,
        command_event_count=command_event_count,
        serial_write_count=serial_write_count,
        nonzero_serial_write_count=nonzero_serial_write_count,
        dropped_state_count=dropped_state_count,
        estimated_step_rate_hz=estimated_rate_hz,
        max_state_to_decision_ms=max(state_to_decision_ms, default=0.0),
        max_camera_age_ms=max(camera_ages_ms, default=0.0),
        failure_reasons=tuple(failures),
    )

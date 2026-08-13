"""Read-only USART2 integrity probe for the authoritative STM32 telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Iterable

from .stm32_protocol import Stm32ProtocolError, Stm32TelemetryParser


@dataclass(frozen=True)
class Stm32LinkDiagnosticReport:
    passed: bool
    telemetry_frame_count: int
    parse_failure_count: int
    control_sequence_gap_count: int
    estimated_rate_hz: float
    failure_reasons: tuple[str, ...]


def inspect_stm32_telemetry_samples(
    samples: Iterable[tuple[int, bytes]],
) -> Stm32LinkDiagnosticReport:
    """Validate captured receive timestamps and bytes without tolerating corruption."""

    parser = Stm32TelemetryParser()
    receive_times_ns: list[int] = []
    previous_sequence: int | None = None
    parse_failures: list[str] = []
    sequence_gap_count = 0
    for receive_ns, raw_line in samples:
        try:
            frame = parser.parse_line(raw_line, receive_monotonic_ns=receive_ns)
        except Stm32ProtocolError as exc:
            parse_failures.append(str(exc))
            continue
        if frame is None:
            continue
        if previous_sequence is not None:
            expected = (previous_sequence + 1) & 0xFFFFFFFF
            if frame.control_seq != expected:
                sequence_gap_count += 1
        previous_sequence = frame.control_seq
        receive_times_ns.append(receive_ns)

    rate_hz = 0.0
    if len(receive_times_ns) >= 2:
        span_ns = receive_times_ns[-1] - receive_times_ns[0]
        if span_ns > 0:
            rate_hz = (len(receive_times_ns) - 1) * 1_000_000_000 / span_ns

    reasons = list(dict.fromkeys(parse_failures))
    if not receive_times_ns:
        reasons.append("no valid STM32 telemetry frames were received")
    if sequence_gap_count:
        reasons.append(f"control sequence gap count is {sequence_gap_count}")
    if not 18.0 <= rate_hz <= 22.0:
        reasons.append(f"STM32 telemetry rate {rate_hz:.3f} Hz is outside [18, 22]")
    return Stm32LinkDiagnosticReport(
        passed=not reasons,
        telemetry_frame_count=len(receive_times_ns),
        parse_failure_count=len(parse_failures),
        control_sequence_gap_count=sequence_gap_count,
        estimated_rate_hz=rate_hz,
        failure_reasons=tuple(reasons),
    )


def probe_stm32_telemetry(
    serial_port: Any,
    *,
    duration_s: float,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> Stm32LinkDiagnosticReport:
    """Read USART2 for a bounded interval. This function never writes to serial."""

    if duration_s <= 0:
        raise ValueError("STM32 diagnostic duration must be positive")
    started_ns = monotonic_ns()
    deadline_ns = started_ns + int(duration_s * 1_000_000_000)
    samples: list[tuple[int, bytes]] = []
    while monotonic_ns() < deadline_ns:
        raw_line = serial_port.readline()
        if raw_line:
            samples.append((monotonic_ns(), raw_line))
    return inspect_stm32_telemetry_samples(samples)


def run_stm32_link_diagnostic(
    config_path: str | Path, *, duration_s: float = 10.0
) -> Stm32LinkDiagnosticReport:
    """Open the configured USART2 exclusively and perform the read-only probe."""

    from .collector.config import load_collection_config

    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required for STM32 link diagnosis") from exc
    config = load_collection_config(config_path)
    with serial.Serial(
        config.serial.port,
        config.serial.baudrate,
        timeout=0.1,
        write_timeout=0.1,
        exclusive=True,
    ) as serial_port:
        return probe_stm32_telemetry(serial_port, duration_s=duration_s)

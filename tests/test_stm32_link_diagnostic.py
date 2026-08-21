from __future__ import annotations

import csv
import io

from excavator_il.stm32_link_diagnostic import (
    inspect_stm32_telemetry_samples,
    probe_stm32_telemetry,
)
from excavator_il.stm32_protocol import STM32_TELEMETRY_FIELDS


def _row(control_seq: int) -> bytes:
    values = {field: "0" for field in STM32_TELEMETRY_FIELDS}
    values.update(
        schema_version="stm32_control_telemetry.v2",
        control_seq=str(control_seq),
        sensor_seq=str(control_seq // 2),
        sensor_is_new=str(control_seq % 2),
        rs485_ok="1",
        dwj_ok="1",
        imu_ok="1",
    )
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerow(
        [values[field] for field in STM32_TELEMETRY_FIELDS]
    )
    return stream.getvalue().encode("ascii")


def test_stm32_link_diagnostic_accepts_clean_20_hz_v2_telemetry():
    samples = [
        (1_000_000_000 + index * 50_000_000, _row(index))
        for index in range(201)
    ]

    report = inspect_stm32_telemetry_samples(samples)

    assert report.passed is True
    assert report.telemetry_frame_count == 201
    assert report.estimated_rate_hz == 20.0
    assert report.estimated_control_rate_hz == 20.0
    assert report.max_receive_period_ms == 50.0
    assert report.parse_failure_count == 0
    assert report.control_sequence_gap_count == 0


def test_stm32_link_diagnostic_accepts_control_and_telemetry_phase_swaps():
    samples = [
        (1_000_000_000, _row(100)),
        (1_050_000_000, _row(100)),
        (1_100_000_000, _row(102)),
        (1_150_000_000, _row(103)),
    ]

    report = inspect_stm32_telemetry_samples(samples)

    assert report.passed is True
    assert report.control_sequence_gap_count == 0
    assert report.estimated_control_rate_hz == 20.0


def test_stm32_link_diagnostic_fails_closed_on_corruption_and_sequence_gap():
    samples = [
        (1_000_000_000, _row(1)),
        (1_050_000_000, b"\xff,not-ascii\n"),
        (1_100_000_000, _row(3)),
    ]

    report = inspect_stm32_telemetry_samples(samples)

    assert report.passed is False
    assert report.parse_failure_count == 1
    assert report.control_sequence_gap_count == 0
    assert any("ASCII" in reason for reason in report.failure_reasons)


def test_stm32_link_diagnostic_rejects_control_jump_larger_than_phase_swap():
    samples = [
        (1_000_000_000, _row(10)),
        (1_050_000_000, _row(14)),
        (1_100_000_000, _row(15)),
    ]

    report = inspect_stm32_telemetry_samples(samples)

    assert report.passed is False
    assert report.control_sequence_gap_count == 1
    assert any("control sequence discontinuity" in reason for reason in report.failure_reasons)


def test_stm32_link_diagnostic_rejects_missing_20_hz_wire_frame():
    samples = [
        (1_000_000_000 + index * 50_000_000, _row(index))
        for index in range(100)
    ]
    samples.extend(
        (6_000_000_000 + index * 50_000_000, _row(100 + index))
        for index in range(1, 101)
    )

    report = inspect_stm32_telemetry_samples(samples)

    assert 18.0 <= report.estimated_rate_hz <= 22.0
    assert report.max_receive_period_ms == 100.0
    assert report.passed is False
    assert any("receive period" in reason for reason in report.failure_reasons)


def test_stm32_link_probe_never_writes_to_serial():
    class ReadOnlySerial:
        def __init__(self):
            self.lines = iter((_row(1), _row(2)))
            self.write_count = 0

        def readline(self):
            return next(self.lines, b"")

        def write(self, _payload):
            self.write_count += 1
            raise AssertionError("diagnostic must never write serial data")

    serial_port = ReadOnlySerial()
    ticks = iter((0, 0, 50_000_000, 100_000_000, 150_000_000))

    probe_stm32_telemetry(
        serial_port,
        duration_s=0.1,
        monotonic_ns=lambda: next(ticks),
    )

    assert serial_port.write_count == 0

import json

from excavator_il.stm32_protocol import (
    STM32_TELEMETRY_FIELDS,
    Stm32ManualCommandEncoder,
    Stm32TelemetryFrame,
    Stm32TelemetryParser,
)


def test_manual_command_encoder_resumes_sequence_and_preserves_axis_contract():
    values = {field: 0 for field in STM32_TELEMETRY_FIELDS}
    values.update(
        schema_version="stm32_control_telemetry.v2",
        command_rx_seq=41,
        command_timed_out=1,
    )
    encoder = Stm32ManualCommandEncoder()
    encoder.synchronize(
        Stm32TelemetryFrame(receive_monotonic_ns=1, values=values)
    )

    payload = encoder.encode(
        axes=(-0.4, -0.2, 0.0, 0.3, 0.1, 0.0),
        monotonic_ns=1_234_567_890,
    )

    command = json.loads(payload)
    assert command["schema_version"] == "stm32_manual_command.v1"
    assert command["command_seq"] == 42
    assert command["command_source_stamp_ms"] == 1234
    assert [command[name] for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2")] == [
        -0.4,
        -0.2,
        0.0,
        0.3,
        0.1,
        0.0,
    ]


def test_v2_telemetry_parser_preserves_command_action_and_receive_time():
    parser = Stm32TelemetryParser()
    header = ",".join(STM32_TELEMETRY_FIELDS).encode("ascii")
    assert parser.parse_line(header, receive_monotonic_ns=900) is None

    values = {field: "0" for field in STM32_TELEMETRY_FIELDS}
    values.update(
        {
            "schema_version": "stm32_control_telemetry.v2",
            "control_seq": "42",
            "sensor_seq": "21",
            "sensor_is_new": "1",
            "command_action_boom": "-0.6",
            "command_action_stick": "0.4",
            "command_action_bucket": "0.2",
            "command_action_swing": "-0.8",
            "rs485_ok": "1",
            "dwj_ok": "1",
            "imu_ok": "1",
            "command_valid": "1",
            "control_enabled": "1",
        }
    )
    row = ",".join(values[field] for field in STM32_TELEMETRY_FIELDS).encode("ascii")

    frame = parser.parse_line(row, receive_monotonic_ns=1_234_567)

    assert frame is not None
    assert frame.receive_monotonic_ns == 1_234_567
    assert frame.control_seq == 42
    assert frame.sensor_seq == 21
    assert frame.sensor_is_new is True
    assert frame.command_action == (-0.6, 0.4, 0.2, -0.8)
    assert frame.sensor_valid is True


def test_v2_data_row_is_restart_safe_when_stm32_header_was_sent_before_collector_started():
    parser = Stm32TelemetryParser()
    values = {field: "0" for field in STM32_TELEMETRY_FIELDS}
    values["schema_version"] = "stm32_control_telemetry.v2"
    row = ",".join(values[field] for field in STM32_TELEMETRY_FIELDS).encode("ascii")

    frame = parser.parse_line(row, receive_monotonic_ns=123)

    assert frame is not None
    assert frame.receive_monotonic_ns == 123

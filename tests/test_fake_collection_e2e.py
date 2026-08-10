import io
import json

from PIL import Image

from excavator_il.collector.core import CollectorCore
from excavator_il.collector.recorder import EpisodeRecorder, EpisodeStart
from excavator_il.episode_builder import build_steps
from excavator_il.joystick_protocol import (
    ControllerIdentity,
    JoystickPacket,
    encode_joystick_packet,
)
from excavator_il.raw_episode import validate_episode
from excavator_il.stm32_protocol import STM32_TELEMETRY_FIELDS


def _jpeg_fixture():
    output = io.BytesIO()
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(output, format="JPEG")
    return output.getvalue()


def test_fake_20hz_control_20hz_telemetry_10hz_state_30hz_camera_pipeline(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    episode = recorder.start(
        EpisodeStart(
            "ExecuteDig",
            "operator_01",
            (0.8, 0.1, -0.2),
            "soil",
            {},
            {
                "device_id": "fixture",
                "width": 32,
                "height": 24,
                "nominal_fps": 30,
                "pixel_format": "RGB8",
                "timestamp_clock": "CLOCK_MONOTONIC",
            },
        ),
        start_wall_ns=1,
        start_monotonic_ns=1_000_000_000,
    )
    core = CollectorCore(
        recorder=recorder,
        expected_device_ids=("left", "right"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        deadzone=0.15,
    )
    start_ns = 1_000_000_000
    jpeg = _jpeg_fixture()
    for index in range(30):
        recorder.record_camera(
            encoded_image=jpeg,
            capture_monotonic_ns=start_ns + index * 33_333_333,
            extension="jpg",
        )

    for index in range(20):
        action_ns = start_ns + index * 50_000_000
        packet = JoystickPacket(
            "session",
            index,
            action_ns,
            action_ns,
            (0.2, -0.3, 0.0, 0.4, -0.5, 0.0),
            (
                ControllerIdentity(1, "left", "left", (True,)),
                ControllerIdentity(2, "right", "right", (False,)),
            ),
            True,
            "dual_stick.v1",
            "raw.v1",
        )
        decision = core.accept_joystick(
            encode_joystick_packet(packet),
            source_addr="192.168.0.220:40000",
            receive_monotonic_ns=action_ns,
            receive_wall_ns=action_ns,
        )
        core.record_command_result(
            decision, tx_monotonic_ns=action_ns, write_ok=True, write_error=""
        )

        telemetry = {field: "0" for field in STM32_TELEMETRY_FIELDS}
        telemetry.update(
            {
                "schema_version": "stm32_control_telemetry.v2",
                "control_seq": str(index),
                "sensor_seq": str(index // 2),
                "sensor_stamp_ms": str(index * 50),
                "sensor_is_new": "1" if index % 2 == 0 else "0",
                "boom_pos_mm": "150",
                "stick_pos_mm": "140",
                "bucket_pos_mm": "130",
                "control_mode": "1",
                "rs485_ok": "1",
                "dwj_ok": "1",
                "imu_ok": "1",
            }
        )
        row = ",".join(telemetry[field] for field in STM32_TELEMETRY_FIELDS)
        core.accept_stm32(
            row.encode("ascii"),
            receive_monotonic_ns=action_ns + 10_000_000,
            receive_wall_ns=action_ns + 10_000_000,
        )

    recorder.stop(
        success=True,
        failure_reason="",
        intervention=False,
        end_wall_ns=2,
        end_monotonic_ns=2_000_000_000,
    )
    report = build_steps(episode)
    validation = validate_episode(episode)

    assert report.training_step_count == 10
    assert round(report.stream_timing["stm32_telemetry"]["estimated_rate_hz"]) == 20
    assert round(report.stream_timing["new_sensor_state"]["estimated_rate_hz"]) == 10
    assert round(report.stream_timing["camera_front"]["estimated_rate_hz"]) == 30
    assert validation.step_count == 10

import json

from excavator_il.collector.core import CollectorCore
from excavator_il.collector.recorder import EpisodeRecorder, EpisodeStart
from excavator_il.joystick_protocol import (
    ControllerIdentity,
    JoystickPacket,
    encode_joystick_packet,
)
from excavator_il.stm32_protocol import STM32_TELEMETRY_FIELDS


def test_collector_accepts_new_session_action_and_rejects_duplicate_sequence(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    episode = recorder.start(
        EpisodeStart(
            task="ExecuteDig",
            operator_id="operator_01",
            dig_target_m=(0.8, 0.1, -0.2),
            material_id="dry_soil_01",
            provenance={},
            camera_front={"device_id": "/dev/video0"},
        ),
        start_wall_ns=1,
        start_monotonic_ns=2,
    )
    core = CollectorCore(
        recorder=recorder,
        expected_device_ids=("left-guid", "right-guid"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        deadzone=0.15,
    )
    packet = JoystickPacket(
        session_id="session-a",
        sample_seq=0,
        pc_sample_monotonic_ns=100,
        pc_sample_wall_ns=200,
        axes=(-0.8, 0.4, 0.0, 0.2, -0.6, 0.0),
        controllers=(
            ControllerIdentity(1, "left-guid", "left", (True,)),
            ControllerIdentity(2, "right-guid", "right", (False,)),
        ),
        deadman_pressed=True,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )
    datagram = encode_joystick_packet(packet)

    accepted = core.accept_joystick(
        datagram,
        source_addr="192.168.0.220:40000",
        receive_monotonic_ns=1_000_000_000,
        receive_wall_ns=2_000_000_000,
    )
    duplicate = core.accept_joystick(
        datagram,
        source_addr="192.168.0.220:40000",
        receive_monotonic_ns=1_050_000_000,
        receive_wall_ns=2_050_000_000,
    )

    assert accepted.accepted is True
    serial_command = json.loads(accepted.serial_payload.decode("ascii"))
    assert [serial_command[name] for name in ("Y2", "Y1", "X2", "X1")] == [
        -0.6,
        0.4,
        0.2,
        -0.8,
    ]
    assert serial_command["command_seq"] == 0
    assert duplicate.accepted is False
    assert duplicate.reason == "duplicate_or_out_of_order"
    assert duplicate.serial_payload is None

    recorder.stop(
        success=True,
        failure_reason="",
        intervention=False,
        end_wall_ns=3,
        end_monotonic_ns=4,
    )
    action = json.loads((episode / "expert_action.jsonl").read_text(encoding="utf-8"))
    assert [action[name] for name in (
        "action_boom", "action_stick", "action_bucket", "action_swing"
    )] == [-0.6, 0.4, 0.2, -0.8]


def test_collector_records_restart_safe_stm32_raw_and_control_rows(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    episode = recorder.start(
        EpisodeStart(
            task="ExecuteDig",
            operator_id="operator_01",
            dig_target_m=(0.8, 0.1, -0.2),
            material_id="dry_soil_01",
            provenance={},
            camera_front={"device_id": "/dev/video0"},
        ),
        start_wall_ns=1,
        start_monotonic_ns=2,
    )
    core = CollectorCore(
        recorder=recorder,
        expected_device_ids=("left-guid", "right-guid"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        deadzone=0.15,
    )
    values = {field: "0" for field in STM32_TELEMETRY_FIELDS}
    values["schema_version"] = "stm32_control_telemetry.v2"
    values["control_seq"] = "12"
    row = ",".join(values[field] for field in STM32_TELEMETRY_FIELDS).encode("ascii")

    frame = core.accept_stm32(
        row, receive_monotonic_ns=1_000, receive_wall_ns=2_000
    )

    assert frame is not None
    assert frame.control_seq == 12
    recorder.stop(
        success=False,
        failure_reason="fixture",
        intervention=False,
        end_wall_ns=3,
        end_monotonic_ns=4,
    )
    raw = json.loads((episode / "stm32_raw.jsonl").read_text(encoding="utf-8"))
    assert raw["parse_ok"] is True
    control_lines = (episode / "control.csv").read_text(encoding="utf-8").splitlines()
    assert control_lines[0].startswith("episode_id,raw_frame_seq,orin_receive_monotonic_ns")
    assert "stm32_control_telemetry.v2" in control_lines[1]


def test_collector_builds_explicit_safe_zero_after_joystick_timeout(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    core = CollectorCore(
        recorder=recorder,
        expected_device_ids=("left-guid", "right-guid"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        deadzone=0.15,
    )

    zero = core.make_safe_zero(monotonic_ns=900_000_000, reason="joystick_timeout")

    command = json.loads(zero.serial_payload.decode("ascii"))
    assert [command[name] for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2")] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert zero.command_kind == "safe_zero:joystick_timeout"

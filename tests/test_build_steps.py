import csv
import json

from PIL import Image

from excavator_il.episode_builder import build_steps
from excavator_il.training_segments import locate_joystick_timeout_events
from excavator_il.stm32_protocol import STM32_TELEMETRY_FIELDS


def test_build_steps_uses_only_new_state_and_latest_causal_action_and_image(tmp_path):
    episode = tmp_path / "episode_0001"
    camera = episode / "camera_front"
    camera.mkdir(parents=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(camera / "000000.jpg")
    (episode / "camera_front_timestamps.csv").write_text(
        "camera_frame_index,camera_stamp_monotonic_ns,image_path\n"
        "0,950000000,camera_front/000000.jpg\n",
        encoding="utf-8",
    )

    telemetry = {field: 0 for field in STM32_TELEMETRY_FIELDS}
    telemetry.update(
        {
            "schema_version": "stm32_control_telemetry.v2",
            "sensor_seq": 5,
            "sensor_stamp_ms": 5000,
            "sensor_is_new": 1,
            "boom_pos_mm": 150.0,
            "stick_pos_mm": 140.0,
            "bucket_pos_mm": 130.0,
            "boom_vel_mmps": 10.0,
            "stick_vel_mmps": -20.0,
            "bucket_vel_mmps": 30.0,
            "boom_angle_deg": 30.0,
            "arm_angle_deg": 60.0,
            "bucket_angle_deg": 90.0,
            "swing_angle_deg": 180.0,
            "swing_vel_degps": -10.0,
            "pump_percent": -30.0,
            "control_mode": 1,
            "command_valid": 1,
            "control_enabled": 1,
            "rs485_ok": 1,
            "dwj_ok": 1,
            "imu_ok": 1,
        }
    )
    stm32_record = {
        "episode_id": "episode_0001",
        "raw_frame_seq": 9,
        "orin_receive_monotonic_ns": 1_000_000_000,
        "parse_ok": True,
        "parse_error": "",
        "raw_payload": "fixture",
        "telemetry": telemetry,
    }
    (episode / "stm32_raw.jsonl").write_text(
        json.dumps(stm32_record) + "\n", encoding="utf-8"
    )
    actions = [
        {
            "episode_id": "episode_0001",
            "action_seq": 1,
            "source_joystick_sample_seq": 10,
            "action_stamp_monotonic_ns": 900_000_000,
            "action_boom": 0.1,
            "action_stick": 0.2,
            "action_bucket": 0.3,
            "action_swing": 0.4,
            "action_valid": True,
            "mapping_id": "dual_stick.v1",
            "calibration_id": "raw.v1",
        },
        {
            "episode_id": "episode_0001",
            "action_seq": 2,
            "source_joystick_sample_seq": 11,
            "action_stamp_monotonic_ns": 980_000_000,
            "action_boom": -0.6,
            "action_stick": 0.4,
            "action_bucket": 0.2,
            "action_swing": -0.8,
            "action_valid": True,
            "mapping_id": "dual_stick.v1",
            "calibration_id": "raw.v1",
        },
        {
            "episode_id": "episode_0001",
            "action_seq": 3,
            "source_joystick_sample_seq": 12,
            "action_stamp_monotonic_ns": 1_010_000_000,
            "action_boom": 1.0,
            "action_stick": 1.0,
            "action_bucket": 1.0,
            "action_swing": 1.0,
            "action_valid": True,
            "mapping_id": "dual_stick.v1",
            "calibration_id": "raw.v1",
        },
    ]
    (episode / "expert_action.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in actions), encoding="utf-8"
    )
    (episode / "command_tx.jsonl").write_text(
        json.dumps(
            {
                "command_seq": 7,
                "command_kind": "safe_zero:joystick_timeout",
                "write_ok": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_steps(episode)

    with (episode / "steps.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["state_seq"] == "5"
    assert float(rows[0]["boom_pos_m"]) == 0.15
    assert float(rows[0]["boom_angle_rad"]) > 0.52
    assert [float(rows[0][field]) for field in (
        "action_boom", "action_stick", "action_bucket", "action_swing"
    )] == [-0.6, 0.4, 0.2, -0.8]
    assert report.training_step_count == 1
    assert report.rejected_state_count == 0
    assert report.max_action_age_ms == 20.0
    assert report.max_camera_age_ms == 50.0
    assert report.serial_parse_failure_count == 0
    assert report.joystick_timeout_count == 1
    assert report.action_age_ms["p95"] == 20.0
    assert report.camera_age_ms["p95"] == 50.0
    quality = json.loads((episode / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["training_step_count"] == 1
    assert quality["camera_queue_drop_count"] == 0
    assert quality["joystick_timeout_count"] == 1


def test_build_steps_uses_episode_metadata_when_preroll_telemetry_has_null_id(
    tmp_path,
):
    episode = tmp_path / "episode_0009"
    camera = episode / "camera_front"
    camera.mkdir(parents=True)
    (episode / "episode.json").write_text(
        json.dumps({"episode_id": "episode_0009"}) + "\n", encoding="utf-8"
    )
    Image.new("RGB", (32, 24)).save(camera / "000000.jpg")
    (episode / "camera_front_timestamps.csv").write_text(
        "camera_frame_index,camera_stamp_monotonic_ns,image_path\n"
        "0,950000000,camera_front/000000.jpg\n",
        encoding="utf-8",
    )
    telemetry = {field: 0 for field in STM32_TELEMETRY_FIELDS}
    telemetry.update(
        {
            "schema_version": "stm32_control_telemetry.v2",
            "sensor_seq": 1,
            "sensor_stamp_ms": 1000,
            "sensor_is_new": 1,
            "control_mode": 1,
            "rs485_ok": 1,
            "dwj_ok": 1,
            "imu_ok": 1,
        }
    )
    (episode / "stm32_raw.jsonl").write_text(
        json.dumps(
            {
                "episode_id": None,
                "raw_frame_seq": 0,
                "orin_receive_monotonic_ns": 1_000_000_000,
                "parse_ok": True,
                "telemetry": telemetry,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (episode / "expert_action.jsonl").write_text(
        json.dumps(
            {
                "action_stamp_monotonic_ns": 980_000_000,
                "action_boom": 0.0,
                "action_stick": 0.0,
                "action_bucket": 0.0,
                "action_swing": 0.0,
                "action_valid": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_steps(episode)

    assert report.episode_id == "episode_0009"
    with (episode / "steps.csv").open(newline="", encoding="utf-8") as stream:
        assert next(csv.DictReader(stream))["episode_id"] == "episode_0009"
    segments = json.loads(
        (episode / "training_segments.json").read_text(encoding="utf-8")
    )
    assert segments["parent_episode_id"] == "episode_0009"


def test_build_steps_splits_around_recovered_joystick_timeout(tmp_path):
    episode = tmp_path / "episode_0005"
    camera = episode / "camera_front"
    camera.mkdir(parents=True)

    telemetry_template = {field: 0 for field in STM32_TELEMETRY_FIELDS}
    telemetry_template.update(
        {
            "schema_version": "stm32_control_telemetry.v2",
            "sensor_is_new": 1,
            "control_mode": 1,
            "command_valid": 1,
            "control_enabled": 1,
            "rs485_ok": 1,
            "dwj_ok": 1,
            "imu_ok": 1,
        }
    )
    stm32_records = []
    actions = []
    camera_rows = []
    for index in range(6):
        state_ns = 1_000_000_000 + index * 100_000_000
        telemetry = {
            **telemetry_template,
            "sensor_seq": 10 + index,
            "sensor_stamp_ms": 5_000 + index * 100,
        }
        stm32_records.append(
            {
                "episode_id": "episode_0005",
                "raw_frame_seq": index,
                "orin_receive_monotonic_ns": state_ns,
                "parse_ok": True,
                "parse_error": "",
                "raw_payload": "fixture",
                "telemetry": telemetry,
            }
        )
        actions.append(
            {
                "episode_id": "episode_0005",
                "action_seq": index,
                "source_joystick_sample_seq": index,
                "action_stamp_monotonic_ns": state_ns - 10_000_000,
                "action_boom": 0.1,
                "action_stick": 0.2,
                "action_bucket": 0.3,
                "action_swing": 0.4,
                "action_valid": True,
                "mapping_id": "dual_stick.v1",
                "calibration_id": "raw.v1",
            }
        )
        image_name = f"{index:06d}.jpg"
        Image.new("RGB", (32, 24), color=(index, index, index)).save(
            camera / image_name
        )
        camera_rows.append(
            f"{index},{state_ns - 5_000_000},camera_front/{image_name}\n"
        )

    (episode / "stm32_raw.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in stm32_records),
        encoding="utf-8",
    )
    (episode / "expert_action.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in actions),
        encoding="utf-8",
    )
    (episode / "camera_front_timestamps.csv").write_text(
        "camera_frame_index,camera_stamp_monotonic_ns,image_path\n"
        + "".join(camera_rows),
        encoding="utf-8",
    )
    (episode / "command_tx.jsonl").write_text(
        json.dumps(
            {
                "command_seq": 50,
                "command_tx_monotonic_ns": 1_250_000_000,
                "command_kind": "safe_zero:joystick_timeout",
                "write_ok": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    joystick_records = [
        {
            "joystick_sample_seq": index,
            "orin_receive_monotonic_ns": 1_260_000_000 + index * 10_000_000,
            "parse_ok": True,
            "parse_error": "",
        }
        for index in range(10)
    ]
    (episode / "joystick_raw.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in joystick_records),
        encoding="utf-8",
    )

    report = build_steps(episode)

    segments = json.loads(
        (episode / "training_segments.json").read_text(encoding="utf-8")
    )
    assert report.training_step_count == 5
    assert report.training_segment_count == 2
    assert report.excluded_training_step_count == 1
    assert report.rejection_reasons["safety_event_quarantine"] == 1
    assert [segment["step_count"] for segment in segments["segments"]] == [3, 2]
    assert segments["segments"][0]["end_frame_index_exclusive"] == 3
    assert segments["segments"][1]["start_frame_index"] == 3
    assert segments["fault_events"][0]["recovered"] is True


def test_timeout_recovery_requires_successful_safe_zero_write():
    events = locate_joystick_timeout_events(
        [
            {
                "command_kind": "safe_zero:joystick_timeout",
                "command_tx_monotonic_ns": 1_000,
                "write_ok": False,
            }
        ],
        [
            {
                "joystick_sample_seq": sequence,
                "orin_receive_monotonic_ns": 1_001 + sequence,
                "parse_ok": True,
            }
            for sequence in range(10)
        ],
    )

    assert len(events) == 1
    assert events[0].event_stamp_monotonic_ns == 1_000
    assert events[0].recovery_stamp_monotonic_ns is None
    assert events[0].recovered is False

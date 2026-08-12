import csv
import json

import pytest

import excavator_il.guided_episode as guided_episode
from excavator_il.guided_episode import SystemGuidedEpisodeOperations
from excavator_il.stm32_protocol import STM32_TELEMETRY_FIELDS
from excavator_il.zero_soak import inspect_zero_command_episode, run_zero_command_soak


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _safe_episode(tmp_path):
    episode = tmp_path / "episode_0001"
    episode.mkdir()
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "episode_id": "episode_0001",
                "status": "aborted",
                "failure_reason": "zero_command_soak_complete",
            }
        ),
        encoding="utf-8",
    )
    telemetry_records = []
    for index in range(21):
        telemetry = {field: 0 for field in STM32_TELEMETRY_FIELDS}
        telemetry.update(
            {
                "schema_version": "stm32_control_telemetry.v2",
                "control_seq": index,
                "sensor_seq": index // 2,
                "sensor_is_new": int(index % 2 == 0),
                "control_mode": 1,
                "command_action_boom": 0.0,
                "command_action_stick": 0.0,
                "command_action_bucket": 0.0,
                "command_action_swing": 0.0,
                "rs485_ok": 1,
                "dwj_ok": 1,
                "imu_ok": 1,
            }
        )
        telemetry_records.append(
            {
                "raw_frame_seq": index,
                "orin_receive_monotonic_ns": 1_000_000_000 + index * 50_000_000,
                "parse_ok": True,
                "telemetry": telemetry,
            }
        )
    _write_jsonl(episode / "stm32_raw.jsonl", telemetry_records)
    _write_jsonl(
        episode / "joystick_raw.jsonl",
        [
            {
                "orin_receive_monotonic_ns": 1_000_000_000 + index * 50_000_000,
                "parse_ok": True,
                "joystick_sample_seq": index,
            }
            for index in range(21)
        ],
    )
    _write_jsonl(
        episode / "expert_action.jsonl",
        [
            {
                "action_stamp_monotonic_ns": 1_000_000_000 + index * 50_000_000,
                "action_valid": False,
            }
            for index in range(21)
        ],
    )
    zero_payload = json.dumps(
        {
            "X1": 0.0,
            "Y1": 0.0,
            "Z1": 0.0,
            "X2": 0.0,
            "Y2": 0.0,
            "Z2": 0.0,
        }
    )
    _write_jsonl(
        episode / "command_tx.jsonl",
        [
            {
                "command_seq": index,
                "command_kind": "safe_zero:deadman_released",
                "raw_serial_payload": zero_payload,
                "write_ok": True,
            }
            for index in range(21)
        ],
    )
    with (episode / "camera_front_timestamps.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "camera_frame_index",
                "camera_stamp_monotonic_ns",
                "image_path",
            ),
        )
        writer.writeheader()
        for index in range(31):
            writer.writerow(
                {
                    "camera_frame_index": index,
                    "camera_stamp_monotonic_ns": 1_000_000_000
                    + index * 33_333_333,
                    "image_path": f"camera_front/{index:06d}.jpg",
                }
            )
    return episode


def test_zero_soak_accepts_safe_zero_commands_and_nominal_stream_rates(tmp_path):
    episode = _safe_episode(tmp_path)

    report = inspect_zero_command_episode(episode)

    assert report.passed is True
    assert report.nonzero_command_count == 0
    assert report.valid_action_count == 0
    assert report.joystick_timeout_count == 0
    assert report.stream_rates_hz == {
        "stm32_telemetry": 20.0,
        "new_sensor_state": 10.0,
        "expert_action": 20.0,
        "camera_front": 30.000000300000004,
    }


def test_zero_soak_automates_safe_lifecycle_before_inspection():
    class Operations:
        def __init__(self):
            self.events = []

        def preflight(self):
            self.events.append("preflight")

        def start_collector(self):
            self.events.append("start_collector")

        def start_episode(self):
            self.events.append("start_episode")
            return "/data/episode_0001"

        def start_teleop(self):
            self.events.append("start_teleop")

        def wait_for_ack(self, timeout_s):
            self.events.append(("wait_for_ack", timeout_s))

        def monitor_deadman_released(self, duration_s):
            self.events.append(("monitor_deadman_released", duration_s))

        def abort_episode(self, reason):
            self.events.append(("abort_episode", reason))
            return "/data/episode_0001"

        def stop_teleop(self):
            self.events.append("stop_teleop")

        def stop_collector(self):
            self.events.append("stop_collector")

        def inspect_zero_soak(self, path):
            self.events.append(("inspect_zero_soak", path))
            return {"passed": True, "episode_id": "episode_0001"}

    operations = Operations()

    report = run_zero_command_soak(
        operations,
        duration_s=30,
        ack_timeout_s=8,
    )

    assert report["passed"] is True
    assert operations.events == [
        "preflight",
        "start_collector",
        "start_episode",
        "start_teleop",
        ("wait_for_ack", 8),
        ("monitor_deadman_released", 30),
        ("abort_episode", "zero_command_soak_complete"),
        "stop_teleop",
        "stop_collector",
        ("inspect_zero_soak", "/data/episode_0001"),
    ]


def test_zero_soak_monitor_fails_closed_if_deadman_is_pressed(tmp_path, monkeypatch):
    clock = [0.0]
    lines = iter(
        (
            "teleop seq=1 ack=1 ack_lag=0 accepted_acks=1 rejected_acks=0 "
            "deadman=False axes=(0,0,0,0,0,0)",
            "teleop seq=2 ack=2 ack_lag=0 accepted_acks=2 rejected_acks=0 "
            "deadman=True axes=(0,0,0,0,0,0)",
        )
    )

    class TeleopLines:
        def wait_for(self, predicate, timeout_s, after_index=-1):
            line = next(lines)
            clock[0] += 0.05
            assert predicate(line)
            return after_index + 1, line

    config = type("Config", (), {"log_dir": tmp_path})()
    operations = SystemGuidedEpisodeOperations(config)
    operations._teleop = TeleopLines()
    monkeypatch.setattr(guided_episode.time, "monotonic", lambda: clock[0])

    with pytest.raises(RuntimeError, match="deadman was pressed"):
        operations.monitor_deadman_released(30)


def test_zero_soak_rejects_invalid_joystick_packet(tmp_path):
    episode = _safe_episode(tmp_path)
    joystick_path = episode / "joystick_raw.jsonl"
    records = [json.loads(line) for line in joystick_path.read_text().splitlines()]
    records[3].update(
        parse_ok=False,
        parse_error="mapping_id does not match collector configuration",
        joystick_sample_seq=None,
    )
    _write_jsonl(joystick_path, records)

    report = inspect_zero_command_episode(episode)

    assert report.passed is False
    assert report.joystick_parse_failure_count == 1
    assert "joystick parse failure count is 1" in report.failure_reasons

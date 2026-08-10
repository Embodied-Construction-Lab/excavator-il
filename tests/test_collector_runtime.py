import json

from excavator_il.collector.config import (
    CameraConfig,
    CollectionConfig,
    ControllerConfig,
    EpisodeDefaults,
    JoystickUdpConfig,
    SerialConfig,
)
from excavator_il.collector.core import CollectorCore
from excavator_il.collector.recorder import EpisodeRecorder, EpisodeStart
from excavator_il.collector.runtime import CollectorRuntime
from excavator_il.joystick_protocol import (
    ControllerIdentity,
    JoystickPacket,
    encode_joystick_packet,
)


class _Serial:
    def __init__(self):
        self.writes = []

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def flush(self):
        pass


class _Frame:
    capture_monotonic_ns = 999
    encoded_image = b"jpeg"
    extension = "jpg"


class _Camera:
    def read_encoded(self):
        return _Frame()


def _packet(sample_seq=0):
    return encode_joystick_packet(
        JoystickPacket(
            session_id="session-a",
            sample_seq=sample_seq,
            pc_sample_monotonic_ns=1,
            pc_sample_wall_ns=2,
            axes=(-0.8, 0.4, 0.0, 0.2, -0.6, 0.0),
            controllers=(
                ControllerIdentity(1, "left-guid", "left", (True,)),
                ControllerIdentity(2, "right-guid", "right", (False,)),
            ),
            deadman_pressed=True,
            mapping_id="dual_stick.v1",
            calibration_id="raw.v1",
        )
    )


def test_runtime_forwards_packet_records_real_write_and_sends_one_timeout_zero(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    episode = recorder.start(
        EpisodeStart(
            "ExecuteDig",
            "operator_01",
            (0.8, 0.1, -0.2),
            "soil",
            {},
            {"device_id": "/dev/video0"},
        ),
        start_wall_ns=1,
        start_monotonic_ns=2,
    )
    serial = _Serial()
    core = CollectorCore(
        recorder=recorder,
        expected_device_ids=("left-guid", "right-guid"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        deadzone=0.15,
    )
    runtime = CollectorRuntime(
        core=core,
        recorder=recorder,
        serial_port=serial,
        camera=_Camera(),
        allowed_pc_host="192.168.0.220",
        joystick_timeout_ms=150,
    )

    ack = runtime.handle_joystick(
        _packet(),
        source=("192.168.0.220", 40000),
        receive_monotonic_ns=1_000_000_000,
        receive_wall_ns=2_000_000_000,
    )
    first_timeout = runtime.enforce_joystick_timeout(1_151_000_000)
    repeated_timeout = runtime.enforce_joystick_timeout(1_300_000_000)
    image_path = runtime.capture_once()

    assert json.loads(ack)["accepted"] is True
    assert first_timeout is True
    assert repeated_timeout is False
    assert len(serial.writes) == 2
    assert all(json.loads(payload)["X1"] == expected for payload, expected in zip(serial.writes, (-0.8, 0.0)))
    assert image_path == "camera_front/000000.jpg"

    recorder.stop(
        success=True,
        failure_reason="",
        intervention=False,
        end_wall_ns=3,
        end_monotonic_ns=4,
    )
    records = [
        json.loads(line)
        for line in (episode / "command_tx.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["write_ok"] for record in records] == [True, True]
    assert records[1]["command_kind"] == "safe_zero:joystick_timeout"


def test_runtime_rejects_unexpected_pc_without_touching_serial(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    serial = _Serial()
    core = CollectorCore(
        recorder=recorder,
        expected_device_ids=("left-guid", "right-guid"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        deadzone=0.15,
    )
    runtime = CollectorRuntime(
        core=core,
        recorder=recorder,
        serial_port=serial,
        camera=_Camera(),
        allowed_pc_host="192.168.0.220",
        joystick_timeout_ms=150,
    )

    ack = runtime.handle_joystick(
        _packet(),
        source=("192.168.0.99", 40000),
        receive_monotonic_ns=1,
        receive_wall_ns=2,
    )

    assert json.loads(ack)["reason"] == "source_not_allowed"
    assert serial.writes == []

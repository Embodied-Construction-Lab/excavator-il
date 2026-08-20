import json
import math
import socket
import threading
import time
import urllib.request

import pytest

from excavator_il.collector.client import send_episode_command
from excavator_il.collector.config import (
    CameraConfig,
    CameraPreviewConfig,
    CollectionConfig,
    ControllerConfig,
    EpisodeDefaults,
    JoystickUdpConfig,
    MachineStateUdpConfig,
    SerialConfig,
)
from excavator_il.collector.service import CollectorService
from excavator_il.joystick_protocol import (
    ControllerIdentity,
    JoystickPacket,
    encode_joystick_packet,
)
from excavator_il.stm32_protocol import STM32_TELEMETRY_FIELDS


class _Serial:
    def __init__(self, lines):
        self.writes = []
        self.lines = list(lines)

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def flush(self):
        pass

    def readline(self):
        if self.lines:
            return self.lines.pop(0)
        time.sleep(0.005)
        return b""


class _Frame:
    extension = "jpg"
    encoded_image = b"jpeg"

    @property
    def capture_monotonic_ns(self):
        return time.monotonic_ns()


class _Camera:
    def read_encoded(self):
        time.sleep(0.005)
        return _Frame()


def _free_udp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _free_tcp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_collector_service_connects_udp_episode_socket_camera_and_safe_shutdown(tmp_path):
    port = _free_udp_port()
    preview_port = _free_tcp_port()
    control_socket = tmp_path / "collector.sock"
    config = CollectionConfig(
        data_root=tmp_path / "raw",
        joystick=JoystickUdpConfig("127.0.0.1", port, "127.0.0.1", 150),
        controllers=ControllerConfig(
            ("left-guid", "right-guid"), "dual_stick.v1", "raw.v1", 0.15
        ),
        serial=SerialConfig("fixture", 460800),
        camera=CameraConfig("fixture", 32, 24, 30, 95),
        episode_control_socket=control_socket,
        episode_defaults=EpisodeDefaults((0.8, 0.1, -0.2), "soil", {}),
        camera_preview=CameraPreviewConfig("127.0.0.1", preview_port),
    )
    telemetry = {field: "0" for field in STM32_TELEMETRY_FIELDS}
    telemetry.update(
        {
            "schema_version": "stm32_control_telemetry.v2",
            "command_rx_seq": "197",
            "command_timed_out": "1",
        }
    )
    telemetry_row = ",".join(
        telemetry[field] for field in STM32_TELEMETRY_FIELDS
    ).encode("ascii")
    serial = _Serial([telemetry_row])
    service = CollectorService(config, serial_port=serial, camera=_Camera())
    thread = threading.Thread(target=service.run)
    thread.start()
    deadline = time.monotonic() + 2.0
    while not control_socket.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert control_socket.exists()
    with urllib.request.urlopen(
        f"http://127.0.0.1:{preview_port}/camera/front.jpg", timeout=1.0
    ) as response:
        assert response.read() == b"jpeg"

    started = send_episode_command(
        control_socket,
        {"command": "start", "task": "ExecuteDig", "operator_id": "operator_01"},
    )
    packet = JoystickPacket(
        session_id="session-a",
        sample_seq=0,
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
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.settimeout(1.0)
    udp.sendto(encode_joystick_packet(packet), ("127.0.0.1", port))
    ack, _ = udp.recvfrom(2048)
    udp.close()
    stopped = send_episode_command(control_socket, {"command": "stop", "success": True})

    service.request_stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert json.loads(ack)["accepted"] is True
    assert started["episode_id"] == "episode_0001"
    assert stopped["status"] == "complete"
    assert len(serial.writes) >= 3  # startup zero, manual sample, shutdown zero
    startup_zero = json.loads(serial.writes[0].decode("ascii"))
    assert startup_zero["command_seq"] == 198
    episode = tmp_path / "raw" / "episode_0001"
    assert (episode / "joystick_raw.jsonl").stat().st_size > 0
    assert (episode / "camera_front_timestamps.csv").stat().st_size > 0


def test_collector_service_forwards_stm32_state_to_airylidar_udp(tmp_path):
    joystick_port = _free_udp_port()
    state_port = _free_udp_port()
    state_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_receiver.bind(("127.0.0.1", state_port))
    state_receiver.settimeout(2.0)
    config = CollectionConfig(
        data_root=tmp_path / "raw",
        joystick=JoystickUdpConfig(
            "127.0.0.1", joystick_port, "127.0.0.1", 150
        ),
        controllers=ControllerConfig(
            ("left-guid", "right-guid"), "dual_stick.v1", "raw.v1", 0.15
        ),
        serial=SerialConfig("fixture", 460800),
        camera=CameraConfig("fixture", 32, 24, 30, 95),
        episode_control_socket=tmp_path / "collector.sock",
        episode_defaults=EpisodeDefaults((0.8, 0.1, -0.2), "soil", {}),
        machine_state_udp=MachineStateUdpConfig(
            "127.0.0.1", state_port, "scale_excavator_v1"
        ),
    )
    telemetry = {field: "0" for field in STM32_TELEMETRY_FIELDS}
    telemetry.update(
        schema_version="stm32_control_telemetry.v2",
        control_stamp_ms="700",
        sensor_seq="10",
        sensor_stamp_ms="650",
        sensor_is_new="1",
        command_rx_seq="21",
        command_timed_out="1",
        boom_pos_mm="160.0",
        stick_pos_mm="170.0",
        bucket_pos_mm="180.0",
        boom_vel_mmps="1.5",
        stick_vel_mmps="-2.5",
        bucket_vel_mmps="3.5",
        boom_angle_deg="30.0",
        arm_angle_deg="40.0",
        bucket_angle_deg="50.0",
        swing_angle_deg="-10.0",
        swing_vel_degps="2.0",
        control_enabled="1",
        rs485_ok="1",
        dwj_ok="1",
        imu_ok="1",
    )
    row = ",".join(
        telemetry[field] for field in STM32_TELEMETRY_FIELDS
    ).encode("ascii")
    serial = _Serial([row, row])
    service = CollectorService(config, serial_port=serial, camera=_Camera())
    thread = threading.Thread(target=service.run)
    thread.start()
    try:
        payload, source = state_receiver.recvfrom(4096)
    finally:
        service.request_stop()
        thread.join(timeout=2.0)
        state_receiver.close()

    packet = json.loads(payload)
    assert source[0] == "127.0.0.1"
    assert packet["type"] == "machine_state_v1"
    assert packet["schema_version"] == "1.0"
    assert packet["seq"] == 0
    assert packet["stm32_stamp_ms"] == 700
    assert packet["machine_id"] == "scale_excavator_v1"
    assert packet["safety"] == {
        "estop": False,
        "stm32_alive": True,
        "sensor_valid": True,
        "control_enabled": True,
        "fault_flags": [],
    }
    assert packet["actuator_state"]["boom"] == {
        "position_m": 0.16,
        "velocity_mps": 0.0015,
    }
    assert packet["actuator_state"]["stick"] == {
        "position_m": 0.17,
        "velocity_mps": -0.0025,
    }
    assert packet["actuator_state"]["bucket"] == {
        "position_m": 0.18,
        "velocity_mps": 0.0035,
    }
    assert packet["joint_state"]["position_rad"]["boom"] == pytest.approx(
        math.radians(30.0)
    )
    assert packet["joint_state"]["position_rad"]["arm"] == pytest.approx(
        math.radians(40.0)
    )
    assert packet["joint_state"]["position_rad"]["bucket"] == pytest.approx(
        math.radians(50.0)
    )
    assert packet["joint_state"]["position_rad"]["swing"] == pytest.approx(
        math.radians(-10.0)
    )
    assert not thread.is_alive()


def test_episode_client_surfaces_collector_rejection(tmp_path):
    socket_path = tmp_path / "reject.sock"
    ready = threading.Event()

    def serve():
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(1)
        ready.set()
        connection, _ = server.accept()
        with connection:
            connection.recv(4096)
            connection.sendall(b'{"ok":false,"error":"fixture rejection"}\n')
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(1.0)
    try:
        send_episode_command(socket_path, {"command": "status"})
    except RuntimeError as exc:
        assert "fixture rejection" in str(exc)
    else:
        raise AssertionError("collector rejection must raise RuntimeError")
    thread.join(timeout=1.0)

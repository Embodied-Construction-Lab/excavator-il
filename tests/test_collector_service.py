import json
import socket
import threading
import time

from excavator_il.collector.client import send_episode_command
from excavator_il.collector.config import (
    CameraConfig,
    CollectionConfig,
    ControllerConfig,
    EpisodeDefaults,
    JoystickUdpConfig,
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


def test_collector_service_connects_udp_episode_socket_camera_and_safe_shutdown(tmp_path):
    port = _free_udp_port()
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

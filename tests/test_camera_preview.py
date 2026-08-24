import json
import socket
import threading
import time
from urllib.request import urlopen

from excavator_il.collector.preview import (
    LatestJpegFrame,
    LatestTelemetryFrame,
    MjpegPreviewServer,
    _write_preview_payload,
)


def test_latest_jpeg_frame_is_bounded_and_waits_for_a_new_sequence():
    latest = LatestJpegFrame()

    assert latest.wait_after(0, timeout_s=0.01) is None
    first = latest.publish(b"first-jpeg", capture_monotonic_ns=100)
    second = latest.publish(b"second-jpeg", capture_monotonic_ns=200)

    assert first.sequence == 1
    assert second.sequence == 2
    assert latest.wait_after(1, timeout_s=0.01) == second


def test_preview_client_disconnect_is_an_expected_write_result():
    class DisconnectedClient:
        def write(self, _payload):
            raise BrokenPipeError("browser closed the preview request")

    assert _write_preview_payload(DisconnectedClient(), b"jpeg") is False


def test_mjpeg_preview_server_streams_the_collector_owned_latest_frame():
    latest = LatestJpegFrame()
    telemetry = LatestTelemetryFrame()
    server = MjpegPreviewServer(
        latest,
        telemetry=telemetry,
        bind_host="127.0.0.1",
        port=0,
        allowed_client_host="127.0.0.1",
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    latest.publish(b"fixture-jpeg", capture_monotonic_ns=time.monotonic_ns())
    telemetry.publish(
        {
            "control_seq": 42,
            "sensor_seq": 21,
            "boom_pos_mm": 123.4,
            "stick_pos_mm": 234.5,
            "bucket_pos_mm": 345.6,
            "boom_angle_deg": 10.1,
            "arm_angle_deg": 20.2,
            "bucket_angle_deg": 30.3,
            "swing_angle_deg": -4.5,
            "sensor_is_new": 1,
            "sensor_valid": True,
            "control_enabled": 1,
            "command_timed_out": 0,
            "fault_flags": 0,
        },
        receive_monotonic_ns=time.monotonic_ns(),
    )

    with urlopen(f"http://127.0.0.1:{server.port}/healthz", timeout=1.0) as health:
        health_payload = json.load(health)
        assert health_payload["ok"] is True
        assert health_payload["frame_available"] is True
        assert health_payload["cameras"]["front"]["frame_available"] is True
    with urlopen(
        f"http://127.0.0.1:{server.port}/camera/front.jpg", timeout=1.0
    ) as snapshot:
        assert snapshot.headers["Content-Type"] == "image/jpeg"
        assert snapshot.read() == b"fixture-jpeg"
    with urlopen(
        f"http://127.0.0.1:{server.port}/telemetry/latest.json", timeout=1.0
    ) as response:
        payload = json.load(response)
    assert payload["control_seq"] == 42
    assert payload["command_timed_out"] is False
    assert payload["cylinders_mm"] == {
        "boom": 123.4,
        "stick": 234.5,
        "bucket": 345.6,
    }
    assert payload["joint_angles_deg"] == {
        "boom": 10.1,
        "arm": 20.2,
        "bucket": 30.3,
        "swing": -4.5,
    }
    assert payload["age_ms"] >= 0.0

    client = socket.create_connection(("127.0.0.1", server.port), timeout=1.0)
    client.settimeout(1.0)
    client.sendall(
        b"GET /camera/front.mjpg HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Connection: close\r\n\r\n"
    )
    response = b""
    deadline = time.monotonic() + 1.0
    while b"fixture-jpeg" not in response and time.monotonic() < deadline:
        response += client.recv(4096)

    client.close()
    server.close()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert b"HTTP/1.0 200 OK" in response
    assert b"multipart/x-mixed-replace; boundary=frame" in response
    assert b"Content-Type: image/jpeg" in response
    assert b"fixture-jpeg" in response


def test_preview_server_exposes_front_and_dump_as_independent_named_streams():
    front = LatestJpegFrame()
    dump = LatestJpegFrame()
    server = MjpegPreviewServer(
        {"front": front, "dump": dump},
        bind_host="127.0.0.1",
        port=0,
        allowed_client_host="127.0.0.1",
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    front.publish(b"front-jpeg", capture_monotonic_ns=time.monotonic_ns())
    dump.publish(b"dump-jpeg", capture_monotonic_ns=time.monotonic_ns())

    try:
        with urlopen(
            f"http://127.0.0.1:{server.port}/camera/front.jpg", timeout=1.0
        ) as response:
            assert response.read() == b"front-jpeg"
        with urlopen(
            f"http://127.0.0.1:{server.port}/camera/dump.jpg", timeout=1.0
        ) as response:
            assert response.read() == b"dump-jpeg"
        with urlopen(
            f"http://127.0.0.1:{server.port}/healthz", timeout=1.0
        ) as response:
            health = json.load(response)
    finally:
        server.close()
        thread.join(timeout=1.0)

    assert health["ok"] is True
    assert health["cameras"]["front"]["frame_available"] is True
    assert health["cameras"]["dump"]["frame_available"] is True
    assert not thread.is_alive()

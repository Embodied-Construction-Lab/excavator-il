import json
import socket
import threading
import time
from urllib.request import urlopen

from excavator_il.collector.preview import LatestJpegFrame, MjpegPreviewServer


def test_latest_jpeg_frame_is_bounded_and_waits_for_a_new_sequence():
    latest = LatestJpegFrame()

    assert latest.wait_after(0, timeout_s=0.01) is None
    first = latest.publish(b"first-jpeg", capture_monotonic_ns=100)
    second = latest.publish(b"second-jpeg", capture_monotonic_ns=200)

    assert first.sequence == 1
    assert second.sequence == 2
    assert latest.wait_after(1, timeout_s=0.01) == second


def test_mjpeg_preview_server_streams_the_collector_owned_latest_frame():
    latest = LatestJpegFrame()
    server = MjpegPreviewServer(
        latest,
        bind_host="127.0.0.1",
        port=0,
        allowed_client_host="127.0.0.1",
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    latest.publish(b"fixture-jpeg", capture_monotonic_ns=time.monotonic_ns())

    with urlopen(f"http://127.0.0.1:{server.port}/healthz", timeout=1.0) as health:
        assert json.load(health) == {"ok": True, "frame_available": True}
    with urlopen(
        f"http://127.0.0.1:{server.port}/camera/front.jpg", timeout=1.0
    ) as snapshot:
        assert snapshot.headers["Content-Type"] == "image/jpeg"
        assert snapshot.read() == b"fixture-jpeg"

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

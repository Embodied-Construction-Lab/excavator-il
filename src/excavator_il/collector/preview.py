"""Bounded read-only MJPEG preview for the Collector-owned camera."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True)
class JpegPreviewFrame:
    sequence: int
    capture_monotonic_ns: int
    encoded_image: bytes


class LatestJpegFrame:
    """Keep exactly one immutable JPEG and notify readers of replacement."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: JpegPreviewFrame | None = None

    def publish(
        self, encoded_image: bytes, *, capture_monotonic_ns: int
    ) -> JpegPreviewFrame:
        if not isinstance(encoded_image, bytes) or not encoded_image:
            raise ValueError("encoded_image must be non-empty bytes")
        if (
            isinstance(capture_monotonic_ns, bool)
            or not isinstance(capture_monotonic_ns, int)
            or capture_monotonic_ns < 0
        ):
            raise ValueError("capture_monotonic_ns must be a non-negative integer")
        with self._condition:
            sequence = 1 if self._latest is None else self._latest.sequence + 1
            frame = JpegPreviewFrame(
                sequence=sequence,
                capture_monotonic_ns=capture_monotonic_ns,
                encoded_image=encoded_image,
            )
            self._latest = frame
            self._condition.notify_all()
            return frame

    def wait_after(
        self, sequence: int, *, timeout_s: float
    ) -> JpegPreviewFrame | None:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None or self._latest.sequence <= sequence:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    return None
                self._condition.wait(remaining_s)
            return self._latest


class _PreviewHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MjpegPreviewServer:
    """Serve a Collector-owned latest-frame buffer to one allowed PC host."""

    def __init__(
        self,
        frames: LatestJpegFrame,
        *,
        bind_host: str,
        port: int,
        allowed_client_host: str,
    ) -> None:
        if not isinstance(bind_host, str) or not bind_host:
            raise ValueError("bind_host must be non-empty")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        if not isinstance(allowed_client_host, str) or not allowed_client_host:
            raise ValueError("allowed_client_host must be non-empty")
        self._frames = frames
        self._allowed_client_host = allowed_client_host
        self._stop = threading.Event()
        self._server = _PreviewHttpServer(
            (bind_host, port), self._handler_type()
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.1)

    def close(self) -> None:
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()

    def _handler_type(self):
        owner = self

        class PreviewHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.client_address[0] != owner._allowed_client_host:
                    self.send_error(403, "preview client is not allowed")
                    return
                route = self.path.split("?", maxsplit=1)[0]
                if route == "/healthz":
                    self._health()
                elif route == "/camera/front.jpg":
                    self._snapshot()
                elif route == "/camera/front.mjpg":
                    self._stream()
                else:
                    self.send_error(404, "preview route not found")

            def _health(self) -> None:
                frame = owner._frames.wait_after(0, timeout_s=0.001)
                payload = json.dumps(
                    {"ok": True, "frame_available": frame is not None},
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def _snapshot(self) -> None:
                frame = owner._frames.wait_after(0, timeout_s=1.0)
                if frame is None:
                    self.send_error(503, "camera frame is not available")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame.encoded_image)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(frame.encoded_image)

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                sequence = 0
                while not owner._stop.is_set():
                    frame = owner._frames.wait_after(sequence, timeout_s=1.0)
                    if frame is None:
                        continue
                    sequence = frame.sequence
                    part = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame.encoded_image)}\r\n\r\n".encode(
                            "ascii"
                        )
                        + frame.encoded_image
                        + b"\r\n"
                    )
                    try:
                        self.wfile.write(part)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        return

            def log_message(self, _format: str, *_args: object) -> None:
                return

        return PreviewHandler

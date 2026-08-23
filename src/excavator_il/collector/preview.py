"""Bounded read-only MJPEG preview for the Collector-owned camera."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Mapping, Protocol


class _PreviewWriter(Protocol):
    def write(self, payload: bytes) -> object: ...


def _write_preview_payload(writer: _PreviewWriter, payload: bytes) -> bool:
    """Write one response body; a browser closing the request is not a fault."""

    try:
        writer.write(payload)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
        return False
    return True


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


@dataclass(frozen=True)
class TelemetryPreviewFrame:
    receive_monotonic_ns: int
    values: Mapping[str, int | float | str | bool]


class LatestTelemetryFrame:
    """Keep one immutable parsed STM32 frame for read-only operator display."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: TelemetryPreviewFrame | None = None

    def publish(
        self,
        values: Mapping[str, int | float | str | bool],
        *,
        receive_monotonic_ns: int,
    ) -> TelemetryPreviewFrame:
        if (
            isinstance(receive_monotonic_ns, bool)
            or not isinstance(receive_monotonic_ns, int)
            or receive_monotonic_ns < 0
        ):
            raise ValueError("receive_monotonic_ns must be a non-negative integer")
        frame = TelemetryPreviewFrame(
            receive_monotonic_ns=receive_monotonic_ns,
            values=MappingProxyType(dict(values)),
        )
        with self._lock:
            self._latest = frame
        return frame

    def snapshot(self) -> TelemetryPreviewFrame | None:
        with self._lock:
            return self._latest


class _PreviewHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MjpegPreviewServer:
    """Serve Collector-owned named latest-frame buffers to one allowed PC host."""

    def __init__(
        self,
        frames: LatestJpegFrame | Mapping[str, LatestJpegFrame],
        *,
        telemetry: LatestTelemetryFrame | None = None,
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
        if isinstance(frames, LatestJpegFrame):
            named_frames = {"front": frames}
        else:
            named_frames = dict(frames)
        if "front" not in named_frames:
            raise ValueError("frames must contain the front camera")
        if not set(named_frames).issubset({"front", "dump"}):
            raise ValueError("camera preview names must be front or dump")
        if any(not isinstance(value, LatestJpegFrame) for value in named_frames.values()):
            raise ValueError("camera preview values must be LatestJpegFrame")
        self._frames = MappingProxyType(named_frames)
        self._telemetry = telemetry
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
                elif route.startswith("/camera/") and route.endswith(".jpg"):
                    self._snapshot_route(route)
                elif route.startswith("/camera/") and route.endswith(".mjpg"):
                    self._stream_route(route)
                elif route == "/telemetry/latest.json":
                    self._telemetry_snapshot()
                else:
                    self.send_error(404, "preview route not found")

            def _health(self) -> None:
                cameras: dict[str, dict[str, bool | float | int]] = {}
                now_ns = time.monotonic_ns()
                for camera_id, frames in owner._frames.items():
                    frame = frames.wait_after(0, timeout_s=0.001)
                    cameras[camera_id] = {
                        "frame_available": frame is not None,
                        "sequence": 0 if frame is None else frame.sequence,
                        "age_ms": (
                            0.0
                            if frame is None
                            else max(
                                0.0,
                                (now_ns - frame.capture_monotonic_ns) / 1_000_000.0,
                            )
                        ),
                    }
                payload = json.dumps(
                    {
                        "ok": all(
                            camera["frame_available"] for camera in cameras.values()
                        ),
                        "frame_available": cameras["front"]["frame_available"],
                        "cameras": cameras,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                _write_preview_payload(self.wfile, payload)

            @staticmethod
            def _camera_id(route: str, suffix: str) -> str:
                return route.removeprefix("/camera/").removesuffix(suffix)

            def _frames_for(self, camera_id: str) -> LatestJpegFrame | None:
                return owner._frames.get(camera_id)

            def _snapshot_route(self, route: str) -> None:
                frames = self._frames_for(self._camera_id(route, ".jpg"))
                if frames is None:
                    self.send_error(404, "camera preview is disabled")
                    return
                self._snapshot(frames)

            def _stream_route(self, route: str) -> None:
                frames = self._frames_for(self._camera_id(route, ".mjpg"))
                if frames is None:
                    self.send_error(404, "camera preview is disabled")
                    return
                self._stream(frames)

            def _snapshot(self, frames: LatestJpegFrame) -> None:
                frame = frames.wait_after(0, timeout_s=1.0)
                if frame is None:
                    self.send_error(503, "camera frame is not available")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame.encoded_image)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                _write_preview_payload(self.wfile, frame.encoded_image)

            def _telemetry_snapshot(self) -> None:
                if owner._telemetry is None:
                    self.send_error(404, "telemetry preview is disabled")
                    return
                frame = owner._telemetry.snapshot()
                if frame is None:
                    self.send_error(503, "telemetry frame is not available")
                    return
                values = frame.values
                payload = json.dumps(
                    {
                        "control_seq": int(values["control_seq"]),
                        "sensor_seq": int(values["sensor_seq"]),
                        "sensor_is_new": bool(values["sensor_is_new"]),
                        "sensor_valid": bool(values.get("sensor_valid", False)),
                        "control_enabled": bool(values["control_enabled"]),
                        "command_timed_out": bool(values["command_timed_out"]),
                        "fault_flags": int(values["fault_flags"]),
                        "age_ms": max(
                            0.0,
                            (time.monotonic_ns() - frame.receive_monotonic_ns)
                            / 1_000_000.0,
                        ),
                        "cylinders_mm": {
                            "boom": float(values["boom_pos_mm"]),
                            "stick": float(values["stick_pos_mm"]),
                            "bucket": float(values["bucket_pos_mm"]),
                        },
                        "joint_angles_deg": {
                            "boom": float(values["boom_angle_deg"]),
                            "arm": float(values["arm_angle_deg"]),
                            "bucket": float(values["bucket_angle_deg"]),
                            "swing": float(values["swing_angle_deg"]),
                        },
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                _write_preview_payload(self.wfile, payload)

            def _stream(self, frames: LatestJpegFrame) -> None:
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
                    frame = frames.wait_after(sequence, timeout_s=1.0)
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

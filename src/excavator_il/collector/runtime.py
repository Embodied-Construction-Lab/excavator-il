"""Testable runtime operations at the UDP, serial and camera boundaries."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from .core import CollectorCore, CollectorDecision
from .preview import LatestJpegFrame, LatestTelemetryFrame
from .recorder import EpisodeRecorder


class CollectorRuntime:
    def __init__(
        self,
        *,
        core: CollectorCore,
        recorder: EpisodeRecorder,
        serial_port: Any,
        camera: Any,
        allowed_pc_host: str,
        joystick_timeout_ms: int,
        camera_preview: LatestJpegFrame | None = None,
        telemetry_preview: LatestTelemetryFrame | None = None,
    ) -> None:
        self._core = core
        self._recorder = recorder
        self._serial = serial_port
        self._camera = camera
        self._allowed_pc_host = allowed_pc_host
        self._timeout_ns = joystick_timeout_ms * 1_000_000
        self._last_joystick_ns: int | None = None
        self._timeout_zero_sent = False
        self._serial_write_lock = threading.Lock()
        self._camera_preview = camera_preview
        self._telemetry_preview = telemetry_preview

    @staticmethod
    def _rejection_ack(*, receive_monotonic_ns: int, reason: str) -> bytes:
        return json.dumps(
            {
                "schema_version": "excavator_joystick_ack.v1",
                "sample_seq": None,
                "accepted": False,
                "reason": reason,
                "orin_receive_monotonic_ns": receive_monotonic_ns,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def _write(self, decision: CollectorDecision, *, monotonic_ns: int) -> None:
        assert decision.serial_payload is not None
        write_ok = False
        write_error = ""
        try:
            with self._serial_write_lock:
                written = self._serial.write(decision.serial_payload)
                if written != len(decision.serial_payload):
                    raise OSError(
                        f"short serial write: {written}/{len(decision.serial_payload)} bytes"
                    )
                self._serial.flush()
            write_ok = True
        except Exception as exc:
            write_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._core.record_command_result(
                decision,
                tx_monotonic_ns=monotonic_ns,
                write_ok=write_ok,
                write_error=write_error,
            )

    def handle_joystick(
        self,
        datagram: bytes,
        *,
        source: tuple[str, int],
        receive_monotonic_ns: int,
        receive_wall_ns: int,
    ) -> bytes:
        if source[0] != self._allowed_pc_host:
            return self._rejection_ack(
                receive_monotonic_ns=receive_monotonic_ns,
                reason="source_not_allowed",
            )
        decision = self._core.accept_joystick(
            datagram,
            source_addr=f"{source[0]}:{source[1]}",
            receive_monotonic_ns=receive_monotonic_ns,
            receive_wall_ns=receive_wall_ns,
        )
        if decision.accepted and decision.serial_payload is not None:
            self._write(decision, monotonic_ns=receive_monotonic_ns)
            self._last_joystick_ns = receive_monotonic_ns
            self._timeout_zero_sent = False
        return decision.ack_payload

    def enforce_joystick_timeout(self, monotonic_ns: int) -> bool:
        if self._last_joystick_ns is None or self._timeout_zero_sent:
            return False
        if monotonic_ns - self._last_joystick_ns <= self._timeout_ns:
            return False
        decision = self._core.make_safe_zero(
            monotonic_ns=monotonic_ns, reason="joystick_timeout"
        )
        self._write(decision, monotonic_ns=monotonic_ns)
        self._timeout_zero_sent = True
        return True

    def send_safe_zero(self, *, reason: str, monotonic_ns: int | None = None) -> None:
        stamp = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        decision = self._core.make_safe_zero(monotonic_ns=stamp, reason=reason)
        self._write(decision, monotonic_ns=stamp)

    def accept_stm32(
        self, raw_line: bytes, *, receive_monotonic_ns: int, receive_wall_ns: int
    ) -> None:
        frame = self._core.accept_stm32(
            raw_line,
            receive_monotonic_ns=receive_monotonic_ns,
            receive_wall_ns=receive_wall_ns,
        )
        if frame is not None and self._telemetry_preview is not None:
            values = dict(frame.values)
            values["sensor_valid"] = frame.sensor_valid
            self._telemetry_preview.publish(
                values,
                receive_monotonic_ns=receive_monotonic_ns,
            )

    def capture_once(self) -> str | None:
        frame = self._camera.read_encoded()
        if self._camera_preview is not None:
            self._camera_preview.publish(
                frame.encoded_image,
                capture_monotonic_ns=frame.capture_monotonic_ns,
            )
        return self._recorder.record_camera(
            encoded_image=frame.encoded_image,
            capture_monotonic_ns=frame.capture_monotonic_ns,
            extension=frame.extension,
        )

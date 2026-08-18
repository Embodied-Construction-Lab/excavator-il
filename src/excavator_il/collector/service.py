"""Orin hardware collector service for joystick, STM32 telemetry and UVC RGB."""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Any

from .camera import UvcCamera
from .config import CollectionConfig, load_collection_config
from .control import EpisodeController
from .core import CollectorCore
from .preview import LatestJpegFrame, MjpegPreviewServer
from .recorder import EpisodeRecorder
from .runtime import CollectorRuntime


LOGGER = logging.getLogger("excavator_il.collector")
_MAX_CONTROL_REQUEST_BYTES = 16_384
_STM32_SYNC_TIMEOUT_S = 2.0


class CollectorService:
    def __init__(
        self,
        config: CollectionConfig,
        *,
        serial_port: Any,
        camera: Any,
    ) -> None:
        self._config = config
        self._serial = serial_port
        self._camera = camera
        self._recorder = EpisodeRecorder(config.data_root)
        self._camera_preview = LatestJpegFrame()
        self._preview_server: MjpegPreviewServer | None = None
        self._core = CollectorCore(
            recorder=self._recorder,
            expected_device_ids=config.controllers.device_ids,
            mapping_id=config.controllers.mapping_id,
            calibration_id=config.controllers.calibration_id,
            deadzone=config.controllers.deadzone,
        )
        self._runtime = CollectorRuntime(
            core=self._core,
            recorder=self._recorder,
            serial_port=serial_port,
            camera=camera,
            allowed_pc_host=config.joystick.allowed_pc_host,
            joystick_timeout_ms=config.joystick.timeout_ms,
            camera_preview=self._camera_preview,
        )
        self._episode_controller = EpisodeController(
            recorder=self._recorder,
            defaults=config.episode_defaults,
            camera=config.camera,
        )
        self._stop = threading.Event()
        self._worker_errors: queue.Queue[BaseException] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._control_socket: socket.socket | None = None

    def request_stop(self) -> None:
        self._stop.set()

    def _fail_worker(self, worker: str, exc: BaseException) -> None:
        LOGGER.exception("%s worker failed", worker, exc_info=exc)
        self._worker_errors.put(exc)
        self._stop.set()

    def _serial_loop(self) -> None:
        try:
            while not self._stop.is_set():
                raw_line = self._serial.readline()
                receive_monotonic_ns = time.monotonic_ns()
                if not raw_line:
                    continue
                self._runtime.accept_stm32(
                    raw_line,
                    receive_monotonic_ns=receive_monotonic_ns,
                    receive_wall_ns=time.time_ns(),
                )
        except BaseException as exc:
            self._fail_worker("serial", exc)

    def _camera_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._runtime.capture_once()
        except BaseException as exc:
            self._fail_worker("camera", exc)

    @staticmethod
    def _read_control_request(connection: socket.socket) -> dict[str, Any]:
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_CONTROL_REQUEST_BYTES:
            chunk = connection.recv(min(4096, _MAX_CONTROL_REQUEST_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if b"\n" in chunk:
                break
        if total > _MAX_CONTROL_REQUEST_BYTES:
            raise ValueError("episode control request is too large")
        raw = b"".join(chunks).split(b"\n", maxsplit=1)[0]
        try:
            request = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid episode control JSON: {exc}") from exc
        if not isinstance(request, dict):
            raise ValueError("episode control request must be an object")
        return request

    def _control_loop(self) -> None:
        path = self._config.episode_control_socket
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._control_socket = server
            server.bind(str(path))
            os.chmod(path, 0o600)
            server.listen(4)
            server.settimeout(0.2)
            while not self._stop.is_set():
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    continue
                with connection:
                    try:
                        response = self._episode_controller.handle(
                            self._read_control_request(connection)
                        )
                    except (RuntimeError, ValueError) as exc:
                        response = {"ok": False, "error": str(exc)}
                    connection.sendall(
                        (json.dumps(response, separators=(",", ":")) + "\n").encode(
                            "utf-8"
                        )
                    )
        except BaseException as exc:
            self._fail_worker("episode-control", exc)
        finally:
            if self._control_socket is not None:
                self._control_socket.close()
                self._control_socket = None
            path.unlink(missing_ok=True)

    def _preview_loop(self) -> None:
        assert self._preview_server is not None
        try:
            self._preview_server.serve_forever()
        except BaseException as exc:
            self._fail_worker("camera-preview", exc)

    def _start_workers(self) -> None:
        if self._config.camera_preview is not None:
            self._preview_server = MjpegPreviewServer(
                self._camera_preview,
                bind_host=self._config.camera_preview.bind_host,
                port=self._config.camera_preview.port,
                allowed_client_host=self._config.joystick.allowed_pc_host,
            )
        workers = [
            ("stm32-telemetry", self._serial_loop),
            ("camera-front", self._camera_loop),
            ("episode-control", self._control_loop),
        ]
        if self._preview_server is not None:
            workers.append(("camera-preview", self._preview_loop))
        for name, target in workers:
            thread = threading.Thread(name=name, target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def _synchronize_command_sequence(self) -> int:
        deadline = time.monotonic() + _STM32_SYNC_TIMEOUT_S
        while not self._stop.is_set() and time.monotonic() < deadline:
            raw_line = self._serial.readline()
            receive_monotonic_ns = time.monotonic_ns()
            if not raw_line:
                continue
            frame = self._core.accept_stm32(
                raw_line,
                receive_monotonic_ns=receive_monotonic_ns,
                receive_wall_ns=time.time_ns(),
            )
            if frame is not None:
                return self._core.synchronize_command_sequence_from_stm32(frame)
        raise RuntimeError(
            "cannot synchronize command sequence: no valid STM32 telemetry"
        )

    def _abort_active_episode(self, reason: str) -> None:
        if not self._recorder.active:
            return
        try:
            self._episode_controller.handle({"command": "abort", "reason": reason})
        except Exception:
            LOGGER.exception("failed to finalize active Episode after %s", reason)

    def run(self) -> None:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind((self._config.joystick.bind_host, self._config.joystick.port))
        udp.settimeout(0.02)
        next_command_seq = self._synchronize_command_sequence()
        self._runtime.send_safe_zero(reason="collector_startup")
        self._start_workers()
        LOGGER.info(
            "collector ready: joystick=%s:%d allowed_pc=%s serial=%s@%d "
            "camera=%s initial_command_seq=%d",
            self._config.joystick.bind_host,
            self._config.joystick.port,
            self._config.joystick.allowed_pc_host,
            self._config.serial.port,
            self._config.serial.baudrate,
            self._config.camera.device,
            next_command_seq,
        )
        if self._preview_server is not None:
            LOGGER.info(
                "camera preview ready: http=%s:%d allowed_pc=%s",
                self._config.camera_preview.bind_host,
                self._preview_server.port,
                self._config.joystick.allowed_pc_host,
            )
        try:
            while not self._stop.is_set():
                try:
                    datagram, source = udp.recvfrom(16_384)
                except TimeoutError:
                    self._runtime.enforce_joystick_timeout(time.monotonic_ns())
                    continue
                receive_monotonic_ns = time.monotonic_ns()
                ack = self._runtime.handle_joystick(
                    datagram,
                    source=source,
                    receive_monotonic_ns=receive_monotonic_ns,
                    receive_wall_ns=time.time_ns(),
                )
                udp.sendto(ack, source)
            if not self._worker_errors.empty():
                raise RuntimeError(f"collector worker failed: {self._worker_errors.get()}")
        except BaseException:
            self._abort_active_episode("collector_runtime_error")
            raise
        finally:
            self._stop.set()
            try:
                self._runtime.send_safe_zero(reason="collector_shutdown")
            except Exception:
                LOGGER.exception("failed to send shutdown safe-zero command")
            udp.close()
            if self._preview_server is not None:
                self._preview_server.close()
            for thread in self._threads:
                thread.join(timeout=1.0)
            self._abort_active_episode("collector_shutdown")


def run_collector(config_path: str | Path) -> None:
    config = load_collection_config(config_path)
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for collection; install excavator-il[collector]"
        ) from exc
    serial_port = serial.Serial(
        config.serial.port,
        config.serial.baudrate,
        timeout=0.1,
        write_timeout=0.1,
        exclusive=True,
    )
    try:
        camera = UvcCamera(config.camera)
        try:
            service = CollectorService(config, serial_port=serial_port, camera=camera)
            previous_handlers: dict[int, Any] = {}

            def stop_handler(signum: int, _frame: Any) -> None:
                LOGGER.info("received signal %d; stopping collector", signum)
                service.request_stop()

            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, stop_handler)
            try:
                service.run()
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
        finally:
            camera.close()
    finally:
        serial_port.close()

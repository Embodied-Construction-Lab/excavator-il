"""Camera-only operator preview used while the RL Runtime owns STM32."""

from __future__ import annotations

import logging
from pathlib import Path
import signal
import threading
from typing import Any

from .collector.camera import UvcCamera
from .collector.config import load_collection_config
from .collector.preview import LatestJpegFrame, MjpegPreviewServer


LOGGER = logging.getLogger("excavator_il.camera_preview")


class CameraPreviewService:
    """Serve the configured camera without touching the STM32 serial device."""

    def __init__(
        self,
        *,
        camera: Any,
        frames: LatestJpegFrame,
        server: Any,
        ready_message: str,
    ) -> None:
        self._camera = camera
        self._frames = frames
        self._server = server
        self._ready_message = ready_message
        self._stop = threading.Event()
        self._error: BaseException | None = None

    def request_stop(self) -> None:
        self._stop.set()

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except BaseException as exc:
            self._error = exc
            self._stop.set()

    def run(self) -> None:
        server_thread = threading.Thread(target=self._serve, daemon=True)
        server_thread.start()
        try:
            ready = False
            while not self._stop.is_set():
                frame = self._camera.read_encoded()
                self._frames.publish(
                    frame.encoded_image,
                    capture_monotonic_ns=frame.capture_monotonic_ns,
                )
                if not ready:
                    LOGGER.info("%s", self._ready_message)
                    ready = True
            if self._error is not None:
                raise RuntimeError(f"camera preview worker failed: {self._error}")
        finally:
            self._stop.set()
            self._server.close()
            server_thread.join(timeout=1.0)


def run_camera_preview(config_path: str | Path) -> None:
    config = load_collection_config(config_path)
    if config.camera_preview is None:
        raise ValueError("collection config must enable camera preview")
    frames = LatestJpegFrame()
    camera = UvcCamera(config.camera)
    server = MjpegPreviewServer(
        frames,
        bind_host=config.camera_preview.bind_host,
        port=config.camera_preview.port,
        allowed_client_host=config.joystick.allowed_pc_host,
    )
    service = CameraPreviewService(
        camera=camera,
        frames=frames,
        server=server,
        ready_message=(
            "camera preview ready: "
            f"http={config.camera_preview.bind_host}:{config.camera_preview.port} "
            f"allowed_pc={config.joystick.allowed_pc_host}"
        ),
    )
    previous: dict[int, Any] = {}

    def stop(signum: int, _frame: Any) -> None:
        LOGGER.info("received signal %d; stopping camera preview", signum)
        service.request_stop()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        service.run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        camera.close()

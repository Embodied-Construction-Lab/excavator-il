"""UVC camera boundary with Orin monotonic completion timestamps."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import CameraConfig


@dataclass(frozen=True)
class EncodedCameraFrame:
    capture_monotonic_ns: int
    encoded_image: bytes
    extension: str


@dataclass(frozen=True)
class RgbCameraFrame:
    capture_monotonic_ns: int
    rgb: np.ndarray
    encoded_image: bytes | None = None


class UvcCamera:
    def __init__(
        self,
        config: CameraConfig,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        cv2_module: Any | None = None,
    ) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as exc:
                raise RuntimeError(
                    "OpenCV is required for collection; install excavator-il[collector]"
                ) from exc
        self._config = config
        self._clock = monotonic_ns
        self._cv2 = cv2_module
        self._capture = cv2_module.VideoCapture(config.device)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"cannot open UVC camera {config.device}")
        settings = (
            (cv2_module.CAP_PROP_FRAME_WIDTH, config.width),
            (cv2_module.CAP_PROP_FRAME_HEIGHT, config.height),
            (cv2_module.CAP_PROP_FPS, config.nominal_fps),
        )
        for key, value in settings:
            self._capture.set(key, value)

    def read_encoded(self) -> EncodedCameraFrame:
        ok, frame = self._capture.read()
        capture_monotonic_ns = self._clock()
        if not ok or frame is None:
            raise RuntimeError(f"camera read failed for {self._config.device}")
        ok, encoded = self._cv2.imencode(
            ".jpg",
            frame,
            [self._cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("camera JPEG encoding failed")
        return EncodedCameraFrame(
            capture_monotonic_ns=capture_monotonic_ns,
            encoded_image=encoded.tobytes(),
            extension="jpg",
        )

    def read_rgb(self) -> RgbCameraFrame:
        ok, frame = self._capture.read()
        capture_monotonic_ns = self._clock()
        if not ok or frame is None:
            raise RuntimeError(f"camera read failed for {self._config.device}")
        ok, encoded = self._cv2.imencode(
            ".jpg",
            frame,
            [self._cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("camera JPEG encoding failed")
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        if rgb.dtype != np.uint8:
            raise RuntimeError("camera RGB frame must use uint8 pixels")
        return RgbCameraFrame(
            capture_monotonic_ns=capture_monotonic_ns,
            rgb=np.ascontiguousarray(rgb),
            encoded_image=encoded.tobytes(),
        )

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> UvcCamera:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

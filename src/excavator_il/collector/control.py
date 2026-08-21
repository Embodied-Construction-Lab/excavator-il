"""Episode lifecycle commands shared by the Unix socket service and CLI."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping

from .config import CameraConfig, EpisodeDefaults
from .recorder import EpisodeRecorder, EpisodeStart


class EpisodeController:
    def __init__(
        self,
        *,
        recorder: EpisodeRecorder,
        defaults: EpisodeDefaults,
        camera: CameraConfig,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._recorder = recorder
        self._defaults = defaults
        self._camera = camera
        self._monotonic_ns = monotonic_ns
        self._wall_ns = wall_ns

    @staticmethod
    def _text(request: Mapping[str, Any], field: str) -> str:
        value = request.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be non-empty text")
        return value

    def _start(self, request: Mapping[str, Any]) -> dict[str, Any]:
        target = request.get("dig_target_m", self._defaults.dig_target_m)
        if not isinstance(target, (list, tuple)) or len(target) != 3:
            raise ValueError("dig_target_m must contain three coordinates")
        target_values = tuple(float(value) for value in target)
        if any(not math.isfinite(value) for value in target_values):
            raise ValueError("dig_target_m must contain finite coordinates")
        material_id = request.get("material_id", self._defaults.material_id)
        if not isinstance(material_id, str) or not material_id:
            raise ValueError("material_id must be non-empty text")
        path = self._recorder.start(
            EpisodeStart(
                task=self._text(request, "task"),
                operator_id=self._text(request, "operator_id"),
                dig_target_m=target_values,
                material_id=material_id,
                provenance=self._defaults.provenance,
                camera_front={
                    "device_id": self._camera.device,
                    "width": self._camera.width,
                    "height": self._camera.height,
                    "nominal_fps": self._camera.nominal_fps,
                    "pixel_format": "RGB8",
                    "storage_encoding": "JPEG",
                    "timestamp_clock": "CLOCK_MONOTONIC",
                },
            ),
            start_wall_ns=self._wall_ns(),
            start_monotonic_ns=self._monotonic_ns(),
        )
        return {"ok": True, "active": True, "episode_id": path.name, "path": str(path)}

    def _finish(
        self, *, success: bool, reason: str, intervention: bool, aborted: bool
    ) -> dict[str, Any]:
        path = self._recorder.stop(
            success=success,
            failure_reason=reason,
            intervention=intervention,
            end_wall_ns=self._wall_ns(),
            end_monotonic_ns=self._monotonic_ns(),
            aborted=aborted,
        )
        return {
            "ok": True,
            "active": False,
            "episode_id": path.name,
            "path": str(path),
            "status": "aborted" if aborted else ("complete" if success else "failed"),
        }

    def _seal(self) -> dict[str, Any]:
        path = self._recorder.seal(
            end_wall_ns=self._wall_ns(),
            end_monotonic_ns=self._monotonic_ns(),
        )
        return {
            "ok": True,
            "active": False,
            "episode_id": path.name,
            "path": str(path),
            "status": "pending_review",
        }

    def _finalize(self, request: Mapping[str, Any]) -> dict[str, Any]:
        result = self._text(request, "result")
        path = self._recorder.finalize_pending(
            self._text(request, "path"),
            result=result,
            failure_reason=str(request.get("failure_reason", "")),
        )
        status = {
            "success": "complete",
            "failure": "failed",
            "aborted": "aborted",
        }[result]
        return {
            "ok": True,
            "active": False,
            "episode_id": path.name,
            "path": str(path),
            "status": status,
        }

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = self._text(request, "command")
        if command == "status":
            return {
                "ok": True,
                "active": self._recorder.active,
                "episode_id": self._recorder.episode_id,
            }
        if command == "start":
            return self._start(request)
        if command == "seal":
            return self._seal()
        if command == "finalize":
            return self._finalize(request)
        if command == "stop":
            success = request.get("success")
            if not isinstance(success, bool):
                raise ValueError("success must be boolean")
            reason = str(request.get("failure_reason", ""))
            if not success and not reason:
                raise ValueError("failure_reason is required when success is false")
            return self._finish(
                success=success,
                reason=reason,
                intervention=bool(request.get("intervention", False)),
                aborted=False,
            )
        if command == "abort":
            return self._finish(
                success=False,
                reason=self._text(request, "reason"),
                intervention=True,
                aborted=True,
            )
        raise ValueError(f"unknown episode command: {command}")

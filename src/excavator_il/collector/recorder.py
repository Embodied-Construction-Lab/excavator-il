"""Episode lifecycle and append-only raw stream persistence."""

from __future__ import annotations

import json
import re
import csv
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

from ..stm32_protocol import STM32_TELEMETRY_FIELDS


EPISODE_SCHEMA_VERSION = "excavator_demo_raw.v1"
JSON_STREAMS = ("stm32_raw", "joystick_raw", "expert_action", "command_tx")
_EPISODE_NAME = re.compile(r"episode_(\d{4,})$")
_WRITER_QUEUE_CAPACITY = 4096
_WRITER_STOP = object()


@dataclass(frozen=True)
class EpisodeStart:
    task: str
    operator_id: str
    dig_target_m: tuple[float, float, float]
    material_id: str
    provenance: Mapping[str, Any]
    camera_front: Mapping[str, Any]


class EpisodeRecorder:
    """Own one append-only demonstration Episode at a time."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser()
        self._episode_path: Path | None = None
        self._metadata: dict[str, Any] | None = None
        self._streams: dict[str, TextIO] = {}
        self._control_handle: TextIO | None = None
        self._control_writer: csv.DictWriter | None = None
        self._camera_handle: TextIO | None = None
        self._camera_writer: csv.DictWriter | None = None
        self._camera_frame_index = 0
        self._lock = threading.RLock()
        self._writer_queue: queue.Queue[object] | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_error: BaseException | None = None

    @property
    def active(self) -> bool:
        return self._episode_path is not None

    @property
    def episode_path(self) -> Path | None:
        return self._episode_path

    @property
    def episode_id(self) -> str | None:
        return None if self._episode_path is None else self._episode_path.name

    def _next_episode_id(self) -> str:
        maximum = 0
        if self._root.is_dir():
            for child in self._root.iterdir():
                match = _EPISODE_NAME.fullmatch(child.name)
                if match:
                    maximum = max(maximum, int(match.group(1)))
        return f"episode_{maximum + 1:04d}"

    @staticmethod
    def _require_text(value: str, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be non-empty text")
        return value

    @staticmethod
    def _write_metadata(episode_path: Path, value: Mapping[str, Any]) -> None:
        target = episode_path / "episode.json"
        temporary = episode_path / "episode.json.pending"
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def start(
        self,
        request: EpisodeStart,
        *,
        start_wall_ns: int,
        start_monotonic_ns: int,
    ) -> Path:
        with self._lock:
            return self._start_unlocked(
                request,
                start_wall_ns=start_wall_ns,
                start_monotonic_ns=start_monotonic_ns,
            )

    def _start_unlocked(
        self,
        request: EpisodeStart,
        *,
        start_wall_ns: int,
        start_monotonic_ns: int,
    ) -> Path:
        if self.active:
            raise RuntimeError("an Episode is already recording")
        if len(request.dig_target_m) != 3:
            raise ValueError("dig_target_m must contain three coordinates")
        episode_id = self._next_episode_id()
        episode_path = self._root / episode_id
        (episode_path / "camera_front").mkdir(parents=True, exist_ok=False)
        streams = {
            name: (episode_path / f"{name}.jsonl").open("x", encoding="utf-8")
            for name in JSON_STREAMS
        }
        control_handle = (episode_path / "control.csv").open(
            "x", newline="", encoding="utf-8"
        )
        control_fields = (
            "episode_id",
            "raw_frame_seq",
            "orin_receive_monotonic_ns",
            *STM32_TELEMETRY_FIELDS,
        )
        control_writer = csv.DictWriter(control_handle, fieldnames=control_fields)
        control_writer.writeheader()
        camera_handle = (episode_path / "camera_front_timestamps.csv").open(
            "x", newline="", encoding="utf-8"
        )
        camera_writer = csv.DictWriter(
            camera_handle,
            fieldnames=(
                "camera_frame_index",
                "camera_stamp_monotonic_ns",
                "image_path",
            ),
        )
        camera_writer.writeheader()
        metadata = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": episode_id,
            "task": self._require_text(request.task, "task"),
            "operator_id": self._require_text(request.operator_id, "operator_id"),
            "dig_target_m": [float(value) for value in request.dig_target_m],
            "material_id": self._require_text(request.material_id, "material_id"),
            "status": "recording",
            "success": None,
            "failure_reason": "",
            "intervention": False,
            "start_wall_ns": int(start_wall_ns),
            "start_monotonic_ns": int(start_monotonic_ns),
            "camera_front": dict(request.camera_front),
            **dict(request.provenance),
        }
        self._episode_path = episode_path
        self._metadata = metadata
        self._streams = streams
        self._control_handle = control_handle
        self._control_writer = control_writer
        self._camera_handle = camera_handle
        self._camera_writer = camera_writer
        self._camera_frame_index = 0
        self._write_metadata(episode_path, metadata)
        self._writer_queue = queue.Queue(maxsize=_WRITER_QUEUE_CAPACITY)
        self._writer_error = None
        self._writer_thread = threading.Thread(
            name=f"{episode_id}-writer",
            target=self._writer_loop,
            daemon=True,
        )
        self._writer_thread.start()
        return episode_path

    def _writer_loop(self) -> None:
        assert self._writer_queue is not None
        try:
            while True:
                item = self._writer_queue.get()
                if item is _WRITER_STOP:
                    return
                kind, payload = item
                if kind == "json":
                    stream, record = payload
                    handle = self._streams[stream]
                    handle.write(
                        json.dumps(
                            record, ensure_ascii=False, separators=(",", ":")
                        )
                        + "\n"
                    )
                elif kind == "control":
                    assert self._control_writer is not None
                    self._control_writer.writerow(payload)
                else:
                    raise RuntimeError(f"unknown Episode writer operation: {kind}")
        except BaseException as exc:
            self._writer_error = exc

    def _enqueue(self, operation: object) -> None:
        if self._writer_error is not None:
            raise RuntimeError(f"Episode writer failed: {self._writer_error}")
        assert self._writer_queue is not None
        try:
            self._writer_queue.put_nowait(operation)
        except queue.Full as exc:
            raise RuntimeError("Episode writer queue is full; Episode must be aborted") from exc

    def record_json(self, stream: str, record: Mapping[str, Any]) -> None:
        with self._lock:
            if not self.active:
                return
            if stream not in self._streams:
                raise ValueError(f"unknown Episode stream: {stream}")
            self._enqueue(("json", (stream, dict(record))))

    def record_control(
        self,
        *,
        raw_frame_seq: int,
        receive_monotonic_ns: int,
        telemetry: Mapping[str, Any],
    ) -> None:
        with self._lock:
            if not self.active:
                return
            self._enqueue(
                (
                    "control",
                    {
                        "episode_id": self.episode_id,
                        "raw_frame_seq": raw_frame_seq,
                        "orin_receive_monotonic_ns": receive_monotonic_ns,
                        **dict(telemetry),
                    },
                )
            )

    def record_camera(
        self,
        *,
        encoded_image: bytes,
        capture_monotonic_ns: int,
        extension: str,
    ) -> str | None:
        """Atomically persist one encoded camera frame and its Orin timestamp."""
        if not encoded_image:
            raise ValueError("encoded_image must not be empty")
        if not re.fullmatch(r"[a-z0-9]+", extension):
            raise ValueError("camera image extension must be lowercase alphanumeric text")
        with self._lock:
            if not self.active:
                return None
            assert self._episode_path is not None
            assert self._camera_writer is not None
            assert self._camera_handle is not None
            frame_index = self._camera_frame_index
            relative_path = f"camera_front/{frame_index:06d}.{extension}"
            target = self._episode_path / relative_path
            temporary = target.with_suffix(target.suffix + ".pending")
            temporary.write_bytes(encoded_image)
            temporary.replace(target)
            self._camera_writer.writerow(
                {
                    "camera_frame_index": frame_index,
                    "camera_stamp_monotonic_ns": int(capture_monotonic_ns),
                    "image_path": relative_path,
                }
            )
            self._camera_handle.flush()
            self._camera_frame_index += 1
            return relative_path

    def stop(
        self,
        *,
        success: bool,
        failure_reason: str,
        intervention: bool,
        end_wall_ns: int,
        end_monotonic_ns: int,
        aborted: bool = False,
    ) -> Path:
        with self._lock:
            if self._episode_path is None or self._metadata is None:
                raise RuntimeError("no Episode is recording")
            final_metadata = {
                **self._metadata,
                "status": "aborted" if aborted else ("complete" if success else "failed"),
                "success": bool(success),
                "failure_reason": str(failure_reason),
                "intervention": bool(intervention),
                "end_wall_ns": int(end_wall_ns),
                "end_monotonic_ns": int(end_monotonic_ns),
            }
            return self._close(final_metadata)

    def seal(self, *, end_wall_ns: int, end_monotonic_ns: int) -> Path:
        """Close all raw streams immediately, leaving only result review pending."""
        with self._lock:
            if self._episode_path is None or self._metadata is None:
                raise RuntimeError("no Episode is recording")
            pending_metadata = {
                **self._metadata,
                "status": "pending_review",
                "success": None,
                "failure_reason": "",
                "intervention": False,
                "end_wall_ns": int(end_wall_ns),
                "end_monotonic_ns": int(end_monotonic_ns),
            }
            return self._close(pending_metadata)

    def _close(self, final_metadata: Mapping[str, Any]) -> Path:
        assert self._episode_path is not None
        episode_path = self._episode_path
        assert self._writer_queue is not None
        assert self._writer_thread is not None
        self._writer_queue.put(_WRITER_STOP)
        self._writer_thread.join(timeout=5.0)
        if self._writer_thread.is_alive():
            raise RuntimeError("Episode writer did not stop within 5 seconds")
        writer_error = self._writer_error
        if writer_error is not None:
            final_metadata = {
                **final_metadata,
                "status": "aborted",
                "success": False,
                "failure_reason": f"episode_writer_failed: {writer_error}",
                "intervention": True,
            }
        for handle in self._streams.values():
            handle.flush()
            handle.close()
        for handle in (self._control_handle, self._camera_handle):
            if handle is not None:
                handle.flush()
                handle.close()
        self._write_metadata(episode_path, final_metadata)
        self._episode_path = None
        self._metadata = None
        self._streams = {}
        self._control_handle = None
        self._control_writer = None
        self._camera_handle = None
        self._camera_writer = None
        self._writer_queue = None
        self._writer_thread = None
        self._writer_error = None
        if writer_error is not None:
            raise RuntimeError(f"Episode writer failed: {writer_error}")
        return episode_path

    def finalize_pending(
        self,
        episode_path: str | Path,
        *,
        result: str,
        failure_reason: str,
    ) -> Path:
        """Atomically classify a sealed Episode without reopening raw streams."""
        with self._lock:
            root = self._root.resolve()
            path = Path(episode_path).expanduser().resolve()
            if path.parent != root or _EPISODE_NAME.fullmatch(path.name) is None:
                raise ValueError("pending Episode must be a direct child of data_root")
            try:
                metadata = json.loads(
                    (path / "episode.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot load pending Episode metadata: {exc}") from exc
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("status") != "pending_review"
            ):
                raise RuntimeError("Episode status must be pending_review")
            if result == "success":
                if failure_reason:
                    raise ValueError("successful Episode cannot have a failure reason")
                status = "complete"
                success = True
                intervention = False
            elif result in ("failure", "aborted"):
                if not failure_reason:
                    raise ValueError(f"{result} Episode requires a failure reason")
                status = "failed" if result == "failure" else "aborted"
                success = False
                intervention = result == "aborted"
            else:
                raise ValueError("result must be success, failure or aborted")
            final_metadata = {
                **dict(metadata),
                "status": status,
                "success": success,
                "failure_reason": failure_reason,
                "intervention": intervention,
            }
            self._write_metadata(path, final_metadata)
            return path

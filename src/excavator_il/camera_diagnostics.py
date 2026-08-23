"""Dual-RGB camera-only field diagnostic with no motion-control dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .collector.camera import RgbCameraFrame, UvcCamera
from .collector.config import CameraConfig, load_collection_config


DIAGNOSTIC_SCHEMA_VERSION = "excavator_dual_camera_diagnostic.v1"


class CameraReader(Protocol):
    def read_rgb(self) -> RgbCameraFrame: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CameraDiagnosticThresholds:
    """Conservative rejection thresholds for obviously invalid camera streams."""

    fps_relative_tolerance: float = 1.0 / 6.0
    near_black_value: int = 5
    near_white_value: int = 250
    max_near_black_fraction: float = 0.995
    max_near_white_fraction: float = 0.995

    def __post_init__(self) -> None:
        if (
            isinstance(self.fps_relative_tolerance, bool)
            or not math.isfinite(self.fps_relative_tolerance)
            or not 0.0 < self.fps_relative_tolerance < 1.0
        ):
            raise ValueError("fps_relative_tolerance must be in (0, 1)")
        for name, value in (
            ("near_black_value", self.near_black_value),
            ("near_white_value", self.near_white_value),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError(f"{name} must be an integer in [0, 255]")
        if self.near_black_value >= self.near_white_value:
            raise ValueError("near_black_value must be below near_white_value")
        for name, value in (
            ("max_near_black_fraction", self.max_near_black_fraction),
            ("max_near_white_fraction", self.max_near_white_fraction),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class CameraDiagnosticResult:
    role: str
    configured_device: str
    resolved_device: str
    nominal_fps: int
    expected_shape: tuple[int, int, int]
    successful_frame_count: int
    measured_fps: float
    read_latency_p95_ms: float | None
    read_latency_max_ms: float | None
    frame_shape: tuple[int, int, int] | None
    mean_luma: float | None
    near_black_fraction: float | None
    near_white_fraction: float | None
    jpeg_sha256: str | None
    saved_jpeg: str | None
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["expected_shape"] = list(self.expected_shape)
        payload["frame_shape"] = (
            None if self.frame_shape is None else list(self.frame_shape)
        )
        payload["failure_reasons"] = list(self.failure_reasons)
        return payload


@dataclass(frozen=True)
class DualCameraDiagnosticReport:
    config_path: str
    duration_s: float
    thresholds: CameraDiagnosticThresholds
    devices_distinct: bool
    cameras: tuple[CameraDiagnosticResult, ...]
    failure_reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failure_reasons and all(
            not camera.failure_reasons for camera in self.cameras
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "passed": self.passed,
            "config_path": self.config_path,
            "duration_s": self.duration_s,
            "thresholds": asdict(self.thresholds),
            "devices_distinct": self.devices_distinct,
            "cameras": {
                camera.role: camera.to_dict() for camera in self.cameras
            },
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class _SampledCamera:
    result: CameraDiagnosticResult
    latest_jpeg: bytes | None


@dataclass(frozen=True)
class _FrameStatistics:
    shapes: frozenset[tuple[int, ...]]
    measured_fps: float
    mean_luma: float
    near_black_fraction: float
    near_white_fraction: float
    latest_jpeg: bytes | None


@dataclass(frozen=True)
class _FrameAccumulator:
    frame_count: int
    shapes: frozenset[tuple[int, ...]]
    luma_sum: float
    pixel_count: int
    near_black_count: int
    near_white_count: int
    latest_shape: tuple[int, ...] | None
    latest_jpeg: bytes | None


def _round_metric(value: float) -> float:
    return round(value, 6)


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _resolve_device(device: str) -> tuple[str, tuple[int, int]]:
    resolved = os.path.realpath(device)
    stat_result = os.stat(resolved)
    return resolved, (stat_result.st_dev, stat_result.st_ino)


def _frame_luma(rgb: np.ndarray) -> np.ndarray:
    return np.mean(rgb.astype(np.float32), axis=2)


def _initial_frame_accumulator() -> _FrameAccumulator:
    return _FrameAccumulator(
        frame_count=0,
        shapes=frozenset(),
        luma_sum=0.0,
        pixel_count=0,
        near_black_count=0,
        near_white_count=0,
        latest_shape=None,
        latest_jpeg=None,
    )


def _accumulate_frame(
    accumulator: _FrameAccumulator,
    frame: RgbCameraFrame,
    thresholds: CameraDiagnosticThresholds,
) -> _FrameAccumulator:
    luma = _frame_luma(frame.rgb)
    shape = tuple(frame.rgb.shape)
    return _FrameAccumulator(
        frame_count=accumulator.frame_count + 1,
        shapes=accumulator.shapes | {shape},
        luma_sum=accumulator.luma_sum + float(np.sum(luma, dtype=np.float64)),
        pixel_count=accumulator.pixel_count + int(luma.size),
        near_black_count=accumulator.near_black_count
        + int(np.count_nonzero(luma <= thresholds.near_black_value)),
        near_white_count=accumulator.near_white_count
        + int(np.count_nonzero(luma >= thresholds.near_white_value)),
        latest_shape=shape,
        latest_jpeg=frame.encoded_image,
    )


def _frame_statistics(
    accumulator: _FrameAccumulator,
    duration_s: float,
) -> _FrameStatistics:
    if accumulator.frame_count <= 0 or accumulator.pixel_count <= 0:
        raise ValueError("frame accumulator must contain pixels")
    return _FrameStatistics(
        shapes=accumulator.shapes,
        measured_fps=accumulator.frame_count / duration_s,
        mean_luma=accumulator.luma_sum / accumulator.pixel_count,
        near_black_fraction=accumulator.near_black_count / accumulator.pixel_count,
        near_white_fraction=accumulator.near_white_count / accumulator.pixel_count,
        latest_jpeg=accumulator.latest_jpeg,
    )


def _quality_reasons(
    config: CameraConfig,
    thresholds: CameraDiagnosticThresholds,
    statistics: _FrameStatistics,
) -> tuple[str, ...]:
    expected_shape = (config.height, config.width, 3)
    reasons: list[str] = []
    if statistics.shapes != {expected_shape}:
        reasons.append(
            f"frame shape mismatch: expected {expected_shape}, "
            f"observed {sorted(statistics.shapes)}"
        )
    tolerance = config.nominal_fps * thresholds.fps_relative_tolerance
    lower, upper = config.nominal_fps - tolerance, config.nominal_fps + tolerance
    if not lower <= statistics.measured_fps <= upper:
        reasons.append(
            f"measured fps {statistics.measured_fps:.3f} outside "
            f"[{lower:.3f}, {upper:.3f}]"
        )
    if statistics.near_black_fraction >= thresholds.max_near_black_fraction:
        reasons.append(
            "near-black pixel fraction "
            f"{statistics.near_black_fraction:.6f} is too high"
        )
    if statistics.near_white_fraction >= thresholds.max_near_white_fraction:
        reasons.append(
            "near-white pixel fraction "
            f"{statistics.near_white_fraction:.6f} is too high"
        )
    if not statistics.latest_jpeg:
        reasons.append("latest frame has no encoded JPEG")
    return tuple(reasons)


def _result_for_failure(
    role: str,
    config: CameraConfig,
    resolved_device: str,
    reason: str,
) -> _SampledCamera:
    return _SampledCamera(
        result=CameraDiagnosticResult(
            role=role,
            configured_device=config.device,
            resolved_device=resolved_device,
            nominal_fps=config.nominal_fps,
            expected_shape=(config.height, config.width, 3),
            successful_frame_count=0,
            measured_fps=0.0,
            read_latency_p95_ms=None,
            read_latency_max_ms=None,
            frame_shape=None,
            mean_luma=None,
            near_black_fraction=None,
            near_white_fraction=None,
            jpeg_sha256=None,
            saved_jpeg=None,
            failure_reasons=(reason,),
        ),
        latest_jpeg=None,
    )


def _sample_camera(
    *,
    role: str,
    config: CameraConfig,
    resolved_device: str,
    duration_s: float,
    thresholds: CameraDiagnosticThresholds,
    start_barrier: threading.Barrier,
    camera_factory: Callable[[CameraConfig], CameraReader],
) -> _SampledCamera:
    camera: CameraReader | None = None
    try:
        camera = camera_factory(config)
        start_barrier.wait(timeout=5.0)
        start_ns = time.monotonic_ns()
        deadline_ns = start_ns + round(duration_s * 1_000_000_000)
        accumulator = _initial_frame_accumulator()
        latencies_ms: list[float] = []
        while time.monotonic_ns() < deadline_ns:
            read_start_ns = time.monotonic_ns()
            frame = camera.read_rgb()
            read_end_ns = time.monotonic_ns()
            if read_end_ns > deadline_ns:
                break
            accumulator = _accumulate_frame(accumulator, frame, thresholds)
            latencies_ms.append((read_end_ns - read_start_ns) / 1_000_000.0)
        return _summarize_camera(
            role=role,
            config=config,
            resolved_device=resolved_device,
            duration_s=duration_s,
            thresholds=thresholds,
            accumulator=accumulator,
            latencies_ms=latencies_ms,
        )
    except threading.BrokenBarrierError:
        start_barrier.abort()
        return _result_for_failure(
            role,
            config,
            resolved_device,
            "paired camera failed before concurrent sampling",
        )
    except Exception as exc:
        start_barrier.abort()
        return _result_for_failure(role, config, resolved_device, str(exc))
    finally:
        if camera is not None:
            camera.close()


def _summarize_camera(
    *,
    role: str,
    config: CameraConfig,
    resolved_device: str,
    duration_s: float,
    thresholds: CameraDiagnosticThresholds,
    accumulator: _FrameAccumulator,
    latencies_ms: list[float],
) -> _SampledCamera:
    if accumulator.frame_count == 0:
        return _result_for_failure(role, config, resolved_device, "no successful frames")
    statistics = _frame_statistics(accumulator, duration_s)
    expected_shape = (config.height, config.width, 3)
    return _SampledCamera(
        result=CameraDiagnosticResult(
            role=role,
            configured_device=config.device,
            resolved_device=resolved_device,
            nominal_fps=config.nominal_fps,
            expected_shape=expected_shape,
            successful_frame_count=accumulator.frame_count,
            measured_fps=_round_metric(statistics.measured_fps),
            read_latency_p95_ms=_round_metric(_percentile_95(latencies_ms)),
            read_latency_max_ms=_round_metric(max(latencies_ms)),
            frame_shape=accumulator.latest_shape,
            mean_luma=_round_metric(statistics.mean_luma),
            near_black_fraction=_round_metric(statistics.near_black_fraction),
            near_white_fraction=_round_metric(statistics.near_white_fraction),
            jpeg_sha256=(
                None
                if not statistics.latest_jpeg
                else hashlib.sha256(statistics.latest_jpeg).hexdigest()
            ),
            saved_jpeg=None,
            failure_reasons=_quality_reasons(config, thresholds, statistics),
        ),
        latest_jpeg=statistics.latest_jpeg,
    )


def _validated_duration(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("duration_s must be a finite number")
    duration_s = float(value)
    if not math.isfinite(duration_s) or not 0.2 <= duration_s <= 30.0:
        raise ValueError("duration_s must be in [0.2, 30.0]")
    return duration_s


def _sample_both_cameras(
    specifications: tuple[tuple[str, CameraConfig, str], ...],
    *,
    duration_s: float,
    thresholds: CameraDiagnosticThresholds,
    camera_factory: Callable[[CameraConfig], CameraReader],
) -> tuple[_SampledCamera, ...]:
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="camera-diagnostic") as pool:
        futures = tuple(
            pool.submit(
                _sample_camera,
                role=role,
                config=config,
                resolved_device=resolved,
                duration_s=duration_s,
                thresholds=thresholds,
                start_barrier=barrier,
                camera_factory=camera_factory,
            )
            for role, config, resolved in specifications
        )
        return tuple(future.result() for future in futures)


def run_dual_camera_diagnostic(
    config_path: str | Path,
    *,
    duration_s: float = 3.0,
    output_dir: str | Path | None = None,
    thresholds: CameraDiagnosticThresholds = CameraDiagnosticThresholds(),
    camera_factory: Callable[[CameraConfig], CameraReader] = UvcCamera,
) -> DualCameraDiagnosticReport:
    """Open and sample both configured RGB cameras without touching motion I/O."""
    duration_s = _validated_duration(duration_s)
    config_file = Path(config_path).expanduser()
    config = load_collection_config(config_file)
    if config.camera_dump is None:
        raise ValueError("collection config v2 must define camera_dump")
    front_resolved, front_identity = _resolve_device(config.camera_front.device)
    dump_resolved, dump_identity = _resolve_device(config.camera_dump.device)
    if front_identity == dump_identity:
        raise ValueError("camera_front and camera_dump resolve to the same device")
    specifications = (
        ("front", config.camera_front, front_resolved),
        ("dump", config.camera_dump, dump_resolved),
    )
    sampled = _sample_both_cameras(
        specifications,
        duration_s=duration_s,
        thresholds=thresholds,
        camera_factory=camera_factory,
    )
    if output_dir is not None:
        sampled = _save_latest_jpegs(sampled, Path(output_dir).expanduser())
    return DualCameraDiagnosticReport(
        config_path=str(config_file.resolve()),
        duration_s=duration_s,
        thresholds=thresholds,
        devices_distinct=True,
        cameras=tuple(value.result for value in sampled),
        failure_reasons=(),
    )


def _save_latest_jpegs(
    sampled: tuple[_SampledCamera, ...], output_dir: Path
) -> tuple[_SampledCamera, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    updated: list[_SampledCamera] = []
    for value in sampled:
        saved_path: str | None = None
        if value.latest_jpeg:
            path = output_dir / f"{value.result.role}.jpg"
            path.write_bytes(value.latest_jpeg)
            saved_path = str(path.resolve())
        result_payload = asdict(value.result)
        result_payload["saved_jpeg"] = saved_path
        updated.append(
            _SampledCamera(
                result=CameraDiagnosticResult(**result_payload),
                latest_jpeg=value.latest_jpeg,
            )
        )
    return tuple(updated)


def _error_payload(
    *,
    config_path: Path,
    duration_s: float | None,
    thresholds: CameraDiagnosticThresholds,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "passed": False,
        "config_path": str(config_path.expanduser().resolve()),
        "duration_s": duration_s,
        "thresholds": asdict(thresholds),
        "devices_distinct": False,
        "cameras": {},
        "failure_reasons": [reason],
    }


def _normalized_error_duration(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    duration_s = float(value)
    if not math.isfinite(duration_s):
        return None
    return duration_s


def build_parser(default_config_path: str | Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(default_config_path),
        help="authoritative excavator_collection_config.v2 file",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=3.0,
        help="fixed concurrent sampling window in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional directory for the latest front.jpg and dump.jpg",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    default_config_path: str | Path = "config/collection.orin.json",
) -> int:
    args = build_parser(default_config_path).parse_args(argv)
    thresholds = CameraDiagnosticThresholds()
    try:
        report = run_dual_camera_diagnostic(
            args.config,
            duration_s=args.duration_s,
            output_dir=args.output_dir,
            thresholds=thresholds,
        )
        payload = report.to_dict()
        exit_code = 0 if report.passed else 2
    except (OSError, RuntimeError, ValueError) as exc:
        payload = _error_payload(
            config_path=args.config,
            duration_s=_normalized_error_duration(args.duration_s),
            thresholds=thresholds,
            reason=str(exc),
        )
        exit_code = 2
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return exit_code

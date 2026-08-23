from __future__ import annotations

import json
import builtins
import os
import runpy
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from excavator_il.camera_diagnostics import (
    CameraDiagnosticThresholds,
    run_dual_camera_diagnostic,
)
from excavator_il.collector.camera import RgbCameraFrame


def _write_collection_config(tmp_path: Path) -> Path:
    front = tmp_path / "video-front"
    dump = tmp_path / "video-dump"
    front.touch()
    dump.touch()
    path = tmp_path / "collection.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_collection_config.v2",
                "data_root": str(tmp_path / "raw"),
                "joystick_udp": {
                    "bind_host": "0.0.0.0",
                    "port": 18090,
                    "allowed_pc_host": "192.168.50.1",
                    "timeout_ms": 150,
                },
                "controllers": {
                    "device_ids": ["left", "right"],
                    "mapping_id": "dual_stick.v1",
                    "calibration_id": "raw.v1",
                    "deadzone": 0.15,
                },
                "stm32_serial": {"port": "/dev/never-open", "baudrate": 460800},
                "camera_front": {
                    "device": str(front),
                    "width": 32,
                    "height": 24,
                    "fps": 10,
                    "jpeg_quality": 95,
                },
                "camera_dump": {
                    "device": str(dump),
                    "width": 32,
                    "height": 24,
                    "fps": 10,
                    "jpeg_quality": 95,
                },
                "episode_control_socket": str(tmp_path / "collector.sock"),
                "episode_defaults": {
                    "dig_target_m": [0.8, 0.0, -0.2],
                    "material_id": "soil",
                    "provenance": {},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class _ConcurrentFakeCamera:
    read_barrier = threading.Barrier(2)

    def __init__(self, config) -> None:
        self._config = config
        self._closed = False
        self._first_read = True

    def read_rgb(self) -> RgbCameraFrame:
        if self._first_read:
            self._first_read = False
            self.read_barrier.wait(timeout=1.0)
        time.sleep(0.1)
        image = np.full(
            (self._config.height, self._config.width, 3),
            100,
            dtype=np.uint8,
        )
        return RgbCameraFrame(
            capture_monotonic_ns=time.monotonic_ns(),
            rgb=image,
            encoded_image=f"jpeg:{self._config.device}".encode(),
        )

    def close(self) -> None:
        self._closed = True


class _BlackSlowCamera(_ConcurrentFakeCamera):
    read_barrier = threading.Barrier(2)

    def read_rgb(self) -> RgbCameraFrame:
        if self._first_read:
            self._first_read = False
            self.read_barrier.wait(timeout=1.0)
        time.sleep(0.2)
        image = np.zeros(
            (self._config.height, self._config.width, 3), dtype=np.uint8
        )
        return RgbCameraFrame(
            capture_monotonic_ns=time.monotonic_ns(),
            rgb=image,
            encoded_image=b"black-jpeg",
        )


class _WhiteWrongShapeCamera(_ConcurrentFakeCamera):
    read_barrier = threading.Barrier(2)

    def read_rgb(self) -> RgbCameraFrame:
        if self._first_read:
            self._first_read = False
            self.read_barrier.wait(timeout=1.0)
        time.sleep(0.1)
        image = np.full(
            (self._config.height - 1, self._config.width, 3),
            255,
            dtype=np.uint8,
        )
        return RgbCameraFrame(
            capture_monotonic_ns=time.monotonic_ns(),
            rgb=image,
            encoded_image=b"white-jpeg",
        )


def test_dual_camera_diagnostic_samples_both_roles_concurrently(tmp_path: Path) -> None:
    config_path = _write_collection_config(tmp_path)

    report = run_dual_camera_diagnostic(
        config_path,
        duration_s=0.45,
        thresholds=CameraDiagnosticThresholds(),
        camera_factory=_ConcurrentFakeCamera,
    )
    payload = report.to_dict()

    assert payload["schema_version"] == "excavator_dual_camera_diagnostic.v1"
    assert payload["passed"] is True
    assert payload["devices_distinct"] is True
    assert tuple(payload["cameras"]) == ("front", "dump")
    for role in ("front", "dump"):
        camera = payload["cameras"][role]
        assert camera["successful_frame_count"] == 4
        assert camera["measured_fps"] == 8.888889
        assert camera["frame_shape"] == [24, 32, 3]
        assert camera["jpeg_sha256"]
        assert camera["failure_reasons"] == []


def test_dual_camera_diagnostic_rejects_aliases_of_same_physical_device(
    tmp_path: Path,
) -> None:
    config_path = _write_collection_config(tmp_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    front = Path(raw["camera_front"]["device"])
    dump = Path(raw["camera_dump"]["device"])
    dump.unlink()
    dump.hardlink_to(front)

    with pytest.raises(ValueError, match="same device"):
        run_dual_camera_diagnostic(
            config_path,
            duration_s=0.2,
            camera_factory=_ConcurrentFakeCamera,
        )


def test_dual_camera_diagnostic_reports_black_and_slow_stream_together(
    tmp_path: Path,
) -> None:
    report = run_dual_camera_diagnostic(
        _write_collection_config(tmp_path),
        duration_s=0.45,
        camera_factory=_BlackSlowCamera,
    )

    assert report.passed is False
    for camera in report.to_dict()["cameras"].values():
        reasons = camera["failure_reasons"]
        assert any("measured fps" in reason for reason in reasons)
        assert any("near-black" in reason for reason in reasons)


def test_dual_camera_diagnostic_rejects_wrong_shape_and_near_white_frames(
    tmp_path: Path,
) -> None:
    report = run_dual_camera_diagnostic(
        _write_collection_config(tmp_path),
        duration_s=0.45,
        camera_factory=_WhiteWrongShapeCamera,
    )

    assert report.passed is False
    for camera in report.to_dict()["cameras"].values():
        assert camera["frame_shape"] == [23, 32, 3]
        reasons = camera["failure_reasons"]
        assert any("frame shape mismatch" in reason for reason in reasons)
        assert any("near-white" in reason for reason in reasons)


def test_camera_diagnostic_thresholds_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="fps_relative_tolerance"):
        CameraDiagnosticThresholds(fps_relative_tolerance=-0.1)


def test_dual_camera_diagnostic_never_imports_or_opens_motion_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_collection_config(tmp_path)
    original_import = builtins.__import__
    original_os_open = os.open

    def _guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] == "serial":
            raise AssertionError("camera-only diagnostic imported serial")
        return original_import(name, *args, **kwargs)

    def _guarded_os_open(path, *args, **kwargs):
        if os.fspath(path) == "/dev/never-open":
            raise AssertionError("camera-only diagnostic opened STM32 serial")
        return original_os_open(path, *args, **kwargs)

    def _network_forbidden(*_args, **_kwargs):
        raise AssertionError("camera-only diagnostic opened a network socket")

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    monkeypatch.setattr(os, "open", _guarded_os_open)
    monkeypatch.setattr(socket, "socket", _network_forbidden)

    report = run_dual_camera_diagnostic(
        config_path,
        duration_s=0.45,
        camera_factory=_ConcurrentFakeCamera,
    )

    assert report.passed is True


def test_dual_camera_diagnostic_optionally_saves_latest_jpegs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "snapshots"

    report = run_dual_camera_diagnostic(
        _write_collection_config(tmp_path),
        duration_s=0.45,
        output_dir=output_dir,
        camera_factory=_ConcurrentFakeCamera,
    )

    assert report.passed is True
    for role, camera in report.to_dict()["cameras"].items():
        saved = output_dir / f"{role}.jpg"
        assert saved.read_bytes() == f"jpeg:{camera['configured_device']}".encode()
        assert camera["saved_jpeg"] == str(saved.resolve())
        assert camera["jpeg_sha256"] == __import__("hashlib").sha256(
            saved.read_bytes()
        ).hexdigest()


def test_dual_camera_diagnostic_reports_one_open_failure_for_both_roles(
    tmp_path: Path,
) -> None:
    def _factory(config):
        if config.device.endswith("video-dump"):
            raise RuntimeError("cannot open dump fixture")
        return _ConcurrentFakeCamera(config)

    report = run_dual_camera_diagnostic(
        _write_collection_config(tmp_path),
        duration_s=0.2,
        camera_factory=_factory,
    )
    cameras = report.to_dict()["cameras"]

    assert report.passed is False
    assert cameras["dump"]["failure_reasons"] == ["cannot open dump fixture"]
    assert cameras["front"]["failure_reasons"] == [
        "paired camera failed before concurrent sampling"
    ]


def test_camera_diagnostic_cli_reports_config_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from excavator_il.camera_diagnostics import main

    missing = tmp_path / "missing.json"
    exit_code = main(["--config", str(missing), "--duration-s", "0.2"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert captured.err == ""
    assert payload == {
        "cameras": {},
        "config_path": str(missing.resolve()),
        "devices_distinct": False,
        "duration_s": 0.2,
        "failure_reasons": [
            f"cannot load collection config {missing}: "
            f"[Errno 2] No such file or directory: '{missing}'"
        ],
        "passed": False,
        "schema_version": "excavator_dual_camera_diagnostic.v1",
        "thresholds": {
            "fps_relative_tolerance": 1.0 / 6.0,
            "max_near_black_fraction": 0.995,
            "max_near_white_fraction": 0.995,
            "near_black_value": 5,
            "near_white_value": 250,
        },
    }
    assert "Traceback" not in captured.out


def test_diagnostic_script_defaults_to_authoritative_orin_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import excavator_il.camera_diagnostics as diagnostics

    called: dict[str, Path] = {}

    def _main(*, default_config_path: Path) -> int:
        called["config"] = default_config_path
        return 17

    monkeypatch.setattr(diagnostics, "main", _main)
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(SystemExit, match="17"):
        runpy.run_path(
            str(repo_root / "scripts" / "diagnose_dual_camera.py"),
            run_name="__main__",
        )

    assert called["config"] == repo_root / "config" / "collection.orin.json"

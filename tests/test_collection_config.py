import json

import pytest

from excavator_il.collector.config import load_collection_config


def test_collection_config_loads_authoritative_hardware_contract(tmp_path):
    path = tmp_path / "collection.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_collection_config.v2",
                "data_root": "~/excavator-data/raw",
                "joystick_udp": {
                    "bind_host": "0.0.0.0",
                    "port": 18090,
                    "allowed_pc_host": "192.168.0.220",
                    "timeout_ms": 150,
                },
                "controllers": {
                    "device_ids": ["left-guid", "right-guid"],
                    "mapping_id": "dual_stick.v1",
                    "calibration_id": "raw.v1",
                    "deadzone": 0.15,
                },
                "stm32_serial": {"port": "/dev/ttyTHS1", "baudrate": 460800},
                "camera_front": {
                    "device": "/dev/video0",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "jpeg_quality": 95,
                },
                "camera_dump": {
                    "device": "/dev/v4l/by-path/dump-camera",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "jpeg_quality": 90,
                },
                "camera_preview_http": {
                    "bind_host": "0.0.0.0",
                    "port": 18092,
                },
                "machine_state_udp": {
                    "host": "192.168.0.220",
                    "port": 18081,
                    "machine_id": "scale_excavator_v1",
                },
                "episode_control_socket": "/run/user/1000/excavator-il.sock",
                "episode_defaults": {
                    "dig_target_m": [0.8, 0.1, -0.2],
                    "material_id": "dry_soil_01",
                    "provenance": {"firmware_commit": "abc123"},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_collection_config(path)

    assert config.serial.baudrate == 460800
    assert config.joystick.timeout_ms == 150
    assert config.controllers.device_ids == ("left-guid", "right-guid")
    assert config.camera.nominal_fps == 30
    assert config.camera_dump is not None
    assert config.camera_dump.device == "/dev/v4l/by-path/dump-camera"
    assert config.camera_preview.bind_host == "0.0.0.0"
    assert config.camera_preview.port == 18092
    assert config.machine_state_udp.host == "192.168.0.220"
    assert config.machine_state_udp.port == 18081
    assert config.machine_state_udp.machine_id == "scale_excavator_v1"
    assert config.episode_defaults.dig_target_m == (0.8, 0.1, -0.2)


def test_collection_config_rejects_wrong_frequency_or_controller_count(tmp_path):
    path = tmp_path / "collection.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_collection_config.v1",
                "data_root": "data/raw",
                "joystick_udp": {
                    "bind_host": "0.0.0.0",
                    "port": 18090,
                    "allowed_pc_host": "192.168.0.220",
                    "timeout_ms": 150,
                },
                "controllers": {
                    "device_ids": ["only-one"],
                    "mapping_id": "dual_stick.v1",
                    "calibration_id": "raw.v1",
                    "deadzone": 0.15,
                },
                "stm32_serial": {"port": "/dev/ttyTHS1", "baudrate": 115200},
                "camera_front": {
                    "device": "/dev/video0",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "jpeg_quality": 95,
                },
                "episode_control_socket": "/tmp/test.sock",
                "episode_defaults": {
                    "dig_target_m": [0, 0, 0],
                    "material_id": "soil",
                    "provenance": {},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_collection_config(path)


def _v2_collection_config() -> dict:
    return {
        "schema_version": "excavator_collection_config.v2",
        "data_root": "data/raw",
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
        "stm32_serial": {"port": "/dev/ttyTHS1", "baudrate": 460800},
        "camera_front": {
            "device": "/dev/v4l/by-path/front",
            "width": 640,
            "height": 480,
            "fps": 30,
            "jpeg_quality": 95,
        },
        "camera_dump": {
            "device": "/dev/v4l/by-path/dump",
            "width": 640,
            "height": 480,
            "fps": 30,
            "jpeg_quality": 95,
        },
        "episode_control_socket": "/tmp/test.sock",
        "episode_defaults": {
            "dig_target_m": [0.8, 0.0, -0.2],
            "material_id": "soil",
            "provenance": {},
        },
    }


def test_collection_config_v2_exposes_semantic_dual_camera_mapping(tmp_path):
    path = tmp_path / "collection.json"
    path.write_text(json.dumps(_v2_collection_config()), encoding="utf-8")

    config = load_collection_config(path)

    assert tuple(config.cameras) == ("front", "dump")
    assert config.cameras["front"] is config.camera_front
    assert config.cameras["dump"] is config.camera_dump
    assert config.camera is config.camera_front


def test_collection_config_v1_rejects_dump_camera(tmp_path):
    raw = _v2_collection_config()
    raw["schema_version"] = "excavator_collection_config.v1"
    path = tmp_path / "collection.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="camera_dump requires"):
        load_collection_config(path)


def test_collection_config_v1_remains_front_only_compatible(tmp_path):
    raw = _v2_collection_config()
    raw["schema_version"] = "excavator_collection_config.v1"
    del raw["camera_dump"]
    path = tmp_path / "collection.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_collection_config(path)

    assert tuple(config.cameras) == ("front",)
    assert config.camera is config.camera_front
    assert config.camera_dump is None


def test_collection_config_v2_rejects_same_physical_camera_for_both_roles(tmp_path):
    raw = _v2_collection_config()
    raw["camera_dump"]["device"] = raw["camera_front"]["device"]
    path = tmp_path / "collection.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="distinct devices"):
        load_collection_config(path)

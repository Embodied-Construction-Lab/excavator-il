import json

import pytest

from excavator_il.act_runtime_config import load_act_runtime_config


def _config():
    return {
        "schema_version": "excavator_act_runtime_config.v2",
        "checkpoint_path": "/models/act/checkpoint",
        "checkpoint_model_sha256": "a" * 64,
        "checkpoint_files_sha256": {
            "config.json": "b" * 64,
            "model.safetensors": "a" * 64,
        },
        "deployment_manifest_path": "/deployment/act.json",
        "machine_profile_path": "/config/machine_profile.json",
        "log_root": "/data/act-runtime",
        "device": "cuda",
        "stm32_serial": {"port": "/dev/ttyTHS1", "baudrate": 460800},
        "camera_front": {
            "device": "/dev/video1",
            "width": 640,
            "height": 480,
            "fps": 30,
        },
        "timing": {
            "max_inference_state_age_ms": 100,
            "state_silence_timeout_ms": 250,
            "max_camera_age_ms": 120,
            "max_inference_ms": 100,
        },
    }


def test_runtime_config_loads_checkpoint_hardware_and_timing_contract(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")

    config = load_act_runtime_config(path)

    assert config.checkpoint_path.as_posix() == "/models/act/checkpoint"
    assert config.checkpoint_model_sha256 == "a" * 64
    assert config.checkpoint_files_sha256["config.json"] == "b" * 64
    assert config.deployment_manifest_path.as_posix() == "/deployment/act.json"
    assert config.machine_profile_path.as_posix() == "/config/machine_profile.json"
    assert config.device == "cuda"
    assert config.serial.baudrate == 460800
    assert config.camera.shape == (480, 640)
    assert config.max_inference_state_age_ms == 100
    assert config.state_silence_timeout_ms == 250
    assert config.max_inference_ms == 100


def test_runtime_config_requires_separate_state_freshness_and_silence_limits(tmp_path):
    raw = _config()
    del raw["timing"]["state_silence_timeout_ms"]
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="state_silence_timeout_ms"):
        load_act_runtime_config(path)


def test_runtime_config_rejects_camera_shape_different_from_trained_contract(tmp_path):
    raw = _config()
    raw["camera_front"]["width"] = 1280
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="640x480"):
        load_act_runtime_config(path)


def test_runtime_config_rejects_removed_pc_operator_fields(tmp_path):
    raw = _config()
    raw["operator_auth_key_path"] = "/run/secrets/obsolete"
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected fields"):
        load_act_runtime_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_model_sha256", "not-a-sha"),
        ("device", "cpu"),
    ],
)
def test_runtime_config_rejects_nonproduction_checkpoint_or_device(
    tmp_path, field, value
):
    raw = _config()
    raw[field] = value
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError):
        load_act_runtime_config(path)

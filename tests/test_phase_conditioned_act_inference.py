import json
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from excavator_il.act_phase_conditioning import (
    PHASE_FEATURE_NAMES,
    ActPhaseSchedule,
)
from excavator_il.act_runtime import ActObservation
from excavator_il.act_runtime_config import load_act_runtime_config
from excavator_il.act_deployment import verify_deployment_manifest
from excavator_il.collector.camera import RgbCameraFrame
from excavator_il.resident_act_runtime import ResidentActRuntime
from excavator_il.resident_protocol import ResidentActState


def _runtime_config():
    return {
        "schema_version": "excavator_act_runtime_config.v5",
        "checkpoint_path": "/models/three-phase/checkpoint",
        "checkpoint_model_sha256": "a" * 64,
        "checkpoint_files_sha256": {"model.safetensors": "a" * 64},
        "deployment_manifest_path": "/models/three-phase/deployment/manifest.json",
        "machine_profile_path": "/shared/machine_profile.json",
        "log_root": "/data/act-runtime",
        "device": "cuda",
        "dig_policy_backend": "lerobot_act",
        "act_behavior_id": "act_dig_transport_dump_three_phase",
        "phase_conditioning": {
            "schema_version": "excavator_act_phase_schedule.v1",
            "dig_to_transport_step": 93,
            "transport_to_dump_step": 142,
        },
        "stm32_serial": {"port": "/dev/ttyTHS1", "baudrate": 460800},
        "cameras": {
            "front": {"device": "/dev/video0", "width": 640, "height": 480, "fps": 30},
            "dump": {"device": "/dev/video2", "width": 640, "height": 480, "fps": 30},
        },
        "timing": {
            "max_inference_state_age_ms": 100,
            "state_silence_timeout_ms": 250,
            "max_camera_age_ms": 120,
            "max_inference_ms": 100,
        },
    }


def test_v5_config_binds_the_three_phase_behavior_and_exact_schedule(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(_runtime_config()), encoding="utf-8")

    config = load_act_runtime_config(path)

    assert config.act_behavior_id == "act_dig_transport_dump_three_phase"
    assert config.phase_schedule == ActPhaseSchedule(93, 142)
    assert config.policy_state_fields[-3:] == PHASE_FEATURE_NAMES


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("phase_conditioning"), "phase_conditioning"),
        (
            lambda value: value["phase_conditioning"].update(
                transport_to_dump_step=93
            ),
            "phase boundaries",
        ),
        (
            lambda value: value.update(act_behavior_id="act_dig_lift"),
            "phase_conditioning",
        ),
    ],
)
def test_v5_config_rejects_missing_invalid_or_misbound_phase_schedule(
    tmp_path, mutate, message
):
    raw = _runtime_config()
    mutate(raw)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_act_runtime_config(path)


def test_phase_schedule_appends_exact_one_hot_without_mutating_base_state():
    schedule = ActPhaseSchedule(93, 142)
    observation = ActObservation(
        state=tuple(float(index) for index in range(11)),
        front_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        state_monotonic_ns=2000,
        camera_monotonic_ns=1900,
    )

    conditioned = schedule.condition(observation, completed_steps=93)
    named = conditioned.to_policy_observation().state_by_name

    assert tuple(named)[-3:] == PHASE_FEATURE_NAMES
    assert tuple(named[name] for name in PHASE_FEATURE_NAMES) == (0.0, 1.0, 0.0)
    assert observation.extra_state_by_name == {}


class _Transport:
    def __init__(self):
        self.sent = []

    def send_candidate(self, candidate):
        self.sent.append(candidate)


class _Engine:
    def __init__(self):
        self.reset_count = 0
        self.phase_vectors = []

    def reset(self):
        self.reset_count += 1

    def step(self, *, observation, telemetry):
        named = observation.to_policy_observation().state_by_name
        self.phase_vectors.append(tuple(named[name] for name in PHASE_FEATURE_NAMES))
        return SimpleNamespace(
            commanded_action=(0.1, 0.0, 0.0, 0.0),
            reason="motion_allowed",
            predicted_action_chunk=None,
        )


def _state(sensor_seq):
    return ResidentActState(
        state=(1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        receive_monotonic_ns=2000 + sensor_seq,
        state_monotonic_ns=1900 + sensor_seq,
        control_seq=7 + sensor_seq,
        sensor_seq=sensor_seq,
        sensor_is_new=True,
        control_enabled=True,
        estop=False,
        rs485_ok=True,
        dwj_ok=True,
        imu_ok=True,
        sensor_valid=True,
        stm32_alive=True,
        fault_flags=0,
        control_generation=4,
    )


def test_resident_runtime_resets_chunk_only_when_three_phase_boundary_changes():
    engine = _Engine()
    runtime = ResidentActRuntime(
        transport=_Transport(),
        engine=engine,
        phase_schedule=ActPhaseSchedule(2, 4),
        monotonic_ns=iter(range(10_000, 10_100)).__next__,
    )
    runtime.add_camera_frame(
        RgbCameraFrame(1800, np.zeros((2, 3, 3), dtype=np.uint8))
    )

    for sensor_seq in range(1, 6):
        runtime.process_state(_state(sensor_seq))

    assert engine.phase_vectors == [
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    # One reset for the initial generation, then one at each phase boundary.
    assert engine.reset_count == 3


def test_deployment_verifier_accepts_only_the_exact_phase_conditioned_state_contract(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"phase-model")
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_text(
        json.dumps({"action_order": ["boom", "stick", "bucket", "swing"]}),
        encoding="utf-8",
    )
    state_fields = [
        "boom_pos_m",
        "stick_pos_m",
        "bucket_pos_m",
        "boom_vel_mps",
        "stick_vel_mps",
        "bucket_vel_mps",
        "boom_angle_rad",
        "arm_angle_rad",
        "bucket_angle_rad",
        "swing_angle_rad",
        "swing_vel_radps",
        *PHASE_FEATURE_NAMES,
    ]
    manifest_value = {
        "schema_version": "excavator_act_deployment.v2",
        "checkpoint": {
            "selected": True,
            "selection_reason": "lowest safe validation deployment-prior L1",
            "files_sha256": {
                "model.safetensors": sha256(b"phase-model").hexdigest()
            },
        },
        "evaluation": {
            "validation_frame_count": 1421,
            "deployment_prior_l1": 0.1062048085,
            "max_deployment_prior_l1": 0.2,
            "action_min": -1.0171282,
            "action_max": 1.0256026,
            "all_finite": True,
            "out_of_range_sample_count": 2,
            "gross_out_of_range_sample_count": 0,
            "saturated_value_count": 2,
            "max_tolerated_normalized_magnitude": 1.25,
        },
        "data": {
            "pipeline_validation_present": False,
            "source_dataset_sha256": "c" * 64,
            "train_dataset_sha256": "a" * 64,
            "validation_dataset_sha256": "b" * 64,
        },
        "contract": {
            "action_order": ["boom", "stick", "bucket", "swing"],
            "action_fields": [
                "action_boom",
                "action_stick",
                "action_bucket",
                "action_swing",
            ],
            "state_fields": state_fields,
            "state_dim": 14,
            "action_dim": 4,
            "front_rgb_chw": [3, 480, 640],
            "dump_rgb_chw": [3, 480, 640],
            "chunk_size": 20,
            "n_action_steps": 10,
            "input_feature_keys": [
                "observation.images.dump",
                "observation.images.front",
                "observation.state",
            ],
            "temporal_ensemble_coeff": None,
        },
        "machine_profile_sha256": sha256(machine_profile.read_bytes()).hexdigest(),
    }
    manifest = tmp_path / "deployment.json"
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")

    verified = verify_deployment_manifest(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        machine_profile_path=machine_profile,
    )

    assert verified["contract"]["state_dim"] == 14
    manifest_value["contract"]["state_fields"][-1] = "phase_is_other"
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    with pytest.raises(ValueError, match="state fields"):
        verify_deployment_manifest(
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            machine_profile_path=machine_profile,
        )

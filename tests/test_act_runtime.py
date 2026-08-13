import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("lerobot", reason="install excavator-il[training] for ACT tests")

from excavator_il.act_runtime import (
    ActRuntimeController,
    ActRuntimeEngine,
    ActObservation,
    ActPolicySession,
    CausalObservationBuffer,
    OperatorDeadmanGate,
    RuntimeMode,
    state_from_stm32_telemetry,
    warmup_act_policy_session,
)
from excavator_il.joystick_protocol import (
    ControllerIdentity,
    JoystickPacket,
    encode_joystick_packet,
)
from excavator_il.collector.camera import RgbCameraFrame
from excavator_il.stm32_protocol import Stm32TelemetryFrame


class _Policy:
    def __init__(self):
        self.config = SimpleNamespace(
            chunk_size=20,
            n_action_steps=10,
            input_features={
                "observation.state": SimpleNamespace(shape=(11,)),
                "observation.images.front": SimpleNamespace(shape=(3, 2, 3)),
            },
            output_features={"action": SimpleNamespace(shape=(4,))},
        )
        self.selected_batches = []
        self.reset_count = 0

    def eval(self):
        return self

    def select_action(self, batch):
        self.selected_batches.append(batch)
        return torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32)

    def reset(self):
        self.reset_count += 1


def test_policy_session_converts_live_observation_and_uses_lerobot_select_action():
    policy = _Policy()
    session = ActPolicySession(
        policy=policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda action: action,
        device="cpu",
    )
    observation = ActObservation(
        state=tuple(float(index) for index in range(11)),
        front_rgb=np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        state_monotonic_ns=2_000_000_000,
        camera_monotonic_ns=1_990_000_000,
    )

    result = session.select_action(observation)

    assert result == pytest.approx((0.1, -0.2, 0.3, -0.4))
    assert len(policy.selected_batches) == 1
    batch = policy.selected_batches[0]
    assert tuple(batch["observation.state"].shape) == (1, 11)
    assert tuple(batch["observation.images.front"].shape) == (1, 3, 2, 3)
    assert batch["observation.images.front"].dtype == torch.float32
    assert float(batch["observation.images.front"].max()) == pytest.approx(17 / 255)


def test_policy_session_rejects_temporal_ensemble_checkpoint():
    policy = _Policy()
    policy.config.temporal_ensemble_coeff = 0.01

    with pytest.raises(ValueError, match="temporal ensemble"):
        ActPolicySession(
            policy=policy,
            preprocessor=lambda batch: batch,
            postprocessor=lambda action: action,
            device="cpu",
        )


def test_policy_session_rejects_extra_image_feature():
    policy = _Policy()
    policy.config.input_features["observation.images.wrist"] = SimpleNamespace(
        shape=(3, 2, 3)
    )

    with pytest.raises(ValueError, match="single front RGB"):
        ActPolicySession(
            policy=policy,
            preprocessor=lambda batch: batch,
            postprocessor=lambda action: action,
            device="cpu",
        )


def test_live_state_uses_the_exact_training_units_and_order():
    telemetry = {
        "boom_pos_mm": 1500.0,
        "stick_pos_mm": 1600.0,
        "bucket_pos_mm": 1700.0,
        "boom_vel_mmps": 100.0,
        "stick_vel_mmps": -200.0,
        "bucket_vel_mmps": 300.0,
        "boom_angle_deg": 10.0,
        "arm_angle_deg": -20.0,
        "bucket_angle_deg": 30.0,
        "swing_angle_deg": -40.0,
        "swing_vel_degps": 50.0,
    }

    state = state_from_stm32_telemetry(telemetry)

    assert state[:6] == pytest.approx((1.5, 1.6, 1.7, 0.1, -0.2, 0.3))
    assert state[6:] == pytest.approx(
        tuple(np.deg2rad(value) for value in (10, -20, 30, -40, 50))
    )


def test_observation_buffer_selects_latest_camera_not_newer_than_state():
    buffer = CausalObservationBuffer(capacity=3)
    old = np.full((2, 3, 3), 1, dtype=np.uint8)
    causal = np.full((2, 3, 3), 2, dtype=np.uint8)
    future = np.full((2, 3, 3), 3, dtype=np.uint8)
    for stamp, image in ((900, old), (990, causal), (1_010, future)):
        buffer.add_camera(RgbCameraFrame(stamp, image))
    telemetry = {
        "boom_pos_mm": 1500.0,
        "stick_pos_mm": 1600.0,
        "bucket_pos_mm": 1700.0,
        "boom_vel_mmps": 100.0,
        "stick_vel_mmps": -200.0,
        "bucket_vel_mmps": 300.0,
        "boom_angle_deg": 10.0,
        "arm_angle_deg": -20.0,
        "bucket_angle_deg": 30.0,
        "swing_angle_deg": -40.0,
        "swing_vel_degps": 50.0,
    }

    observation = buffer.build(
        Stm32TelemetryFrame(receive_monotonic_ns=1_000, values=telemetry)
    )

    assert observation.camera_monotonic_ns == 990
    assert observation.state_monotonic_ns == 1_000
    assert np.array_equal(observation.front_rgb, causal)


def test_policy_warmup_checks_output_and_resets_action_queue():
    class _Session:
        def __init__(self):
            self.reset_count = 0

        def select_action(self, observation):
            assert observation.front_rgb.shape == (480, 640, 3)
            assert len(observation.state) == 11
            return (0.1, -0.2, 0.3, -0.4)

        def reset(self):
            self.reset_count += 1

    session = _Session()

    action = warmup_act_policy_session(session)

    assert action == (0.1, -0.2, 0.3, -0.4)
    assert session.reset_count == 1


def test_live_warmup_uses_real_observation_budget_and_resets_action_queue():
    class _Session:
        def __init__(self):
            self.reset_count = 0
            self.seen = []

        def select_action(self, observation):
            self.seen.append(observation)
            return (0.1, -0.2, 0.3, -0.4)

        def reset(self):
            self.reset_count += 1

    ticks = iter((1_000, 1_001))
    session = _Session()
    engine = ActRuntimeEngine(
        session=session,
        controller=ActRuntimeController(mode=RuntimeMode.SHADOW),
        monotonic_ns=lambda: next(ticks),
    )
    observation = ActObservation(
        state=(0.0,) * 11,
        front_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        state_monotonic_ns=900,
        camera_monotonic_ns=800,
    )

    action = engine.warmup_live_observation(observation)

    assert action == (0.1, -0.2, 0.3, -0.4)
    assert session.seen == [observation]
    assert session.reset_count == 1


def test_shadow_mode_logs_prediction_but_never_permits_serial_write():
    controller = ActRuntimeController(mode=RuntimeMode.SHADOW)

    decision = controller.decide(
        predicted_action=(0.1, -0.2, 0.3, -0.4),
        state_monotonic_ns=1_000_000_000,
        camera_monotonic_ns=990_000_000,
        now_monotonic_ns=1_010_000_000,
        telemetry={
            "control_enabled": 1,
            "estop": 0,
            "fault_flags": 0,
            "rs485_ok": 1,
            "dwj_ok": 1,
            "imu_ok": 1,
        },
        operator_enabled=True,
        operator_monotonic_ns=1_005_000_000,
    )

    assert decision.predicted_action == pytest.approx((0.1, -0.2, 0.3, -0.4))
    assert decision.commanded_action == (0.0, 0.0, 0.0, 0.0)
    assert decision.serial_axes is None
    assert decision.reason == "shadow_mode"


def test_shadow_engine_retains_lerobot_action_queue_between_steps():
    class _Session:
        def __init__(self):
            self.reset_count = 0

        def select_action(self, _observation):
            return (0.1, -0.2, 0.3, -0.4)

        def reset(self):
            self.reset_count += 1

    ticks = iter((1_000, 1_001))
    session = _Session()
    engine = ActRuntimeEngine(
        session=session,
        controller=ActRuntimeController(mode=RuntimeMode.SHADOW),
        monotonic_ns=lambda: next(ticks),
    )

    decision = engine.step(
        observation=ActObservation(
            state=(0.0,) * 11,
            front_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            state_monotonic_ns=900,
            camera_monotonic_ns=800,
        ),
        telemetry={},
        operator_snapshot=lambda: (False, None),
    )

    assert decision.reason == "shadow_mode"
    assert session.reset_count == 0


def test_motion_mode_requires_all_live_safety_gates_and_maps_action_to_stm32_axes():
    controller = ActRuntimeController(
        mode=RuntimeMode.MOTION,
        motion_authorization="ALLOW_ACT_MACHINE_MOTION",
    )

    decision = controller.decide(
        predicted_action=(0.1, -0.2, 0.3, -0.4),
        state_monotonic_ns=1_000_000_000,
        camera_monotonic_ns=990_000_000,
        now_monotonic_ns=1_010_000_000,
        telemetry={
            "control_enabled": 1,
            "estop": 0,
            "fault_flags": 0,
            "rs485_ok": 1,
            "dwj_ok": 1,
            "imu_ok": 1,
        },
        operator_enabled=True,
        operator_monotonic_ns=1_005_000_000,
    )

    assert decision.commanded_action == pytest.approx((0.1, -0.2, 0.3, -0.4))
    assert decision.serial_axes == pytest.approx((-0.4, -0.2, 0.0, 0.3, 0.1, 0.0))
    assert decision.reason == "motion_allowed"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"motion_authorization": "wrong"}, "motion_unauthorized"),
        ({"operator_enabled": False}, "operator_disabled"),
        ({"operator_monotonic_ns": 800_000_000}, "operator_stale"),
        ({"state_monotonic_ns": 800_000_000}, "state_stale"),
        ({"camera_monotonic_ns": 800_000_000}, "camera_stale"),
        ({"telemetry": {"control_enabled": 0}}, "safety_state_invalid"),
        ({"telemetry": {"estop": 1}}, "safety_state_invalid"),
        ({"telemetry": {"fault_flags": 1}}, "safety_state_invalid"),
        ({"state_monotonic_ns": 1_020_000_000}, "future_timestamp"),
    ],
)
def test_motion_mode_fails_closed_for_every_runtime_gate(overrides, reason):
    overrides = dict(overrides)
    telemetry = {
        "control_enabled": 1,
        "estop": 0,
        "fault_flags": 0,
        "rs485_ok": 1,
        "dwj_ok": 1,
        "imu_ok": 1,
    }
    telemetry.update(overrides.pop("telemetry", {}))
    authorization = overrides.pop(
        "motion_authorization", "ALLOW_ACT_MACHINE_MOTION"
    )
    inputs = {
        "predicted_action": (0.1, -0.2, 0.3, -0.4),
        "state_monotonic_ns": 1_000_000_000,
        "camera_monotonic_ns": 990_000_000,
        "now_monotonic_ns": 1_010_000_000,
        "telemetry": telemetry,
        "operator_enabled": True,
        "operator_monotonic_ns": 1_005_000_000,
    }
    inputs.update(overrides)
    controller = ActRuntimeController(
        mode=RuntimeMode.MOTION,
        motion_authorization=authorization,
        max_state_age_ms=100,
        max_camera_age_ms=120,
        max_operator_age_ms=150,
    )

    decision = controller.decide(**inputs)

    assert decision.commanded_action == (0.0, 0.0, 0.0, 0.0)
    assert decision.serial_axes == (0.0,) * 6
    assert decision.reason == reason


def test_runtime_engine_resets_lerobot_queue_when_motion_gate_closes():
    class _Session:
        def __init__(self):
            self.reset_count = 0

        def select_action(self, _observation):
            return (0.1, -0.2, 0.3, -0.4)

        def reset(self):
            self.reset_count += 1

    session = _Session()
    engine = ActRuntimeEngine(
        session=session,
        controller=ActRuntimeController(
            mode=RuntimeMode.MOTION,
            motion_authorization="ALLOW_ACT_MACHINE_MOTION",
        ),
    )
    observation = ActObservation(
        state=(0.0,) * 11,
        front_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        state_monotonic_ns=1_000_000_000,
        camera_monotonic_ns=990_000_000,
    )

    decision = engine.step(
        observation=observation,
        telemetry={
            "control_enabled": 1,
            "estop": 0,
            "fault_flags": 0,
            "rs485_ok": 1,
            "dwj_ok": 1,
            "imu_ok": 1,
        },
        operator_snapshot=lambda: (False, None),
    )

    assert decision.reason == "operator_disabled"
    assert decision.commanded_action == (0.0,) * 4
    assert session.reset_count == 1


def test_runtime_engine_converts_policy_failure_to_fail_closed_decision():
    class _BrokenSession:
        def __init__(self):
            self.reset_count = 0

        def select_action(self, _observation):
            raise ValueError("model produced NaN")

        def reset(self):
            self.reset_count += 1

    session = _BrokenSession()
    engine = ActRuntimeEngine(
        session=session,
        controller=ActRuntimeController(mode=RuntimeMode.SHADOW),
    )

    decision = engine.step(
        observation=ActObservation(
            state=(0.0,) * 11,
            front_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            state_monotonic_ns=1_000,
            camera_monotonic_ns=900,
        ),
        telemetry={},
        operator_snapshot=lambda: (False, None),
    )

    assert decision.reason == "policy_error"
    assert decision.commanded_action == (0.0,) * 4
    assert decision.serial_axes is None
    assert session.reset_count == 1


def test_runtime_engine_fails_closed_when_end_to_end_inference_exceeds_budget():
    class _Session:
        def __init__(self):
            self.reset_count = 0

        def select_action(self, _observation):
            return (0.1, -0.2, 0.3, -0.4)

        def reset(self):
            self.reset_count += 1

    ticks = iter((1_000_000_000, 1_101_000_000))
    session = _Session()
    engine = ActRuntimeEngine(
        session=session,
        controller=ActRuntimeController(
            mode=RuntimeMode.MOTION,
            motion_authorization="ALLOW_ACT_MACHINE_MOTION",
        ),
        max_inference_ms=100,
        monotonic_ns=lambda: next(ticks),
    )

    decision = engine.step(
        observation=ActObservation(
            state=(0.0,) * 11,
            front_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            state_monotonic_ns=990_000_000,
            camera_monotonic_ns=980_000_000,
        ),
        telemetry={},
        operator_snapshot=lambda: (True, 1_100_000_000),
    )

    assert decision.reason == "inference_budget_exceeded"
    assert decision.commanded_action == (0.0,) * 4
    assert decision.serial_axes == (0.0,) * 6
    assert session.reset_count == 1


def test_runtime_engine_rechecks_deadman_after_inference_before_authorizing_motion():
    operator = [True, 1_000_000_000]

    class _Session:
        def __init__(self):
            self.reset_count = 0

        def select_action(self, _observation):
            operator[:] = [False, 1_020_000_000]
            return (0.1, -0.2, 0.3, -0.4)

        def reset(self):
            self.reset_count += 1

    ticks = iter((1_010_000_000, 1_030_000_000))
    session = _Session()
    engine = ActRuntimeEngine(
        session=session,
        controller=ActRuntimeController(
            mode=RuntimeMode.MOTION,
            motion_authorization="ALLOW_ACT_MACHINE_MOTION",
        ),
        monotonic_ns=lambda: next(ticks),
    )

    decision = engine.step(
        observation=ActObservation(
            state=(0.0,) * 11,
            front_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            state_monotonic_ns=1_000_000_000,
            camera_monotonic_ns=990_000_000,
        ),
        telemetry={
            "control_enabled": 1,
            "estop": 0,
            "fault_flags": 0,
            "rs485_ok": 1,
            "dwj_ok": 1,
            "imu_ok": 1,
        },
        operator_snapshot=lambda: (bool(operator[0]), int(operator[1])),
    )

    assert decision.reason == "operator_disabled"
    assert decision.commanded_action == (0.0,) * 4
    assert session.reset_count == 1


def test_runtime_engine_uses_inference_completion_time_for_freshness_gate():
    class _Session:
        def __init__(self):
            self.reset_count = 0

        def select_action(self, _observation):
            return (0.1, -0.2, 0.3, -0.4)

        def reset(self):
            self.reset_count += 1

    ticks = iter((1_000_000_000, 1_101_000_000))
    engine = ActRuntimeEngine(
        session=_Session(),
        controller=ActRuntimeController(
            mode=RuntimeMode.MOTION,
            motion_authorization="ALLOW_ACT_MACHINE_MOTION",
            max_state_age_ms=100,
        ),
        max_inference_ms=200,
        monotonic_ns=lambda: next(ticks),
    )

    decision = engine.step(
        observation=ActObservation(
            state=(0.0,) * 11,
            front_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            state_monotonic_ns=1_000_000_000,
            camera_monotonic_ns=990_000_000,
        ),
        telemetry={
            "control_enabled": 1,
            "estop": 0,
            "fault_flags": 0,
            "rs485_ok": 1,
            "dwj_ok": 1,
            "imu_ok": 1,
        },
        operator_snapshot=lambda: (True, 1_100_000_000),
    )

    assert decision.reason == "state_stale"
    assert decision.commanded_action == (0.0,) * 4


def test_operator_gate_accepts_only_configured_fresh_deadman_packets():
    gate = OperatorDeadmanGate(
        allowed_pc_host="192.168.31.219",
        expected_device_ids=("left", "right"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )
    release = JoystickPacket(
        session_id="session-a",
        sample_seq=6,
        pc_sample_monotonic_ns=100,
        pc_sample_wall_ns=200,
        axes=(0.8, -0.7, 0.0, 0.6, -0.5, 0.0),
        controllers=(
            ControllerIdentity(1, "left", "left stick", (True,)),
            ControllerIdentity(2, "right", "right stick", (False,)),
        ),
        deadman_pressed=False,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )

    gate.accept(
        encode_joystick_packet(release),
        source=("192.168.31.219", 40000),
        receive_monotonic_ns=900,
    )
    packet = JoystickPacket(
        **{
            **release.__dict__,
            "sample_seq": 7,
            "deadman_pressed": True,
        }
    )
    ack = json.loads(
        gate.accept(
            encode_joystick_packet(packet),
            source=("192.168.31.219", 40000),
            receive_monotonic_ns=1_000,
        )
    )
    enabled, stamp = gate.snapshot()

    assert ack["accepted"] is True
    assert ack["sample_seq"] == 7
    assert enabled is True
    assert stamp == 1_000


def test_operator_gate_ignores_axes_and_fails_closed_on_invalid_identity():
    gate = OperatorDeadmanGate(
        allowed_pc_host="192.168.31.219",
        expected_device_ids=("left", "right"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )
    packet = JoystickPacket(
        session_id="session-a",
        sample_seq=0,
        pc_sample_monotonic_ns=100,
        pc_sample_wall_ns=200,
        axes=(1.0, 1.0, 0.0, 1.0, 1.0, 0.0),
        controllers=(
            ControllerIdentity(1, "wrong", "left stick", (True,)),
            ControllerIdentity(2, "right", "right stick", (False,)),
        ),
        deadman_pressed=True,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )

    ack = json.loads(
        gate.accept(
            encode_joystick_packet(packet),
            source=("192.168.31.219", 40000),
            receive_monotonic_ns=1_000,
        )
    )

    assert ack["accepted"] is False
    assert gate.snapshot() == (False, 1_000)


def test_motion_operator_gate_requires_runtime_nonce_hmac_and_rejects_replay():
    from excavator_il.joystick_protocol import authenticate_json_message

    key = b"k" * 32
    nonce = "n" * 64
    gate = OperatorDeadmanGate(
        allowed_pc_host="192.168.31.219",
        expected_device_ids=("left", "right"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        authentication_key=key,
        runtime_nonce=nonce,
    )
    packet = JoystickPacket(
        session_id="session-a",
        sample_seq=0,
        pc_sample_monotonic_ns=100,
        pc_sample_wall_ns=200,
        axes=(0.0,) * 6,
        controllers=(
            ControllerIdentity(1, "left", "left", (False,)),
            ControllerIdentity(2, "right", "right", (False,)),
        ),
        deadman_pressed=False,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )
    unsigned = encode_joystick_packet(packet)

    challenge = json.loads(
        gate.accept(
            unsigned,
            source=("192.168.31.219", 40000),
            receive_monotonic_ns=1_000,
        )
    )
    signed = authenticate_json_message(json.loads(unsigned), key=key, nonce=nonce)
    accepted = json.loads(
        gate.accept(
            signed,
            source=("192.168.31.219", 40000),
            receive_monotonic_ns=2_000,
        )
    )

    assert challenge["accepted"] is False
    assert challenge["reason"] == "authentication_required"
    assert challenge["runtime_nonce"] == nonce
    assert accepted["accepted"] is True
    assert json.loads(
        gate.accept(
            signed,
            source=("192.168.31.219", 40000),
            receive_monotonic_ns=3_000,
        )
    )["accepted"] is False


def test_motion_operator_gate_revokes_enabled_state_on_authentication_failure():
    from excavator_il.joystick_protocol import authenticate_json_message

    key = b"k" * 32
    nonce = "n" * 64
    gate = OperatorDeadmanGate(
        allowed_pc_host="192.168.31.219",
        expected_device_ids=("left", "right"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        authentication_key=key,
        runtime_nonce=nonce,
    )
    release = json.loads(_operator_packet(session="session-a", sequence=0, deadman=False))
    press = json.loads(_operator_packet(session="session-a", sequence=1, deadman=True))
    for stamp, packet in (
        (1_000, release),
        (2_000, press),
    ):
        gate.accept(
            authenticate_json_message(packet, key=key, nonce=nonce),
            source=("192.168.31.219", 40000),
            receive_monotonic_ns=stamp,
        )
    assert gate.snapshot() == (True, 2_000)

    rejected = json.loads(
        gate.accept(
            _operator_packet(session="session-a", sequence=2, deadman=True),
            source=("192.168.31.219", 40000),
            receive_monotonic_ns=3_000,
        )
    )

    assert rejected["reason"] == "authentication_required"
    assert gate.snapshot() == (False, 3_000)


def test_operator_gate_requires_release_before_first_deadman_press():
    gate = OperatorDeadmanGate(
        allowed_pc_host="192.168.31.219",
        expected_device_ids=("left", "right"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )

    first_true = _operator_packet(session="session-a", sequence=0, deadman=True)
    release = _operator_packet(session="session-a", sequence=1, deadman=False)
    enabled = _operator_packet(session="session-a", sequence=2, deadman=True)

    rejected = json.loads(
        gate.accept(first_true, source=("192.168.31.219", 40000), receive_monotonic_ns=10)
    )
    gate.accept(release, source=("192.168.31.219", 40000), receive_monotonic_ns=20)
    accepted = json.loads(
        gate.accept(enabled, source=("192.168.31.219", 40000), receive_monotonic_ns=30)
    )

    assert rejected["accepted"] is False
    assert rejected["reason"] == "release_required"
    assert accepted["accepted"] is True
    assert gate.snapshot() == (True, 30)


def test_operator_gate_requires_release_before_switching_session():
    gate = OperatorDeadmanGate(
        allowed_pc_host="192.168.31.219",
        expected_device_ids=("left", "right"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )
    for sequence, deadman in ((0, False), (1, True)):
        gate.accept(
            _operator_packet(
                session="session-a", sequence=sequence, deadman=deadman
            ),
            source=("192.168.31.219", 40000),
            receive_monotonic_ns=sequence + 10,
        )

    rejected = json.loads(
        gate.accept(
            _operator_packet(session="session-b", sequence=0, deadman=True),
            source=("192.168.31.219", 40000),
            receive_monotonic_ns=20,
        )
    )

    assert rejected["accepted"] is False
    assert rejected["reason"] == "release_required"
    assert gate.snapshot()[0] is False


def _operator_packet(*, session: str, sequence: int, deadman: bool) -> bytes:
    return encode_joystick_packet(
        JoystickPacket(
            session_id=session,
            sample_seq=sequence,
            pc_sample_monotonic_ns=100,
            pc_sample_wall_ns=200,
            axes=(0.0,) * 6,
            controllers=(
                ControllerIdentity(1, "left", "left stick", (deadman,)),
                ControllerIdentity(2, "right", "right stick", (False,)),
            ),
            deadman_pressed=deadman,
            mapping_id="dual_stick.v1",
            calibration_id="raw.v1",
        )
    )

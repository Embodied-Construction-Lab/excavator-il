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
    RuntimeMode,
    state_from_stm32_telemetry,
    warmup_act_policy_session,
)
from excavator_il.collector.camera import RgbCameraFrame
from excavator_il.dig_policy import ACTION_ORDER, DigPolicyObservation
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


def test_lerobot_act_adapter_exposes_the_dig_policy_contract_and_lifecycle():
    policy = _Policy()
    adapter = ActPolicySession(
        policy=policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda action: action,
        device="cpu",
    )

    warmup_action = adapter.warmup()

    assert adapter.descriptor.backend_id == "lerobot_act"
    assert adapter.descriptor.action_order == ACTION_ORDER
    assert adapter.descriptor.output_semantics == "manual_action_normalized"
    assert warmup_action == pytest.approx((0.1, -0.2, 0.3, -0.4))
    assert tuple(policy.selected_batches[-1]["observation.images.front"].shape) == (
        1,
        3,
        2,
        3,
    )
    assert policy.reset_count >= 2


def test_lerobot_act_adapter_consumes_named_policy_observation():
    policy = _Policy()
    adapter = ActPolicySession(
        policy=policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda action: action,
        device="cpu",
    )
    observation = DigPolicyObservation(
        state_by_name={
            name: float(index)
            for index, name in enumerate(
                (
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
                )
            )
        },
        rgb_by_role={
            "front": np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        },
        state_monotonic_ns=2_000,
        camera_monotonic_ns_by_role={"front": 1_900},
    )

    action = adapter.select_action(observation)

    assert action == pytest.approx((0.1, -0.2, 0.3, -0.4))
    assert tuple(policy.selected_batches[-1]["observation.state"][0]) == tuple(
        float(index) for index in range(11)
    )


def test_lerobot_act_adapter_uses_both_named_rgb_roles_for_a_dual_camera_checkpoint():
    policy = _Policy()
    policy.config.input_features["observation.images.dump"] = SimpleNamespace(
        shape=(3, 2, 3)
    )
    adapter = ActPolicySession(
        policy=policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda action: action,
        device="cpu",
    )
    front = np.full((2, 3, 3), 10, dtype=np.uint8)
    dump = np.full((2, 3, 3), 20, dtype=np.uint8)
    observation = ActObservation(
        state=(0.0,) * 11,
        front_rgb=front,
        state_monotonic_ns=2_000,
        camera_monotonic_ns=1_900,
        extra_rgb_by_role={"dump": dump},
        extra_camera_monotonic_ns_by_role={"dump": 1_850},
    )

    adapter.select_action(observation)

    batch = policy.selected_batches[-1]
    assert set(batch) == {
        "observation.state",
        "observation.images.front",
        "observation.images.dump",
    }
    assert float(batch["observation.images.front"].mean()) == pytest.approx(10 / 255)
    assert float(batch["observation.images.dump"].mean()) == pytest.approx(20 / 255)


def test_act_observation_rejects_a_state_that_cannot_form_the_named_contract():
    observation = ActObservation(
        state=(0.0,) * 10,
        front_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        state_monotonic_ns=2_000,
        camera_monotonic_ns=1_900,
    )

    with pytest.raises(ValueError, match="11 finite values"):
        observation.to_policy_observation()


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

    with pytest.raises(ValueError, match="named front RGB"):
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
    assert len(session.seen) == 1
    assert session.seen[0].state_by_name == {
        name: 0.0
        for name in (
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
        )
    }
    assert np.array_equal(session.seen[0].rgb_by_role["front"], observation.front_rgb)
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
    )

    assert decision.commanded_action == pytest.approx((0.1, -0.2, 0.3, -0.4))
    assert decision.serial_axes == pytest.approx((-0.4, -0.2, 0.0, 0.3, 0.1, 0.0))
    assert decision.reason == "motion_allowed"


def test_motion_mode_runs_from_local_authorization_without_pc_operator_input():
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
    )

    assert decision.reason == "motion_allowed"
    assert decision.commanded_action == pytest.approx((0.1, -0.2, 0.3, -0.4))


def test_runtime_engine_selects_motion_without_pc_operator_callback():
    class _Session:
        def select_action(self, _observation):
            return (0.1, -0.2, 0.3, -0.4)

        def reset(self):
            pass

    ticks = iter((1_005_000_000, 1_010_000_000))
    engine = ActRuntimeEngine(
        session=_Session(),
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
    )

    assert decision.reason == "motion_allowed"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"motion_authorization": "wrong"}, "motion_unauthorized"),
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
    }
    inputs.update(overrides)
    controller = ActRuntimeController(
        mode=RuntimeMode.MOTION,
        motion_authorization=authorization,
        max_state_age_ms=100,
        max_camera_age_ms=120,
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
        monotonic_ns=iter((1_005_000_000, 1_010_000_000)).__next__,
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
            "estop": 1,
            "fault_flags": 0,
            "rs485_ok": 1,
            "dwj_ok": 1,
            "imu_ok": 1,
        },
    )

    assert decision.reason == "safety_state_invalid"
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
    )

    assert decision.reason == "inference_budget_exceeded"
    assert decision.commanded_action == (0.0,) * 4
    assert decision.serial_axes == (0.0,) * 6
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
    )

    assert decision.reason == "state_stale"
    assert decision.commanded_action == (0.0,) * 4

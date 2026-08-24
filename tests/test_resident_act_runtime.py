from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import types

import numpy as np
import pytest

_lerobot = types.ModuleType("lerobot")
_lerobot.__path__ = []
_lerobot_configs = types.ModuleType("lerobot.configs")
_lerobot_configs.__path__ = []
_lerobot_config_types = types.ModuleType("lerobot.configs.types")
_lerobot_config_types.FeatureType = object
_lerobot_config_types.PolicyFeature = object
_lerobot_policies = types.ModuleType("lerobot.policies")
_lerobot_policies.__path__ = []
_lerobot_policies.get_policy_class = lambda *_args, **_kwargs: None
_lerobot_policies.make_pre_post_processors = (
    lambda *_args, **_kwargs: (None, None)
)
_lerobot_policies_act = types.ModuleType("lerobot.policies.act")
_lerobot_policies_act.__path__ = []
_lerobot_policies_act_config = types.ModuleType(
    "lerobot.policies.act.configuration_act"
)
_lerobot_policies_act_config.ACTConfig = object
_lerobot_policies_act_model = types.ModuleType(
    "lerobot.policies.act.modeling_act"
)
_lerobot_policies_act_model.ACTPolicy = object
_lerobot_datasets = types.ModuleType("lerobot.datasets")
_lerobot_datasets.__path__ = []
_lerobot_dataset_metadata = types.ModuleType("lerobot.datasets.dataset_metadata")
_lerobot_dataset_metadata.LeRobotDatasetMetadata = object
_lerobot_dataset_factory = types.ModuleType("lerobot.datasets.factory")
_lerobot_dataset_factory.resolve_delta_timestamps = (
    lambda *_args, **_kwargs: None
)
_lerobot_dataset_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
_lerobot_dataset_module.LeRobotDataset = object
_lerobot.policies = _lerobot_policies
_lerobot.configs = _lerobot_configs
_lerobot.datasets = _lerobot_datasets
sys.modules.setdefault("lerobot", _lerobot)
sys.modules.setdefault("lerobot.configs", _lerobot_configs)
sys.modules.setdefault("lerobot.configs.types", _lerobot_config_types)
sys.modules.setdefault("lerobot.policies", _lerobot_policies)
sys.modules.setdefault("lerobot.policies.act", _lerobot_policies_act)
sys.modules.setdefault(
    "lerobot.policies.act.configuration_act", _lerobot_policies_act_config
)
sys.modules.setdefault(
    "lerobot.policies.act.modeling_act", _lerobot_policies_act_model
)
sys.modules.setdefault("lerobot.datasets", _lerobot_datasets)
sys.modules.setdefault(
    "lerobot.datasets.dataset_metadata", _lerobot_dataset_metadata
)
sys.modules.setdefault("lerobot.datasets.factory", _lerobot_dataset_factory)
sys.modules.setdefault(
    "lerobot.datasets.lerobot_dataset", _lerobot_dataset_module
)

import excavator_il.resident_act_runtime as resident_runtime_module
from excavator_il.act_runtime import ActRuntimeDecision
from excavator_il.collector.camera import RgbCameraFrame
from excavator_il.dig_policy import DigPolicyDescriptor, DigPolicyFactory
from excavator_il.resident_act_runtime import (
    ResidentActRuntime,
    ResidentActWorker,
    build_resident_act_worker,
    run_resident_act_worker,
)
from excavator_il.resident_protocol import (
    ResidentActOwnerClosed,
    ResidentActState,
    ResidentPolicyCandidate,
)


def _state(**overrides):
    values = {
        "state": (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        "receive_monotonic_ns": 2_000,
        "state_monotonic_ns": 1_900,
        "control_seq": 7,
        "sensor_seq": 11,
        "sensor_is_new": True,
        "control_enabled": True,
        "estop": False,
        "rs485_ok": True,
        "dwj_ok": True,
        "imu_ok": True,
        "sensor_valid": True,
        "stm32_alive": True,
        "fault_flags": 0,
        "control_generation": 4,
    }
    values.update(overrides)
    return ResidentActState(**values)


class _CandidateTransport:
    def __init__(self):
        self.sent = []

    def send_candidate(self, candidate):
        self.sent.append(candidate)


class _TelemetryPreview:
    def __init__(self):
        self.published = []

    def publish(self, values, *, receive_monotonic_ns):
        self.published.append((values, receive_monotonic_ns))


class _Engine:
    def __init__(self, decision=None):
        self.decision = decision
        self.reset_count = 0
        self.observations = []

    def reset(self):
        self.reset_count += 1

    def step(self, *, observation, telemetry):
        self.observations.append((observation, telemetry))
        if self.decision is None:
            raise AssertionError("policy must not run")
        return self.decision


def test_unsafe_state_emits_only_an_exact_zero_without_running_the_policy():
    transport = _CandidateTransport()
    engine = _Engine()
    runtime = ResidentActRuntime(
        transport=transport,
        engine=engine,
        monotonic_ns=lambda: 2_100,
    )

    step = runtime.process_state(_state(control_enabled=False))

    assert step.reason == "safety_state_invalid"
    assert step.state_monotonic_ns == 1_900
    assert step.inference_completed_monotonic_ns == 2_100
    assert step.control_generation == 4
    assert engine.observations == []
    assert transport.sent == [
        ResidentPolicyCandidate(
            source="act_dig",
            control_generation=4,
            mode="manual_action",
            action=(0.0, 0.0, 0.0, 0.0),
            created_monotonic_ns=2_100,
            valid_until_monotonic_ns=300_002_100,
        )
    ]


@pytest.mark.parametrize("candidate_ttl_ms", [199.999, 300.001, float("inf")])
def test_candidate_ttl_is_bounded_around_the_resident_control_cadence(
    candidate_ttl_ms,
):
    with pytest.raises(ValueError, match="candidate TTL"):
        ResidentActRuntime(
            transport=_CandidateTransport(),
            engine=_Engine(),
            candidate_ttl_ms=candidate_ttl_ms,
        )


def test_unsafe_20hz_non_new_state_resets_policy_without_counting_a_candidate():
    decision = ActRuntimeDecision(
        predicted_action=(0.1, -0.2, 0.3, 0.0),
        commanded_action=(0.1, -0.2, 0.3, 0.0),
        serial_axes=(0.0,) * 6,
        reason="motion_allowed",
    )
    transport = _CandidateTransport()
    engine = _Engine(decision)
    telemetry = _TelemetryPreview()
    runtime = ResidentActRuntime(
        transport=transport,
        engine=engine,
        telemetry_preview=telemetry,
        monotonic_ns=iter((2_100, 2_200)).__next__,
    )
    runtime.add_camera_frame(
        RgbCameraFrame(1_800, np.zeros((2, 3, 3), dtype=np.uint8))
    )
    runtime.process_state(_state())
    transport.sent.clear()

    step = runtime.process_state(
        _state(
            sensor_is_new=False,
            control_enabled=False,
            receive_monotonic_ns=2_050,
            state_monotonic_ns=1_950,
        )
    )

    assert step.reason == "safety_state_invalid"
    assert not step.candidate_sent
    assert transport.sent == []
    assert len(engine.observations) == 1
    assert engine.reset_count == 2
    values, receive_ns = telemetry.published[-1]
    assert receive_ns == 2_050
    assert values == {
        "control_seq": 7,
        "sensor_seq": 11,
        "sensor_is_new": False,
        "sensor_valid": True,
        "control_enabled": False,
        "estop": False,
        "command_timed_out": False,
        "fault_flags": 0,
        "control_generation": 4,
        "rs485_ok": True,
        "dwj_ok": True,
        "imu_ok": True,
        "stm32_alive": True,
        "boom_pos_mm": 1_000.0,
        "stick_pos_mm": 2_000.0,
        "bucket_pos_mm": 3_000.0,
        "boom_vel_mmps": 100.0,
        "stick_vel_mmps": 200.0,
        "bucket_vel_mmps": 300.0,
        "boom_angle_deg": pytest.approx(22.918311805),
        "arm_angle_deg": pytest.approx(28.647889757),
        "bucket_angle_deg": pytest.approx(34.377467708),
        "swing_angle_deg": pytest.approx(40.10704566),
        "swing_vel_degps": pytest.approx(45.83662361),
    }


def test_safe_state_uses_a_causal_camera_and_emits_canonical_four_axis_action():
    decision = ActRuntimeDecision(
        predicted_action=(0.1, -0.2, 0.3, -0.4),
        commanded_action=(0.1, -0.2, 0.3, -0.4),
        serial_axes=(-0.4, -0.2, 0.0, 0.3, 0.1, 0.0),
        reason="motion_allowed",
    )
    transport = _CandidateTransport()
    engine = _Engine(decision)
    runtime = ResidentActRuntime(
        transport=transport,
        engine=engine,
        monotonic_ns=lambda: 2_100,
    )
    image = np.full((2, 3, 3), 7, dtype=np.uint8)
    runtime.add_camera_frame(RgbCameraFrame(1_800, image, b"jpeg"))

    step = runtime.process_state(_state())

    assert step.reason == "motion_allowed"
    assert step.commanded_action == pytest.approx((0.1, -0.2, 0.3, -0.4))
    assert len(engine.observations) == 1
    observation, telemetry = engine.observations[0]
    assert observation.state == _state().state
    assert observation.state_monotonic_ns == 1_900
    assert observation.camera_monotonic_ns == 1_800
    assert np.array_equal(observation.front_rgb, image)
    assert telemetry == {
        "control_enabled": 1,
        "estop": 0,
        "fault_flags": 0,
        "rs485_ok": 1,
        "dwj_ok": 1,
        "imu_ok": 1,
    }
    assert transport.sent[-1].action == pytest.approx((0.1, -0.2, 0.3, -0.4))
    assert transport.sent[-1].control_generation == 4
    assert transport.sent[-1].created_monotonic_ns == 2_100


def test_generation_change_resets_the_chunk_and_sequence_gap_forces_one_zero_step():
    decision = ActRuntimeDecision(
        predicted_action=(0.1, -0.2, 0.3, 0.0),
        commanded_action=(0.1, -0.2, 0.3, 0.0),
        serial_axes=(0.0,) * 6,
        reason="motion_allowed",
    )
    transport = _CandidateTransport()
    engine = _Engine(decision)
    ticks = iter((2_100, 2_200, 2_300))
    runtime = ResidentActRuntime(
        transport=transport,
        engine=engine,
        monotonic_ns=lambda: next(ticks),
    )
    runtime.add_camera_frame(
        RgbCameraFrame(1_800, np.zeros((2, 3, 3), dtype=np.uint8))
    )

    first = runtime.process_state(_state(sensor_seq=11))
    gap = runtime.process_state(
        _state(
            sensor_seq=13,
            state_monotonic_ns=1_950,
            receive_monotonic_ns=2_050,
        )
    )
    next_generation = runtime.process_state(
        _state(
            control_generation=5,
            sensor_seq=1,
            state_monotonic_ns=2_000,
            receive_monotonic_ns=2_090,
        )
    )

    assert first.reason == "motion_allowed"
    assert gap.reason == "state_sequence_gap"
    assert gap.commanded_action == (0.0, 0.0, 0.0, 0.0)
    assert next_generation.reason == "motion_allowed"
    assert engine.reset_count == 3  # first activation, gap, and new generation
    assert len(engine.observations) == 2
    assert [candidate.control_generation for candidate in transport.sent] == [4, 4, 5]
    assert transport.sent[1].action == (0.0, 0.0, 0.0, 0.0)


def test_inactive_generation_sends_nothing_and_noncausal_camera_sends_zero():
    transport = _CandidateTransport()
    engine = _Engine()
    ticks = iter((2_100, 2_200))
    runtime = ResidentActRuntime(
        transport=transport,
        engine=engine,
        monotonic_ns=lambda: next(ticks),
    )

    inactive = runtime.process_state(_state(control_generation=0))
    runtime.add_camera_frame(
        RgbCameraFrame(2_000, np.zeros((2, 3, 3), dtype=np.uint8))
    )
    unavailable = runtime.process_state(_state(control_generation=4))

    assert inactive.reason == "inactive_generation"
    assert not inactive.candidate_sent
    assert unavailable.reason == "observation_unavailable"
    assert unavailable.commanded_action == (0.0, 0.0, 0.0, 0.0)
    assert len(transport.sent) == 1
    assert transport.sent[0].action == (0.0, 0.0, 0.0, 0.0)
    assert engine.observations == []


def test_activation_progress_resets_per_generation_without_stopping_the_worker():
    decision = ActRuntimeDecision(
        predicted_action=(0.1, -0.2, 0.3, 0.0),
        commanded_action=(0.1, -0.2, 0.3, 0.0),
        serial_axes=(0.0,) * 6,
        reason="motion_allowed",
    )
    transport = _CandidateTransport()
    engine = _Engine(decision)
    statuses = []
    runtime = ResidentActRuntime(
        transport=transport,
        engine=engine,
        status_callback=statuses.append,
        monotonic_ns=iter(range(2_100, 2_600, 100)).__next__,
    )
    runtime.add_camera_frame(
        RgbCameraFrame(1_800, np.zeros((2, 3, 3), dtype=np.uint8))
    )

    runtime.process_state(_state(sensor_seq=11))
    second = runtime.process_state(
        _state(sensor_seq=12, state_monotonic_ns=1_950)
    )

    assert second.inference_performed
    assert runtime.status.active_generation == 4
    assert runtime.status.completed_steps == 2

    third = runtime.process_state(
        _state(sensor_seq=13, state_monotonic_ns=2_000)
    )
    assert third.reason == "motion_allowed"
    assert third.inference_performed
    assert len(engine.observations) == 3
    assert runtime.status.completed_steps == 3

    runtime.process_state(
        _state(
            control_generation=5,
            sensor_seq=1,
            state_monotonic_ns=2_050,
            receive_monotonic_ns=2_060,
        )
    )
    assert runtime.status.active_generation == 5
    assert runtime.status.completed_steps == 1
    assert statuses[-1] == runtime.status


def test_worker_warms_once_and_stays_resident_while_states_keep_arriving(
    caplog, capsys
):
    decision = ActRuntimeDecision(
        predicted_action=(0.1, -0.2, 0.3, 0.0),
        commanded_action=(0.1, -0.2, 0.3, 0.0),
        serial_axes=(0.0,) * 6,
        reason="motion_allowed",
    )
    warmup_done = threading.Event()
    camera_started = threading.Event()

    class _Transport(_CandidateTransport):
        def __init__(self):
            super().__init__()
            self.connected = False
            self.closed = threading.Event()
            self.states = [
                _state(sensor_seq=11),
                _state(sensor_seq=12, state_monotonic_ns=1_950),
                _state(sensor_seq=13, state_monotonic_ns=2_000),
            ]

        def connect(self, *, timeout_s):
            assert timeout_s > 0
            assert warmup_done.is_set()
            assert camera_started.is_set()
            self.connected = True

        def receive_state(self, *, timeout_s):
            if self.states:
                time.sleep(0.01)
                return self.states.pop(0)
            self.closed.wait(timeout_s)
            return None

        def close(self):
            self.connected = False
            self.closed.set()

    class _Camera:
        def __init__(self):
            self.closed = threading.Event()
            self.read_count = 0
            self.close_count = 0

        def read_rgb(self):
            self.read_count += 1
            if self.read_count == 1:
                camera_started.set()
                return RgbCameraFrame(
                    1_800,
                    np.zeros((2, 3, 3), dtype=np.uint8),
                    b"jpeg",
                )
            self.closed.wait()
            raise RuntimeError("camera closed")

        def close(self):
            self.close_count += 1
            self.closed.set()

    transport = _Transport()
    engine = _Engine(decision)
    runtime = ResidentActRuntime(
        transport=transport,
        engine=engine,
        monotonic_ns=iter(range(2_100, 3_000, 100)).__next__,
    )
    camera = _Camera()
    warmup_count = 0

    def warmup():
        nonlocal warmup_count
        warmup_count += 1
        warmup_done.set()

    worker = ResidentActWorker(
        runtime=runtime,
        camera=camera,
        warmup=warmup,
        connect_timeout_s=0.5,
    )
    thread = threading.Thread(target=worker.run)
    caplog.set_level("INFO", logger="excavator_il.resident_act_runtime")
    thread.start()
    try:
        assert worker.wait_ready(timeout_s=0.5), repr(worker.error)
        deadline = time.monotonic() + 0.5
        while len(transport.sent) < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(transport.sent) == 3
        assert runtime.status.completed_steps == 3
        assert thread.is_alive()
    finally:
        worker.request_stop()
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert warmup_count == 1
    assert camera.read_count >= 1
    assert camera.close_count == 1
    assert worker.error is None
    lifecycle_output = capsys.readouterr().out
    assert "ACT resident warmup starting" in lifecycle_output
    assert "ACT resident warmup passed" in lifecycle_output
    assert "ACT resident worker ready:" in lifecycle_output
    assert "ACT resident worker stopped" in caplog.text


def test_slow_inference_consumes_latest_state_without_replaying_socket_backlog():
    decision = ActRuntimeDecision(
        predicted_action=(0.1, -0.2, 0.3, 0.0),
        commanded_action=(0.1, -0.2, 0.3, 0.0),
        serial_axes=(0.0,) * 6,
        reason="motion_allowed",
    )
    inference_started = threading.Event()
    release_inference = threading.Event()
    latest_delivered = threading.Event()
    gap_candidate_sent = threading.Event()

    class _BlockingEngine(_Engine):
        def step(self, *, observation, telemetry):
            self.observations.append((observation, telemetry))
            if len(self.observations) == 1:
                inference_started.set()
                assert release_inference.wait(0.5)
            return self.decision

    def sequenced_state(sequence):
        base = _state()
        return _state(
            state=(float(sequence),) + base.state[1:],
            sensor_seq=sequence,
            state_monotonic_ns=1_800 + sequence,
            receive_monotonic_ns=1_900 + sequence,
        )

    class _Transport(_CandidateTransport):
        def __init__(self):
            super().__init__()
            self.connected = False
            self.closed = threading.Event()
            self.index = 0

        def connect(self, *, timeout_s):
            self.connected = True

        def receive_state(self, *, timeout_s):
            sequence = (11, 12, 13, 14, 15)
            if self.index >= len(sequence):
                self.closed.wait(timeout_s)
                return None
            if self.index == 1:
                assert inference_started.wait(0.5)
            if self.index == 4:
                assert gap_candidate_sent.wait(0.5)
            state = sequenced_state(sequence[self.index])
            self.index += 1
            if state.sensor_seq == 14:
                latest_delivered.set()
            return state

        def send_candidate(self, candidate):
            super().send_candidate(candidate)
            if len(self.sent) == 2:
                gap_candidate_sent.set()

        def close(self):
            self.connected = False
            self.closed.set()

    class _Camera:
        def __init__(self):
            self.closed = threading.Event()
            self.once = False

        def read_rgb(self):
            if not self.once:
                self.once = True
                return RgbCameraFrame(
                    1_000,
                    np.zeros((2, 3, 3), dtype=np.uint8),
                )
            self.closed.wait()
            raise RuntimeError("camera closed")

        def close(self):
            self.closed.set()

    transport = _Transport()
    engine = _BlockingEngine(decision)
    telemetry = _TelemetryPreview()
    runtime = ResidentActRuntime(
        transport=transport,
        engine=engine,
        telemetry_preview=telemetry,
        monotonic_ns=iter(range(3_000, 5_000, 100)).__next__,
    )
    worker = ResidentActWorker(
        runtime=runtime,
        camera=_Camera(),
        warmup=lambda: None,
        connect_timeout_s=0.5,
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        assert worker.wait_ready(timeout_s=0.5), repr(worker.error)
        assert inference_started.wait(0.5)
        assert latest_delivered.wait(0.5)
        release_inference.set()
        deadline = time.monotonic() + 0.5
        while len(transport.sent) < 3 and time.monotonic() < deadline:
            time.sleep(0.005)

        assert len(transport.sent) == 3
        assert transport.sent[1].action == (0.0, 0.0, 0.0, 0.0)
        assert [item[0].state[0] for item in engine.observations] == [11.0, 15.0]
        assert [values["sensor_seq"] for values, _ in telemetry.published] == [11, 12, 13, 14, 15]
    finally:
        release_inference.set()
        worker.request_stop()
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert worker.error is None


def test_owner_disconnect_resets_policy_and_exits_the_worker_cleanly():
    class _Transport(_CandidateTransport):
        connected = False

        def connect(self, *, timeout_s):
            self.connected = True

        def receive_state(self, *, timeout_s):
            raise ResidentActOwnerClosed("owner disconnected")

        def close(self):
            self.connected = False

    class _Camera:
        def __init__(self):
            self.closed = threading.Event()
            self.once = False

        def read_rgb(self):
            if not self.once:
                self.once = True
                return RgbCameraFrame(
                    1_000,
                    np.zeros((2, 3, 3), dtype=np.uint8),
                )
            self.closed.wait()
            raise RuntimeError("camera closed")

        def close(self):
            self.closed.set()

    engine = _Engine()
    runtime = ResidentActRuntime(transport=_Transport(), engine=engine)
    worker = ResidentActWorker(
        runtime=runtime,
        camera=_Camera(),
        warmup=lambda: None,
        connect_timeout_s=0.5,
    )

    worker.run()

    assert worker.error is None
    assert engine.reset_count == 1
    assert runtime.status.active_generation is None


def test_worker_factory_loads_the_model_and_opens_the_camera_exactly_once(
    monkeypatch, capsys
):
    calls = {"load": 0, "camera": [], "verify": 0, "preview": []}

    class _Policy:
        def __init__(self):
            self.config = SimpleNamespace(
                chunk_size=20,
                n_action_steps=10,
                temporal_ensemble_coeff=None,
                input_features={
                    "observation.state": SimpleNamespace(shape=(11,)),
                    "observation.images.front": SimpleNamespace(shape=(3, 480, 640)),
                },
                output_features={"action": SimpleNamespace(shape=(4,))},
                device="cpu",
            )

        def to(self, _device):
            return self

        def eval(self):
            return self

        def reset(self):
            return None

    policy = _Policy()

    class _PolicyClass:
        @staticmethod
        def from_pretrained(_path):
            calls["load"] += 1
            return policy

    config = SimpleNamespace(
        checkpoint_path=Path("/checkpoint"),
        deployment_manifest_path=Path("/manifest.json"),
        machine_profile_path=Path("/machine.json"),
        device="cuda",
        camera=SimpleNamespace(
            device="/dev/video0", width=640, height=480, nominal_fps=30
        ),
        max_inference_state_age_ms=100.0,
        max_camera_age_ms=120.0,
        max_inference_ms=100.0,
    )
    observation_config = SimpleNamespace(
        camera=SimpleNamespace(
            # Collection uses the stable host path.  The launcher maps this
            # device into the ACT container as config.camera.device.
            device=(
                "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"
            ),
            width=640,
            height=480,
            nominal_fps=30,
            jpeg_quality=83,
        ),
        camera_preview=SimpleNamespace(bind_host="0.0.0.0", port=18092),
        joystick=SimpleNamespace(allowed_pc_host="192.168.50.1"),
    )
    camera = object()
    monkeypatch.setattr(resident_runtime_module, "load_act_runtime_config", lambda _: config)
    monkeypatch.setattr(
        resident_runtime_module,
        "load_collection_config",
        lambda path: observation_config if path == "/operator.json" else None,
    )
    monkeypatch.setattr(
        resident_runtime_module,
        "verify_deployment_manifest",
        lambda **_kwargs: calls.__setitem__("verify", calls["verify"] + 1),
    )
    monkeypatch.setattr(resident_runtime_module, "get_policy_class", lambda _: _PolicyClass)
    monkeypatch.setattr(
        resident_runtime_module,
        "make_pre_post_processors",
        lambda *_args, **_kwargs: (lambda batch: batch, lambda action: action),
    )

    def make_camera(camera_config):
        calls["camera"].append(camera_config)
        return camera

    monkeypatch.setattr(resident_runtime_module, "UvcCamera", make_camera)
    monkeypatch.setattr(resident_runtime_module, "LatestJpegFrame", lambda: "jpeg")
    monkeypatch.setattr(resident_runtime_module, "LatestTelemetryFrame", lambda: "telemetry")

    def make_preview(frames, **kwargs):
        calls["preview"].append((frames, kwargs))
        return "server"

    monkeypatch.setattr(resident_runtime_module, "MjpegPreviewServer", make_preview)

    worker = build_resident_act_worker(
        "/config.json",
        socket_path="/tmp/resident-act-test.sock",
        operator_observation_config="/operator.json",
    )

    assert isinstance(worker, ResidentActWorker)
    assert worker.runtime.status.completed_steps == 0
    assert calls["load"] == 1
    assert calls["verify"] == 2
    assert len(calls["camera"]) == 1
    assert calls["camera"][0].jpeg_quality == 83
    assert calls["preview"] == [
        (
            "jpeg",
            {
                "telemetry": "telemetry",
                "bind_host": "0.0.0.0",
                "port": 18092,
                "allowed_client_host": "192.168.50.1",
            },
        )
    ]
    lifecycle_output = capsys.readouterr().out
    assert "ACT resident build: policy load starting" in lifecycle_output
    assert "ACT resident build: policy load passed" in lifecycle_output
    assert "ACT resident build: CUDA transfer passed" in lifecycle_output
    assert "ACT resident build: processors ready" in lifecycle_output
    assert "ACT resident build: deployment recheck passed" in lifecycle_output


def test_worker_factory_rejects_operator_camera_format_mismatch(monkeypatch):
    act_config = SimpleNamespace(
        camera=SimpleNamespace(
            device="/dev/video0", width=640, height=480, nominal_fps=30
        )
    )
    observation_config = SimpleNamespace(
        camera=SimpleNamespace(
            device=(
                "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"
            ),
            width=1280,
            height=720,
            nominal_fps=30,
        ),
        camera_preview=SimpleNamespace(bind_host="0.0.0.0", port=18092),
    )
    monkeypatch.setattr(
        resident_runtime_module, "load_act_runtime_config", lambda _: act_config
    )
    monkeypatch.setattr(
        resident_runtime_module,
        "load_collection_config",
        lambda _: observation_config,
    )

    with pytest.raises(ValueError, match="camera formats must match"):
        build_resident_act_worker(
            "/config.json",
            socket_path="/tmp/resident-act-test.sock",
            operator_observation_config="/operator.json",
        )


def test_worker_factory_default_provider_fails_closed_for_unknown_backend(
    monkeypatch,
):
    config = SimpleNamespace(
        dig_policy_backend="diffusion_policy",
        camera=SimpleNamespace(
            device="/dev/video0", width=640, height=480, nominal_fps=30
        ),
        max_inference_state_age_ms=100.0,
        max_camera_age_ms=120.0,
        max_inference_ms=100.0,
    )
    load_calls = []
    monkeypatch.setattr(
        resident_runtime_module, "load_act_runtime_config", lambda _: config
    )

    def commissioned_loader(_config):
        load_calls.append(_config)
        return object()

    with pytest.raises(ValueError, match="unknown dig policy backend"):
        build_resident_act_worker(
            "/config.json",
            socket_path="/tmp/resident-act-test.sock",
            commissioned_lerobot_act_loader=commissioned_loader,
        )

    assert load_calls == []


def test_worker_factory_uses_injected_provider_without_commissioned_act_loader(
    monkeypatch,
):
    class _AlternatePolicy:
        descriptor = DigPolicyDescriptor(
            backend_id="diffusion_policy",
            implementation="tests.AlternatePolicy",
        )

        def select_action(self, observation):
            return (0.0, 0.0, 0.0, 0.0)

        def warmup(self):
            return (0.0, 0.0, 0.0, 0.0)

        def reset(self):
            return None

    config = SimpleNamespace(
        dig_policy_backend="diffusion_policy",
        camera=SimpleNamespace(
            device="/dev/video0", width=640, height=480, nominal_fps=30
        ),
        max_inference_state_age_ms=100.0,
        max_camera_age_ms=120.0,
        max_inference_ms=100.0,
    )
    monkeypatch.setattr(
        resident_runtime_module, "load_act_runtime_config", lambda _: config
    )
    monkeypatch.setattr(
        resident_runtime_module,
        "ResidentActDataClient",
        lambda socket_path: SimpleNamespace(socket_path=socket_path, close=lambda: None),
    )
    monkeypatch.setattr(
        resident_runtime_module,
        "UvcCamera",
        lambda camera_config: SimpleNamespace(camera_config=camera_config, close=lambda: None),
    )

    worker = build_resident_act_worker(
        "/config.json",
        socket_path="/tmp/resident-act-test.sock",
        dig_policy_provider=lambda loaded_config: DigPolicyFactory(
            {
                "diffusion_policy": lambda: _AlternatePolicy(),
            }
        ),
        commissioned_lerobot_act_loader=lambda _config: pytest.fail(
            "commissioned ACT loader must not run for alternate provider"
        ),
    )

    assert isinstance(worker, ResidentActWorker)
    assert worker.runtime.status.completed_steps == 0


def test_worker_factory_rejects_injected_provider_descriptor_mismatch(
    monkeypatch,
):
    class _MismatchedPolicy:
        descriptor = DigPolicyDescriptor(
            backend_id="lerobot_act",
            implementation="tests.MismatchedPolicy",
        )

        def select_action(self, observation):
            return (0.0, 0.0, 0.0, 0.0)

        def warmup(self):
            return (0.0, 0.0, 0.0, 0.0)

        def reset(self):
            return None

    config = SimpleNamespace(
        dig_policy_backend="diffusion_policy",
        camera=SimpleNamespace(
            device="/dev/video0", width=640, height=480, nominal_fps=30
        ),
        max_inference_state_age_ms=100.0,
        max_camera_age_ms=120.0,
        max_inference_ms=100.0,
    )
    monkeypatch.setattr(
        resident_runtime_module, "load_act_runtime_config", lambda _: config
    )

    with pytest.raises(ValueError, match="descriptor backend_id"):
        build_resident_act_worker(
            "/config.json",
            socket_path="/tmp/resident-act-test.sock",
            dig_policy_provider=lambda loaded_config: DigPolicyFactory(
                {
                    "diffusion_policy": lambda: _MismatchedPolicy(),
                }
            ),
            commissioned_lerobot_act_loader=lambda _config: pytest.fail(
                "commissioned ACT loader must not run for injected providers"
            ),
        )


def test_runner_builds_once_and_runs_the_resident_worker(monkeypatch):
    calls = []

    class _Worker:
        def run(self):
            calls.append("run")

    monkeypatch.setattr(
        resident_runtime_module,
        "build_resident_act_worker",
        lambda config_path, **kwargs: (
            calls.append((config_path, kwargs)),
            _Worker(),
        )[1],
    )

    run_resident_act_worker(
        "/config.json",
        socket_path="/run/act.sock",
        operator_observation_config="/operator.json",
    )

    assert calls == [
        (
            "/config.json",
            {
                "socket_path": "/run/act.sock",
                "operator_observation_config": "/operator.json",
            },
        ),
        "run",
    ]


def test_module_cli_wires_the_operator_observation_config(monkeypatch):
    calls = []

    class _Worker:
        def run(self):
            calls.append("run")

        def request_stop(self):
            calls.append("stop")

    monkeypatch.setattr(
        resident_runtime_module,
        "build_resident_act_worker",
        lambda config_path, **kwargs: (
            calls.append((config_path, kwargs)),
            _Worker(),
        )[1],
    )
    monkeypatch.setattr(resident_runtime_module.signal, "signal", lambda *_args: None)

    result = resident_runtime_module.main(
        [
            "--config",
            "/config.json",
            "--socket-path",
            "/run/act.sock",
            "--operator-observation-config",
            "/operator.json",
        ]
    )

    assert result == 0
    assert calls == [
        (
            "/config.json",
            {
                "socket_path": "/run/act.sock",
                "operator_observation_config": "/operator.json",
            },
        ),
        "run",
    ]

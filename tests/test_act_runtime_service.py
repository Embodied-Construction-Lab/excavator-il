import json

import numpy as np
import pytest

from excavator_il.act_runtime import (
    ActRuntimeController,
    ActRuntimeDecision,
    ActObservation,
    CausalObservationBuffer,
    RuntimeMode,
)
from excavator_il.act_runtime_service import (
    ActRuntimeStepProcessor,
    LatestStateQueue,
    SensorSequenceTracker,
    Stm32CommandChannel,
    _apply_operator_datagram,
    _route_telemetry_frame,
    _startup_stm32,
)
from excavator_il.act_deployment import verify_deployment_manifest
from excavator_il.collector.camera import RgbCameraFrame
from excavator_il.stm32_protocol import (
    STM32_TELEMETRY_FIELDS,
    Stm32ManualCommandEncoder,
    Stm32TelemetryFrame,
)


class _Engine:
    def __init__(self, decision):
        self.decision = decision

    def step(self, **_kwargs):
        return self.decision

    def reset(self):
        pass


class _Serial:
    def __init__(self):
        self.writes = []

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def flush(self):
        pass


class _StartupSerial(_Serial):
    def __init__(self, post_reset_lines):
        super().__init__()
        self._post_reset_lines = iter(post_reset_lines)
        self.reset_count = 0

    def reset_input_buffer(self):
        self.reset_count += 1

    def readline(self):
        return next(self._post_reset_lines, b"")


def _telemetry(stamp=1_000_000_000):
    values = {field: 0 for field in STM32_TELEMETRY_FIELDS}
    values.update(
        schema_version="stm32_control_telemetry.v2",
        sensor_is_new=1,
        boom_pos_mm=1500.0,
        stick_pos_mm=1600.0,
        bucket_pos_mm=1700.0,
        rs485_ok=1,
        dwj_ok=1,
        imu_ok=1,
        control_enabled=1,
    )
    return Stm32TelemetryFrame(receive_monotonic_ns=stamp, values=values)


def _telemetry_line(**overrides):
    frame = _telemetry()
    values = dict(frame.values)
    values.update(overrides)
    return (
        ",".join(str(values[field]) for field in STM32_TELEMETRY_FIELDS) + "\n"
    ).encode("ascii")


def test_motion_startup_drains_backlog_and_waits_for_exact_zero_ack():
    serial = _StartupSerial(
        [
            _telemetry_line(command_rx_seq=41, command_valid=1),
            _telemetry_line(
                command_rx_seq=42,
                command_valid=1,
                command_timed_out=0,
                command_action_boom=0.0,
                command_action_stick=0.0,
                command_action_bucket=0.0,
                command_action_swing=0.0,
            ),
        ]
    )
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )

    next_sequence = _startup_stm32(
        serial_port=serial,
        command_channel=channel,
        mode=RuntimeMode.MOTION,
        timeout_s=0.1,
    )

    assert serial.reset_count == 1
    assert next_sequence == 42
    assert json.loads(serial.writes[0])["command_seq"] == 42


def test_motion_startup_refuses_ready_without_zero_ack():
    serial = _StartupSerial(
        [_telemetry_line(command_rx_seq=41, command_valid=1)]
    )
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )

    with pytest.raises(RuntimeError, match="zero-command ACK"):
        _startup_stm32(
            serial_port=serial,
            command_channel=channel,
            mode=RuntimeMode.MOTION,
            timeout_s=0.01,
        )


def test_shadow_startup_never_writes_or_requires_command_ack():
    serial = _StartupSerial(
        [_telemetry_line(command_rx_seq=41, command_valid=1)]
    )
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.SHADOW,
    )

    assert _startup_stm32(
        serial_port=serial,
        command_channel=channel,
        mode=RuntimeMode.SHADOW,
        timeout_s=0.1,
    ) == 42
    assert serial.writes == []


def test_sensor_sequence_tracker_marks_uart_gaps_and_wraps_cleanly():
    tracker = SensorSequenceTracker()

    assert tracker.observe(10) == 0
    assert tracker.observe(11) == 0
    assert tracker.observe(13) == 1
    tracker = SensorSequenceTracker()
    assert tracker.observe(0xFFFFFFFF) == 0
    assert tracker.observe(0) == 0


def test_motion_manifest_rejects_wrong_action_order(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_text(
        json.dumps(
            {
                "schema_version": "0.3.0",
                "machine_id": "scale_excavator_v1",
                "action_order": ["boom", "stick", "bucket", "swing"],
            }
        ),
        encoding="utf-8",
    )
    from hashlib import sha256

    manifest = tmp_path / "deployment.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "excavator_act_deployment.v1",
                "checkpoint": {
                    "files_sha256": {
                        "model.safetensors": sha256(b"model").hexdigest()
                    },
                    "selected": True,
                    "selection_reason": "lowest safe validation deployment-prior L1",
                },
                "evaluation": {
                    "validation_frame_count": 3,
                    "deployment_prior_l1": 0.1,
                    "max_deployment_prior_l1": 0.2,
                    "action_min": -0.5,
                    "action_max": 0.5,
                    "all_finite": True,
                    "out_of_range_sample_count": 0,
                },
                    "data": {
                        "pipeline_validation_present": False,
                        "source_dataset_sha256": "c" * 64,
                        "train_dataset_sha256": "a" * 64,
                    "validation_dataset_sha256": "b" * 64,
                },
                "contract": {
                    "action_order": ["swing", "boom", "stick", "bucket"],
                    "action_fields": [
                        "action_boom",
                        "action_stick",
                        "action_bucket",
                        "action_swing",
                    ],
                    "state_fields": ["placeholder"],
                    "state_dim": 11,
                    "action_dim": 4,
                    "front_rgb_chw": [3, 480, 640],
                    "chunk_size": 20,
                    "n_action_steps": 10,
                },
                "machine_profile_sha256": sha256(
                    machine_profile.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action order"):
        verify_deployment_manifest(
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            machine_profile_path=machine_profile,
        )


def test_motion_manifest_rejects_unsafe_evaluation(tmp_path):
    from hashlib import sha256
    from excavator_il.lerobot_conversion import STATE_FIELDS

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_text(
        json.dumps({"action_order": ["boom", "stick", "bucket", "swing"]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "deployment.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "excavator_act_deployment.v1",
                "checkpoint": {
                    "selected": True,
                    "selection_reason": "lowest safe validation deployment-prior L1",
                    "files_sha256": {
                        "model.safetensors": sha256(b"model").hexdigest()
                    },
                },
                "evaluation": {
                    "validation_frame_count": 3,
                    "deployment_prior_l1": 0.1,
                    "max_deployment_prior_l1": 0.2,
                    "action_min": -0.5,
                    "action_max": 1.1,
                    "all_finite": True,
                    "out_of_range_sample_count": 1,
                },
                "data": {
                    "pipeline_validation_present": False,
                    "source_dataset_sha256": "c" * 64,
                    "train_dataset_sha256": "a" * 64,
                    "validation_dataset_sha256": "b" * 64,
                },
                "contract": {
                    "action_order": ["boom", "stick", "bucket", "swing"],
                    "action_fields": ["action_boom", "action_stick", "action_bucket", "action_swing"],
                    "state_fields": list(STATE_FIELDS),
                    "state_dim": 11,
                    "action_dim": 4,
                    "front_rgb_chw": [3, 480, 640],
                    "chunk_size": 20,
                    "n_action_steps": 10,
                },
                "machine_profile_sha256": sha256(machine_profile.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation"):
        verify_deployment_manifest(
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            machine_profile_path=machine_profile,
        )


def test_shadow_step_never_calls_serial_write_and_logs_prediction():
    serial = _Serial()
    records = []
    buffer = CausalObservationBuffer()
    buffer.add_camera(
        RgbCameraFrame(
            capture_monotonic_ns=990_000_000,
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        )
    )
    processor = ActRuntimeStepProcessor(
        observation_buffer=buffer,
        engine=_Engine(
            ActRuntimeDecision(
                predicted_action=(0.1, -0.2, 0.3, -0.4),
                commanded_action=(0.0,) * 4,
                serial_axes=None,
                reason="shadow_mode",
            )
        ),
        command_channel=Stm32CommandChannel(
            serial_port=serial,
            encoder=Stm32ManualCommandEncoder(),
            mode=RuntimeMode.SHADOW,
        ),
        operator_snapshot=lambda: (False, None),
        record=records.append,
        monotonic_ns=lambda: 1_010_000_000,
    )

    decision = processor.process(_telemetry())

    assert decision.reason == "shadow_mode"
    assert serial.writes == []
    assert records[0]["predicted_action"] == [0.1, -0.2, 0.3, -0.4]
    assert records[0]["serial_write_attempted"] is False


def test_motion_fail_closed_step_writes_explicit_zero_command():
    serial = _Serial()
    buffer = CausalObservationBuffer()
    buffer.add_camera(
        RgbCameraFrame(
            capture_monotonic_ns=990_000_000,
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        )
    )
    processor = ActRuntimeStepProcessor(
        observation_buffer=buffer,
        engine=_Engine(
            ActRuntimeDecision(
                predicted_action=(0.1, -0.2, 0.3, -0.4),
                commanded_action=(0.0,) * 4,
                serial_axes=(0.0,) * 6,
                reason="operator_disabled",
            )
        ),
        command_channel=(
            channel := Stm32CommandChannel(
                serial_port=serial,
                encoder=Stm32ManualCommandEncoder(),
                mode=RuntimeMode.MOTION,
            )
        ),
        operator_snapshot=lambda: (False, None),
        record=lambda _value: None,
        monotonic_ns=lambda: 1_010_000_000,
    )
    channel.synchronize(_telemetry())

    processor.process(_telemetry())

    assert len(serial.writes) == 1
    command = json.loads(serial.writes[0])
    assert [command[name] for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2")] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_step_log_records_final_boundary_zero_instead_of_requested_motion():
    serial = _Serial()
    records = []
    buffer = CausalObservationBuffer()
    buffer.add_camera(
        RgbCameraFrame(
            capture_monotonic_ns=990_000_000,
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        )
    )
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )
    frame = _telemetry()
    channel.synchronize(frame)
    generation = channel.update_state(frame)
    processor = ActRuntimeStepProcessor(
        observation_buffer=buffer,
        engine=_Engine(
            ActRuntimeDecision(
                predicted_action=(0.1, -0.2, 0.3, -0.4),
                commanded_action=(0.1, -0.2, 0.3, -0.4),
                serial_axes=(-0.4, -0.2, 0.0, 0.3, 0.1, 0.0),
                reason="motion_allowed",
            )
        ),
        command_channel=channel,
        operator_snapshot=lambda: (True, 1_000_000_000),
        record=records.append,
        monotonic_ns=lambda: 1_010_000_000,
    )

    processor.process(frame, state_generation=generation)

    assert records[0]["requested_serial_axes"] == [-0.4, -0.2, 0.0, 0.3, 0.1, 0.0]
    assert records[0]["effective_serial_axes"] == [0.0] * 6
    assert records[0]["final_gate_reason"] == "operator_not_fresh"
    assert records[0]["command_seq"] == 0


def test_release_and_repress_between_steps_resets_old_act_chunk():
    class _ResetEngine(_Engine):
        def __init__(self, decision):
            super().__init__(decision)
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    serial = _Serial()
    frame = _telemetry()
    buffer = CausalObservationBuffer()
    buffer.add_camera(RgbCameraFrame(990_000_000, np.zeros((480, 640, 3), np.uint8)))
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )
    channel.synchronize(frame)
    generation = channel.update_state(frame)
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)
    engine = _ResetEngine(
        ActRuntimeDecision(
            predicted_action=(0.1,) * 4,
            commanded_action=(0.1,) * 4,
            serial_axes=(0.1, 0.1, 0.0, 0.1, 0.1, 0.0),
            reason="motion_allowed",
        )
    )
    processor = ActRuntimeStepProcessor(
        observation_buffer=buffer,
        engine=engine,
        command_channel=channel,
        operator_snapshot=lambda: (True, 1_020_000_000),
        record=lambda _value: None,
        monotonic_ns=lambda: 1_030_000_000,
    )
    channel.update_operator(False, receive_monotonic_ns=1_010_000_000)
    channel.update_operator(True, receive_monotonic_ns=1_020_000_000)

    processor.process(frame, state_generation=generation)

    assert engine.reset_count == 1


def test_release_during_inference_downgrades_final_write_and_resets_chunk():
    serial = _Serial()
    frame = _telemetry()
    buffer = CausalObservationBuffer()
    buffer.add_camera(RgbCameraFrame(990_000_000, np.zeros((480, 640, 3), np.uint8)))
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )
    channel.synchronize(frame)
    generation = channel.update_state(frame)
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)

    class _ReleaseEngine:
        def __init__(self):
            self.reset_count = 0

        def step(self, **_kwargs):
            channel.update_operator(False, receive_monotonic_ns=1_005_000_000)
            channel.update_operator(True, receive_monotonic_ns=1_006_000_000)
            return ActRuntimeDecision(
                predicted_action=(0.1,) * 4,
                commanded_action=(0.1,) * 4,
                serial_axes=(0.1, 0.1, 0.0, 0.1, 0.1, 0.0),
                reason="motion_allowed",
            )

        def reset(self):
            self.reset_count += 1

    engine = _ReleaseEngine()
    records = []
    processor = ActRuntimeStepProcessor(
        observation_buffer=buffer,
        engine=engine,
        command_channel=channel,
        operator_snapshot=lambda: (True, 1_006_000_000),
        record=records.append,
        monotonic_ns=lambda: 1_010_000_000,
    )

    processor.process(frame, state_generation=generation)

    assert engine.reset_count == 1
    assert records[0]["final_gate_reason"] == "motion_interrupted"
    assert records[0]["effective_serial_axes"] == [0.0] * 6


def test_shadow_command_channel_never_writes_during_full_lifecycle():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.SHADOW,
    )

    channel.synchronize(_telemetry())
    channel.safe_zero(monotonic_ns=1_000, reason="startup")
    channel.write_axes((1.0, -1.0, 0.0, 0.5, -0.5, 0.0), monotonic_ns=2_000)
    channel.safe_zero(monotonic_ns=3_000, reason="shutdown")

    assert serial.writes == []


def test_string_shadow_mode_is_normalized_and_cannot_write():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode="shadow",
    )
    channel.synchronize(_telemetry())

    channel.write_axes(
        (1.0, 1.0, 0.0, 1.0, 1.0, 0.0),
        monotonic_ns=1_000,
        state_generation=1,
    )

    assert serial.writes == []


def test_nonzero_command_without_state_generation_fails_closed():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )
    channel.synchronize(_telemetry())
    channel.update_operator(True, receive_monotonic_ns=1_000_000)

    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0), monotonic_ns=1_010_000
    )

    command = json.loads(serial.writes[-1])
    assert all(command[name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def test_motion_command_channel_resumes_sequence_and_zeros_start_and_shutdown():
    serial = _Serial()
    encoder = Stm32ManualCommandEncoder()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=encoder,
        mode=RuntimeMode.MOTION,
    )
    frame = _telemetry()
    values = dict(frame.values)
    values.update(command_rx_seq=41, command_timed_out=1)

    channel.synchronize(
        Stm32TelemetryFrame(frame.receive_monotonic_ns, values)
    )
    channel.update_operator(True, receive_monotonic_ns=900_000)
    channel.safe_zero(monotonic_ns=1_000_000, reason="startup")
    channel.write_axes((0.1, 0.2, 0.0, 0.3, 0.4, 0.0), monotonic_ns=2_000_000)
    channel.safe_zero(monotonic_ns=3_000_000, reason="shutdown")

    commands = [json.loads(payload) for payload in serial.writes]
    assert [command["command_seq"] for command in commands] == [42, 43, 44]
    assert [commands[0][name] for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2")] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert [commands[2][name] for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2")] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_motion_channel_deadman_release_immediately_writes_zero():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
        max_operator_age_ms=150,
    )
    channel.synchronize(_telemetry())
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)
    generation = channel.update_state(_telemetry())
    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0),
        monotonic_ns=1_010_000_000,
        state_generation=generation,
    )

    channel.update_operator(False, receive_monotonic_ns=1_020_000_000)

    commands = [json.loads(payload) for payload in serial.writes]
    assert len(commands) == 2
    assert any(commands[0][name] != 0.0 for name in ("X1", "Y1", "X2", "Y2"))
    assert all(commands[1][name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def test_terminal_disarm_zero_cannot_be_overwritten_by_late_inference():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
        max_operator_age_ms=150,
    )
    channel.synchronize(_telemetry())
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)

    channel.terminal_disarm(monotonic_ns=1_010_000_000, reason="shutdown")
    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0), monotonic_ns=1_020_000_000
    )

    commands = [json.loads(payload) for payload in serial.writes]
    assert len(commands) == 1
    assert all(commands[0][name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def test_operator_timeout_immediately_zeros_once_without_waiting_for_new_state():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
        max_operator_age_ms=150,
    )
    channel.synchronize(_telemetry())
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)
    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0), monotonic_ns=1_010_000_000
    )

    assert channel.enforce_operator_timeout(monotonic_ns=1_151_000_000) is True
    assert channel.enforce_operator_timeout(monotonic_ns=1_200_000_000) is False

    commands = [json.loads(payload) for payload in serial.writes]
    assert len(commands) == 2
    assert all(commands[-1][name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def test_latest_state_queue_drops_backlog_instead_of_bursting_old_states():
    queue = LatestStateQueue()
    queue.put(_telemetry(1_000))
    queue.put(_telemetry(2_000))
    queue.put(_telemetry(3_000))

    latest, dropped = queue.get(timeout_s=0.01)

    assert latest.receive_monotonic_ns == 3_000
    assert dropped == 2
    assert queue.dropped_count == 2


def test_step_processor_resets_policy_queue_and_zeros_on_state_gap():
    class _GapEngine:
        def __init__(self):
            self.reset_count = 0
            self.step_count = 0

        def reset(self):
            self.reset_count += 1

        def step(self, **_kwargs):
            self.step_count += 1
            return ActRuntimeDecision(
                predicted_action=(0.1, -0.2, 0.3, -0.4),
                commanded_action=(0.1, -0.2, 0.3, -0.4),
                serial_axes=(-0.4, -0.2, 0.0, 0.3, 0.1, 0.0),
                reason="motion_allowed",
            )

    serial = _Serial()
    engine = _GapEngine()
    buffer = CausalObservationBuffer()
    buffer.add_camera(
        RgbCameraFrame(
            capture_monotonic_ns=990_000_000,
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        )
    )
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )
    channel.synchronize(_telemetry())
    processor = ActRuntimeStepProcessor(
        observation_buffer=buffer,
        engine=engine,
        command_channel=channel,
        operator_snapshot=lambda: (True, 1_000_000_000),
        record=lambda _value: None,
        monotonic_ns=lambda: 1_010_000_000,
    )

    decision = processor.process(_telemetry(), dropped_state_count=1)

    assert decision.reason == "state_gap"
    assert decision.commanded_action == (0.0,) * 4
    assert engine.reset_count == 1
    assert engine.step_count == 0


def test_state_silence_immediately_zeros_without_waiting_for_next_frame():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
        max_operator_age_ms=150,
        max_state_age_ms=100,
    )
    channel.synchronize(_telemetry())
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)
    channel.update_state(_telemetry(1_000_000_000))
    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0), monotonic_ns=1_010_000_000
    )

    assert channel.enforce_state_timeout(monotonic_ns=1_101_000_000) is True
    assert channel.enforce_state_timeout(monotonic_ns=1_150_000_000) is False

    commands = [json.loads(payload) for payload in serial.writes]
    assert len(commands) == 2
    assert all(commands[-1][name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def test_late_inference_cannot_overwrite_state_watchdog_zero():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
        max_operator_age_ms=150,
        max_state_age_ms=200,
    )
    frame = _telemetry()
    channel.synchronize(frame)
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)
    generation = channel.update_state(frame)

    channel.enforce_state_timeout(monotonic_ns=1_201_000_000)
    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0),
        monotonic_ns=1_202_000_000,
        state_generation=generation,
    )

    commands = [json.loads(payload) for payload in serial.writes]
    assert len(commands) == 2
    assert all(
        command[name] == 0.0
        for command in commands
        for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2")
    )


def test_new_unsafe_state_invalidates_inflight_inference_at_write_boundary():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
        max_operator_age_ms=150,
        max_state_age_ms=200,
    )
    frame = _telemetry()
    channel.synchronize(frame)
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)
    generation = channel.update_state(frame)
    unsafe_values = dict(frame.values)
    unsafe_values["estop"] = 1
    channel.update_state(
        Stm32TelemetryFrame(receive_monotonic_ns=1_020_000_000, values=unsafe_values)
    )

    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0),
        monotonic_ns=1_030_000_000,
        state_generation=generation,
    )

    command = json.loads(serial.writes[-1])
    assert all(command[name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def test_unsafe_state_immediately_zeros_without_waiting_for_inference():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )
    frame = _telemetry()
    channel.synchronize(frame)
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)
    generation = channel.update_state(frame)
    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0),
        monotonic_ns=1_010_000_000,
        state_generation=generation,
    )
    unsafe = dict(frame.values)
    unsafe["fault_flags"] = 1

    channel.update_state(
        Stm32TelemetryFrame(receive_monotonic_ns=1_020_000_000, values=unsafe)
    )

    commands = [json.loads(payload) for payload in serial.writes]
    assert len(commands) == 2
    assert all(commands[-1][name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def test_nonnew_telemetry_fault_immediately_zeros_without_enqueuing_inference():
    import queue as queue_module

    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )
    frame = _telemetry()
    channel.synchronize(frame)
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)
    generation = channel.update_state(frame)
    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0),
        monotonic_ns=1_010_000_000,
        state_generation=generation,
    )
    values = dict(frame.values)
    values.update(sensor_is_new=0, control_enabled=0, fault_flags=4)
    states = LatestStateQueue()

    _route_telemetry_frame(
        frame=Stm32TelemetryFrame(1_020_000_000, values),
        command_channel=channel,
        states=states,
        sensor_sequences=SensorSequenceTracker(),
    )

    with pytest.raises(queue_module.Empty):
        states.get(timeout_s=0.001)
    commands = [json.loads(payload) for payload in serial.writes]
    assert all(commands[-1][name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def test_safe_nonnew_telemetry_heartbeat_does_not_invalidate_inflight_inference():
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
        max_operator_age_ms=150,
        max_state_age_ms=250,
    )
    frame = _telemetry(1_000_000_000)
    channel.synchronize(frame)
    channel.update_operator(True, receive_monotonic_ns=1_000_000_000)
    generation = channel.update_state(frame)
    heartbeat_values = dict(frame.values)
    heartbeat_values["sensor_is_new"] = 0

    same_generation = channel.update_state(
        Stm32TelemetryFrame(1_050_000_000, heartbeat_values)
    )
    result = channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0),
        monotonic_ns=1_070_000_000,
        state_generation=generation,
    )

    assert same_generation == generation
    assert result.final_gate_reason == "accepted"
    command = json.loads(serial.writes[-1])
    assert any(command[name] != 0.0 for name in ("X1", "Y1", "X2", "Y2"))


def test_rejected_new_operator_session_immediately_disarms_existing_motion():
    from excavator_il.act_runtime import OperatorDeadmanGate

    gate = OperatorDeadmanGate(
        allowed_pc_host="192.168.31.219",
        expected_device_ids=("left", "right"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )
    frame = _telemetry()
    channel.synchronize(frame)
    generation = channel.update_state(frame)
    source = ("192.168.31.219", 40000)
    for session, sequence, deadman, stamp in (
        ("session-a", 0, False, 1_000_000_000),
        ("session-a", 1, True, 1_010_000_000),
    ):
        result = _apply_operator_datagram(
            operator_gate=gate,
            command_channel=channel,
            datagram=_operator_datagram(session, sequence, deadman),
            source=source,
            receive_monotonic_ns=stamp,
        )
        assert json.loads(result)["accepted"] is True
    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0),
        monotonic_ns=1_020_000_000,
        state_generation=generation,
    )

    rejected = _apply_operator_datagram(
        operator_gate=gate,
        command_channel=channel,
        datagram=_operator_datagram("session-b", 0, True),
        source=source,
        receive_monotonic_ns=1_030_000_000,
    )

    assert json.loads(rejected)["accepted"] is False
    commands = [json.loads(payload) for payload in serial.writes]
    assert all(commands[-1][name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def test_malformed_packet_from_allowed_pc_immediately_disarms_motion():
    from excavator_il.act_runtime import OperatorDeadmanGate

    gate = OperatorDeadmanGate(
        allowed_pc_host="192.168.31.219",
        expected_device_ids=("left", "right"),
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )
    serial = _Serial()
    channel = Stm32CommandChannel(
        serial_port=serial,
        encoder=Stm32ManualCommandEncoder(),
        mode=RuntimeMode.MOTION,
    )
    frame = _telemetry()
    channel.synchronize(frame)
    generation = channel.update_state(frame)
    source = ("192.168.31.219", 40000)
    for sequence, deadman, stamp in (
        (0, False, 1_000_000_000),
        (1, True, 1_010_000_000),
    ):
        _apply_operator_datagram(
            operator_gate=gate,
            command_channel=channel,
            datagram=_operator_datagram("session-a", sequence, deadman),
            source=source,
            receive_monotonic_ns=stamp,
        )
    channel.write_axes(
        (0.1, 0.2, 0.0, 0.3, 0.4, 0.0),
        monotonic_ns=1_020_000_000,
        state_generation=generation,
    )

    ack = _apply_operator_datagram(
        operator_gate=gate,
        command_channel=channel,
        datagram=b"not-json",
        source=source,
        receive_monotonic_ns=1_030_000_000,
    )

    assert json.loads(ack)["accepted"] is False
    commands = [json.loads(payload) for payload in serial.writes]
    assert all(commands[-1][name] == 0.0 for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2"))


def _operator_datagram(session, sequence, deadman):
    from excavator_il.joystick_protocol import (
        ControllerIdentity,
        JoystickPacket,
        encode_joystick_packet,
    )

    return encode_joystick_packet(
        JoystickPacket(
            session_id=session,
            sample_seq=sequence,
            pc_sample_monotonic_ns=1,
            pc_sample_wall_ns=2,
            axes=(0.0,) * 6,
            controllers=(
                ControllerIdentity(1, "left", "left", (deadman,)),
                ControllerIdentity(2, "right", "right", (False,)),
            ),
            deadman_pressed=deadman,
            mapping_id="dual_stick.v1",
            calibration_id="raw.v1",
        )
    )

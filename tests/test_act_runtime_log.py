import json
import os
from pathlib import Path
import subprocess

from excavator_il.act_runtime_log import inspect_act_runtime_log
from excavator_il.cli import main


def _step(*, state_ns: int, decision_ns: int) -> dict[str, object]:
    return {
        "schema_version": "excavator_act_runtime_step.v1",
        "state_monotonic_ns": state_ns,
        "camera_monotonic_ns": state_ns - 10_000_000,
        "decision_monotonic_ns": decision_ns,
        "predicted_action": [0.1, -0.2, 0.3, -0.4],
        "commanded_action": [0.0, 0.0, 0.0, 0.0],
        "reason": "shadow_mode",
        "serial_write_attempted": False,
        "requested_serial_axes": None,
        "effective_serial_axes": None,
        "final_gate_reason": None,
        "command_seq": None,
        "serial_write_performed": False,
        "dropped_state_count": 0,
    }


def _write_log(path, events):
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def _command(*, sequence: int, axes: list[float], reason: str) -> dict[str, object]:
    return {
        "schema_version": "excavator_act_runtime_command.v1",
        "command_monotonic_ns": 1_000_000_000 + sequence,
        "command_seq": sequence,
        "serial_axes": axes,
        "reason": reason,
        "serial_write_performed": True,
    }


def test_shadow_log_passes_when_predictions_are_causal_and_serial_is_never_written(
    tmp_path,
):
    log = tmp_path / "act_runtime_shadow.jsonl"
    _write_log(
        log,
        [
            _step(state_ns=1_000_000_000, decision_ns=1_020_000_000),
            _step(state_ns=1_100_000_000, decision_ns=1_120_000_000),
        ],
    )

    report = inspect_act_runtime_log(log, mode="shadow")

    assert report.passed is True
    assert report.step_count == 2
    assert report.command_event_count == 0
    assert report.serial_write_count == 0
    assert report.dropped_state_count == 0
    assert report.estimated_step_rate_hz == 10.0
    assert report.max_state_to_decision_ms == 20.0
    assert report.max_camera_age_ms == 10.0
    assert report.failure_reasons == ()


def test_runtime_log_rejects_step_rate_outside_the_10hz_contract(tmp_path):
    log = tmp_path / "slow_shadow.jsonl"
    _write_log(
        log,
        [
            _step(state_ns=1_000_000_000, decision_ns=1_020_000_000),
            _step(state_ns=1_500_000_000, decision_ns=1_520_000_000),
        ],
    )

    report = inspect_act_runtime_log(log, mode="shadow")

    assert report.passed is False
    assert any("step rate" in reason for reason in report.failure_reasons)


def test_runtime_log_rejects_boolean_or_text_action_values(tmp_path):
    log = tmp_path / "malformed_action.jsonl"
    first = _step(state_ns=1_000_000_000, decision_ns=1_020_000_000)
    first["predicted_action"] = [True, "-0.2", 0.3, -0.4]
    _write_log(
        log,
        [first, _step(state_ns=1_100_000_000, decision_ns=1_120_000_000)],
    )

    report = inspect_act_runtime_log(log, mode="shadow")

    assert report.passed is False
    assert any("predicted_action" in reason for reason in report.failure_reasons)


def test_shadow_log_rejects_serial_result_fields_even_when_write_flags_are_false(
    tmp_path,
):
    log = tmp_path / "shadow_with_serial_result.jsonl"
    first = _step(state_ns=1_000_000_000, decision_ns=1_020_000_000)
    first["requested_serial_axes"] = [0.0] * 6
    _write_log(
        log,
        [first, _step(state_ns=1_100_000_000, decision_ns=1_120_000_000)],
    )

    report = inspect_act_runtime_log(log, mode="shadow")

    assert report.passed is False
    assert any("shadow attempted motion" in reason for reason in report.failure_reasons)


def test_runtime_log_rejects_state_timestamp_regression_hidden_by_valid_endpoints(
    tmp_path,
):
    log = tmp_path / "state_timestamp_regression.jsonl"
    _write_log(
        log,
        [
            _step(state_ns=1_000_000_000, decision_ns=1_020_000_000),
            _step(state_ns=900_000_000, decision_ns=920_000_000),
            _step(state_ns=1_200_000_000, decision_ns=1_220_000_000),
        ],
    )

    report = inspect_act_runtime_log(log, mode="shadow")

    assert report.passed is False
    assert any("strictly increasing" in reason for reason in report.failure_reasons)


def test_motion_log_passes_with_mapped_action_and_terminal_zero(tmp_path):
    log = tmp_path / "act_runtime_motion.jsonl"
    mapped = [-0.4, -0.2, 0.0, 0.3, 0.1, 0.0]
    step = _step(state_ns=1_000_000_000, decision_ns=1_020_000_000)
    step.update(
        commanded_action=[0.1, -0.2, 0.3, -0.4],
        reason="motion_allowed",
        serial_write_attempted=True,
        requested_serial_axes=mapped,
        effective_serial_axes=mapped,
        final_gate_reason="accepted",
        command_seq=42,
        serial_write_performed=True,
    )
    second_step = dict(step)
    second_step.update(
        state_monotonic_ns=1_100_000_000,
        camera_monotonic_ns=1_090_000_000,
        decision_monotonic_ns=1_120_000_000,
        command_seq=43,
    )
    _write_log(
        log,
        [
            _command(sequence=41, axes=[0.0] * 6, reason="act_runtime_startup"),
            _command(sequence=42, axes=mapped, reason="accepted"),
            step,
            _command(sequence=43, axes=mapped, reason="accepted"),
            second_step,
            _command(sequence=44, axes=[0.0] * 6, reason="act_runtime_shutdown"),
        ],
    )

    report = inspect_act_runtime_log(log, mode="motion")

    assert report.passed is True
    assert report.command_event_count == 4
    assert report.serial_write_count == 4
    assert report.nonzero_serial_write_count == 2
    assert report.failure_reasons == ()


def test_motion_log_rejects_wrong_action_axis_order(tmp_path):
    log = tmp_path / "wrong_mapping.jsonl"
    wrong_axes = [0.1, -0.2, 0.0, 0.3, -0.4, 0.0]
    step = _step(state_ns=1_000_000_000, decision_ns=1_020_000_000)
    step.update(
        commanded_action=[0.1, -0.2, 0.3, -0.4],
        reason="motion_allowed",
        serial_write_attempted=True,
        requested_serial_axes=wrong_axes,
        effective_serial_axes=wrong_axes,
        final_gate_reason="accepted",
        command_seq=42,
        serial_write_performed=True,
    )
    _write_log(
        log,
        [
            _command(sequence=41, axes=[0.0] * 6, reason="act_runtime_startup"),
            _command(sequence=42, axes=wrong_axes, reason="accepted"),
            step,
            _command(sequence=43, axes=[0.0] * 6, reason="act_runtime_shutdown"),
        ],
    )

    report = inspect_act_runtime_log(log, mode="motion")

    assert report.passed is False
    assert any("mapping" in reason for reason in report.failure_reasons)


def test_motion_log_rejects_command_event_that_differs_from_step_result(tmp_path):
    log = tmp_path / "command_mismatch.jsonl"
    mapped = [-0.4, -0.2, 0.0, 0.3, 0.1, 0.0]
    step = _step(state_ns=1_000_000_000, decision_ns=1_020_000_000)
    step.update(
        commanded_action=[0.1, -0.2, 0.3, -0.4],
        reason="motion_allowed",
        serial_write_attempted=True,
        requested_serial_axes=mapped,
        effective_serial_axes=mapped,
        final_gate_reason="accepted",
        command_seq=42,
        serial_write_performed=True,
    )
    second_step = dict(step)
    second_step.update(
        state_monotonic_ns=1_100_000_000,
        camera_monotonic_ns=1_090_000_000,
        decision_monotonic_ns=1_120_000_000,
        command_seq=43,
    )
    _write_log(
        log,
        [
            _command(sequence=41, axes=[0.0] * 6, reason="act_runtime_startup"),
            _command(sequence=42, axes=[0.2] * 6, reason="accepted"),
            step,
            _command(sequence=43, axes=mapped, reason="accepted"),
            second_step,
            _command(sequence=44, axes=[0.0] * 6, reason="act_runtime_shutdown"),
        ],
    )

    report = inspect_act_runtime_log(log, mode="motion")

    assert report.passed is False
    assert any("command event differs" in reason for reason in report.failure_reasons)


def test_motion_log_accepts_a_final_gate_downgrade_that_wrote_zero(tmp_path):
    log = tmp_path / "safe_downgrade.jsonl"
    mapped = [-0.4, -0.2, 0.0, 0.3, 0.1, 0.0]
    accepted = _step(state_ns=1_000_000_000, decision_ns=1_020_000_000)
    accepted.update(
        commanded_action=[0.1, -0.2, 0.3, -0.4],
        reason="motion_allowed",
        serial_write_attempted=True,
        requested_serial_axes=mapped,
        effective_serial_axes=mapped,
        final_gate_reason="accepted",
        command_seq=42,
        serial_write_performed=True,
    )
    downgraded = dict(accepted)
    downgraded.update(
        state_monotonic_ns=1_100_000_000,
        camera_monotonic_ns=1_090_000_000,
        decision_monotonic_ns=1_120_000_000,
        effective_serial_axes=[0.0] * 6,
        final_gate_reason="state_not_fresh_or_current",
        command_seq=43,
    )
    _write_log(
        log,
        [
            _command(sequence=41, axes=[0.0] * 6, reason="act_runtime_startup"),
            _command(sequence=42, axes=mapped, reason="accepted"),
            accepted,
            _command(
                sequence=43,
                axes=[0.0] * 6,
                reason="state_not_fresh_or_current",
            ),
            downgraded,
            _command(sequence=44, axes=[0.0] * 6, reason="act_runtime_shutdown"),
        ],
    )

    report = inspect_act_runtime_log(log, mode="motion")

    assert report.passed is True
    assert report.nonzero_serial_write_count == 1


def test_motion_log_accepts_late_inference_blocked_after_terminal_zero(tmp_path):
    log = tmp_path / "terminally_disarmed.jsonl"
    mapped = [-0.4, -0.2, 0.0, 0.3, 0.1, 0.0]
    first = _step(state_ns=1_000_000_000, decision_ns=1_020_000_000)
    first.update(
        commanded_action=[0.1, -0.2, 0.3, -0.4],
        reason="motion_allowed",
        serial_write_attempted=True,
        requested_serial_axes=mapped,
        effective_serial_axes=mapped,
        final_gate_reason="accepted",
        command_seq=42,
        serial_write_performed=True,
    )
    second = dict(first)
    second.update(
        state_monotonic_ns=1_100_000_000,
        camera_monotonic_ns=1_090_000_000,
        decision_monotonic_ns=1_120_000_000,
        command_seq=43,
    )
    blocked = dict(first)
    blocked.update(
        state_monotonic_ns=1_200_000_000,
        camera_monotonic_ns=1_190_000_000,
        decision_monotonic_ns=1_220_000_000,
        effective_serial_axes=[0.0] * 6,
        final_gate_reason="terminally_disarmed",
        command_seq=None,
        serial_write_performed=False,
    )
    _write_log(
        log,
        [
            _command(sequence=41, axes=[0.0] * 6, reason="act_runtime_startup"),
            _command(sequence=42, axes=mapped, reason="accepted"),
            first,
            _command(sequence=43, axes=mapped, reason="accepted"),
            second,
            _command(sequence=44, axes=[0.0] * 6, reason="act_runtime_shutdown"),
            blocked,
        ],
    )

    report = inspect_act_runtime_log(log, mode="motion")

    assert report.passed is True
    assert report.nonzero_serial_write_count == 2


def test_cli_inspects_runtime_log_with_the_authoritative_runtime_config(
    tmp_path, capsys
):
    log = tmp_path / "shadow.jsonl"
    _write_log(
        log,
        [
            _step(state_ns=1_000_000_000, decision_ns=1_020_000_000),
            _step(state_ns=1_100_000_000, decision_ns=1_120_000_000),
        ],
    )
    config = Path(__file__).resolve().parents[1] / "config" / "act_runtime.orin.json"

    exit_code = main(
        [
            "inspect-act-runtime-log",
            str(log),
            "--mode",
            "shadow",
            "--config",
            str(config),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["passed"] is True
    assert output["step_count"] == 2


def test_latest_log_script_runs_the_offline_inspector_without_a_conda_environment(
    tmp_path,
):
    log = tmp_path / "act_runtime_shadow_20260814_010203.jsonl"
    _write_log(
        log,
        [
            _step(state_ns=1_000_000_000, decision_ns=1_020_000_000),
            _step(state_ns=1_100_000_000, decision_ns=1_120_000_000),
        ],
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "inspect_latest_act_runtime_log.sh"
    )
    environment = {**os.environ, "ACT_RUNTIME_LOG_ROOT": str(tmp_path)}

    completed = subprocess.run(
        [str(script), "shadow"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["passed"] is True

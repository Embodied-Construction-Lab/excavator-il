"""Validate a deadman-released hardware soak Episode."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


_AXES = ("X1", "Y1", "Z1", "X2", "Y2", "Z2")
_ACTION_FIELDS = (
    "command_action_boom",
    "command_action_stick",
    "command_action_bucket",
    "command_action_swing",
)


@dataclass(frozen=True)
class ZeroSoakReport:
    episode_id: str
    passed: bool
    failure_reasons: tuple[str, ...]
    stream_rates_hz: Mapping[str, float]
    nonzero_command_count: int
    nonzero_telemetry_action_count: int
    valid_action_count: int
    joystick_timeout_count: int
    serial_parse_failure_count: int
    joystick_parse_failure_count: int
    command_write_failure_count: int
    sensor_invalid_count: int
    sequence_gap_count: int


class ZeroSoakOperations(Protocol):
    def preflight(self) -> None: ...

    def start_collector(self) -> None: ...

    def start_episode(self) -> str: ...

    def start_teleop(self) -> None: ...

    def wait_for_ack(self, timeout_s: int) -> None: ...

    def monitor_deadman_released(self, duration_s: int) -> None: ...

    def abort_episode(self, reason: str) -> str: ...

    def stop_teleop(self) -> None: ...

    def stop_collector(self) -> None: ...

    def inspect_zero_soak(self, episode_path: str) -> Mapping[str, Any]: ...


def run_zero_command_soak(
    operations: ZeroSoakOperations,
    *,
    duration_s: int,
    ack_timeout_s: int,
) -> Mapping[str, Any]:
    """Run one bounded deadman-released diagnostic and return its report."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    collector_started = False
    teleop_started = False
    episode_active = False
    episode_path: str | None = None
    failure: BaseException | None = None
    try:
        operations.preflight()
        operations.start_collector()
        collector_started = True
        episode_path = operations.start_episode()
        episode_active = True
        operations.start_teleop()
        teleop_started = True
        operations.wait_for_ack(ack_timeout_s)
        operations.monitor_deadman_released(duration_s)
        episode_path = operations.abort_episode("zero_command_soak_complete")
        episode_active = False
    except BaseException as exc:
        failure = exc
        if episode_active:
            try:
                episode_path = operations.abort_episode("zero_command_soak_interrupted")
            except Exception as abort_exc:
                failure = RuntimeError(
                    f"{exc}; additionally failed to abort diagnostic Episode: "
                    f"{abort_exc}"
                )
            episode_active = False
    finally:
        if teleop_started:
            try:
                operations.stop_teleop()
            except Exception as exc:
                failure = failure or exc
        if collector_started:
            try:
                operations.stop_collector()
            except Exception as exc:
                failure = failure or exc
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    assert episode_path is not None
    return operations.inspect_zero_soak(episode_path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number}: record must be an object")
        records.append(value)
    if not records:
        raise ValueError(f"{path.name} contains no records")
    return records


def _rate_hz(stamps_ns: list[int]) -> float:
    if len(stamps_ns) < 2 or stamps_ns[-1] <= stamps_ns[0]:
        return 0.0
    return (len(stamps_ns) - 1) * 1_000_000_000.0 / (stamps_ns[-1] - stamps_ns[0])


def _counter_rate_hz(values: list[int], stamps_ns: list[int]) -> float:
    if (
        len(values) < 2
        or len(values) != len(stamps_ns)
        or stamps_ns[-1] <= stamps_ns[0]
    ):
        return 0.0
    delta = (values[-1] - values[0]) & 0xFFFFFFFF
    return delta * 1_000_000_000.0 / (stamps_ns[-1] - stamps_ns[0])


def _sequence_issues(values: list[int]) -> int:
    issues = 0
    for left, right in zip(values, values[1:]):
        issues += 1 if right <= left else max(0, right - left - 1)
    return issues


def _control_sequence_issues(values: list[int]) -> int:
    """Count discontinuities while allowing independent 20 Hz loop phases.

    The STM32 control loop and telemetry publisher run independently at the
    same nominal rate.  A telemetry sample may therefore repeat the previous
    control sequence or observe a two-step advance without any serial loss.
    """
    return sum(
        ((right - left) & 0xFFFFFFFF) not in (0, 1, 2)
        for left, right in zip(values, values[1:])
    )


def _outside_rate(name: str, value: float) -> bool:
    limits = {
        "stm32_telemetry": (18.0, 22.0),
        "new_sensor_state": (8.0, 12.0),
        "expert_action": (18.0, 22.0),
        "camera_front": (25.0, 35.0),
    }
    lower, upper = limits[name]
    return not lower <= value <= upper


def inspect_zero_command_episode(episode_path: str | Path) -> ZeroSoakReport:
    """Check that a diagnostic Episode remained safe zero at nominal rates."""
    episode = Path(episode_path)
    metadata = _read_json(episode / "episode.json")
    stm32 = _read_jsonl(episode / "stm32_raw.jsonl")
    joystick = _read_jsonl(episode / "joystick_raw.jsonl")
    actions = _read_jsonl(episode / "expert_action.jsonl")
    commands = _read_jsonl(episode / "command_tx.jsonl")
    try:
        with (episode / "camera_front_timestamps.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            cameras = list(csv.DictReader(stream))
    except OSError as exc:
        raise ValueError(f"cannot read camera_front_timestamps.csv: {exc}") from exc
    if not cameras:
        raise ValueError("camera_front_timestamps.csv contains no records")

    parsed_stm32 = [
        record
        for record in stm32
        if record.get("parse_ok") is True and isinstance(record.get("telemetry"), dict)
    ]
    telemetry = [record["telemetry"] for record in parsed_stm32]
    new_states = [
        record for record in parsed_stm32 if int(record["telemetry"]["sensor_is_new"]) == 1
    ]
    stream_rates = {
        "stm32_telemetry": _rate_hz(
            [int(record["orin_receive_monotonic_ns"]) for record in parsed_stm32]
        ),
        "new_sensor_state": _rate_hz(
            [int(record["orin_receive_monotonic_ns"]) for record in new_states]
        ),
        "expert_action": _rate_hz(
            [int(record["action_stamp_monotonic_ns"]) for record in actions]
        ),
        "camera_front": _rate_hz(
            [int(record["camera_stamp_monotonic_ns"]) for record in cameras]
        ),
    }
    telemetry_stamps_ns = [
        int(record["orin_receive_monotonic_ns"]) for record in parsed_stm32
    ]
    control_rate_hz = _counter_rate_hz(
        [int(frame["control_seq"]) for frame in telemetry], telemetry_stamps_ns
    )
    max_stm32_receive_period_ms = max(
        (
            (current - previous) / 1_000_000.0
            for previous, current in zip(
                telemetry_stamps_ns, telemetry_stamps_ns[1:]
            )
        ),
        default=0.0,
    )

    nonzero_commands = 0
    invalid_payloads = 0
    for record in commands:
        try:
            payload = json.loads(str(record["raw_serial_payload"]))
            nonzero_commands += int(
                any(abs(float(payload[field])) > 1e-9 for field in _AXES)
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid_payloads += 1
    nonzero_telemetry = sum(
        any(abs(float(frame[field])) > 1e-9 for field in _ACTION_FIELDS)
        for frame in telemetry
    )
    valid_actions = sum(record.get("action_valid") is True for record in actions)
    timeout_count = sum(
        record.get("command_kind") == "safe_zero:joystick_timeout"
        for record in commands
    )
    parse_failures = sum(record.get("parse_ok") is not True for record in stm32)
    joystick_parse_failures = sum(
        record.get("parse_ok") is not True for record in joystick
    )
    write_failures = sum(record.get("write_ok") is not True for record in commands)
    sensor_invalid = sum(
        any(int(frame.get(field, 0)) != 1 for field in ("rs485_ok", "dwj_ok", "imu_ok"))
        for frame in telemetry
    )
    sequence_issues = sum(
        (
            _sequence_issues([int(record["raw_frame_seq"]) for record in stm32]),
            _control_sequence_issues(
                [int(frame["control_seq"]) for frame in telemetry]
            ),
            _sequence_issues(
                [int(record["telemetry"]["sensor_seq"]) for record in new_states]
            ),
            _sequence_issues(
                [
                    int(record["joystick_sample_seq"])
                    for record in joystick
                    if record.get("parse_ok") is True
                    and record.get("joystick_sample_seq") is not None
                ]
            ),
            _sequence_issues(
                [int(record["camera_frame_index"]) for record in cameras]
            ),
        )
    )

    failures: list[str] = []
    if (
        metadata.get("status") != "aborted"
        or metadata.get("failure_reason") != "zero_command_soak_complete"
    ):
        failures.append("Episode must be aborted with zero_command_soak_complete")
    for name, rate in stream_rates.items():
        if _outside_rate(name, rate):
            failures.append(f"{name} rate {rate:.3f} Hz is outside its allowed range")
    if not 18.0 <= control_rate_hz <= 22.0:
        failures.append(
            f"STM32 control rate {control_rate_hz:.3f} Hz is outside [18, 22]"
        )
    if max_stm32_receive_period_ms > 80.0:
        failures.append(
            "STM32 maximum receive period "
            f"{max_stm32_receive_period_ms:.3f} ms exceeds 80 ms"
        )
    counters = {
        "nonzero command": nonzero_commands,
        "invalid command payload": invalid_payloads,
        "nonzero telemetry action": nonzero_telemetry,
        "valid expert action": valid_actions,
        "joystick timeout": timeout_count,
        "serial parse failure": parse_failures,
        "joystick parse failure": joystick_parse_failures,
        "command write failure": write_failures,
        "invalid sensor": sensor_invalid,
        "sequence gap or reordering": sequence_issues,
    }
    failures.extend(
        f"{name} count is {count}" for name, count in counters.items() if count
    )
    return ZeroSoakReport(
        episode_id=str(metadata.get("episode_id", episode.name)),
        passed=not failures,
        failure_reasons=tuple(failures),
        stream_rates_hz=stream_rates,
        nonzero_command_count=nonzero_commands,
        nonzero_telemetry_action_count=nonzero_telemetry,
        valid_action_count=valid_actions,
        joystick_timeout_count=timeout_count,
        serial_parse_failure_count=parse_failures,
        joystick_parse_failure_count=joystick_parse_failures,
        command_write_failure_count=write_failures,
        sensor_invalid_count=sensor_invalid,
        sequence_gap_count=sequence_issues,
    )


def main(argv: list[str] | None = None) -> int:
    """PC entry point for one bounded zero-command hardware soak."""
    from .guided_episode import GuidedEpisodeConfig, SystemGuidedEpisodeOperations

    default_config = Path(__file__).resolve().parents[2] / "config/guided_episode.pc.json"
    parser = argparse.ArgumentParser(description="run one zero-command hardware soak")
    parser.add_argument("--config", default=str(default_config))
    args = parser.parse_args(argv)
    try:
        config = GuidedEpisodeConfig.load(args.config)
        operations = SystemGuidedEpisodeOperations(config)
        report = run_zero_command_soak(
            operations,
            duration_s=config.zero_soak_duration_s,
            ack_timeout_s=config.ack_timeout_s,
        )
    except KeyboardInterrupt:
        print("zero-command soak interrupted by operator", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dict(report), ensure_ascii=False, indent=2))
    return 0 if report.get("passed") is True else 3

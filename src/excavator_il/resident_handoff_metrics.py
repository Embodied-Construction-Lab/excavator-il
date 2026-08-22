"""Validate and summarize physical RL/ACT resident handoff evidence.

The Orin serial owner emits one ``RESIDENT_HANDOFF_SAMPLE`` record only after
the STM32 has acknowledged both ends of a policy handoff.  This module treats
those records as an experimental evidence contract: malformed or conflicting
records fail the benchmark instead of being silently skipped.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


SAMPLE_LOG_PREFIX = "RESIDENT_HANDOFF_SAMPLE "
SAMPLE_SCHEMA_VERSION = "resident_handoff_sample.v1"
BENCHMARK_SCHEMA_VERSION = "resident_handoff_benchmark.v1"

RL_TO_ACT = "rl_follow/velocity_reference->act_dig/manual_action"
ACT_TO_RL = "act_dig/manual_action->rl_follow/velocity_reference"
_DIRECTIONS = (RL_TO_ACT, ACT_TO_RL)

_MINIMUM_SAMPLES = 20
_P50_LIMIT_MS = 200.0
_P95_LIMIT_MS = 300.0
_MAX_LIMIT_MS = 1000.0

_SAMPLE_KEYS = frozenset(
    {
        "schema_version",
        "runtime_id",
        "generation",
        "from_source",
        "from_mode",
        "to_source",
        "to_mode",
        "terminal_zero_command_seq",
        "terminal_zero_ack_monotonic_ns",
        "target_zero_command_seq",
        "target_zero_ack_monotonic_ns",
        "first_nonzero_command_seq",
        "first_nonzero_action",
        "first_nonzero_write_monotonic_ns",
        "first_nonzero_ack_monotonic_ns",
        "zero_claim_ms",
        "policy_ready_wait_ms",
        "first_command_ack_ms",
        "latency_ms",
    }
)


class HandoffMetricsError(ValueError):
    """A resident handoff evidence record cannot be trusted."""


@dataclass(frozen=True)
class ResidentHandoffEvidence:
    runtime_id: str
    generation: int
    from_source: str
    from_mode: str
    to_source: str
    to_mode: str
    terminal_zero_command_seq: int
    terminal_zero_ack_monotonic_ns: int
    target_zero_command_seq: int
    target_zero_ack_monotonic_ns: int
    first_nonzero_command_seq: int
    first_nonzero_action: tuple[float, float, float, float]
    first_nonzero_write_monotonic_ns: int
    first_nonzero_ack_monotonic_ns: int
    zero_claim_ms: float
    policy_ready_wait_ms: float
    first_command_ack_ms: float
    latency_ms: float

    @property
    def identity(self) -> tuple[str, int]:
        return (self.runtime_id, self.generation)

    @property
    def direction(self) -> str:
        return (
            f"{self.from_source}/{self.from_mode}"
            f"->{self.to_source}/{self.to_mode}"
        )


def analyze_handoff_logs(
    paths: Iterable[str | Path],
) -> dict[str, object]:
    """Return a deterministic benchmark report for one or more owner logs."""

    source_paths = _source_paths(paths)
    samples = parse_handoff_logs(source_paths)
    return _summarize(samples, source_paths)


def parse_handoff_logs(
    paths: Iterable[str | Path],
) -> tuple[ResidentHandoffEvidence, ...]:
    """Parse, strictly validate and deduplicate handoff samples."""

    source_paths = _source_paths(paths)
    samples_by_identity: dict[tuple[str, int], ResidentHandoffEvidence] = {}
    origins: dict[tuple[str, int], str] = {}
    for path in source_paths:
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise HandoffMetricsError(f"cannot read {path}: {exc}") from exc
        try:
            with handle:
                for line_number, line in enumerate(handle, start=1):
                    if SAMPLE_LOG_PREFIX not in line:
                        continue
                    encoded = line.split(SAMPLE_LOG_PREFIX, 1)[1].strip()
                    sample = _decode_sample(
                        encoded,
                        location=f"{path}:{line_number}",
                    )
                    existing = samples_by_identity.get(sample.identity)
                    if existing is None:
                        samples_by_identity[sample.identity] = sample
                        origins[sample.identity] = f"{path}:{line_number}"
                        continue
                    if existing != sample:
                        raise HandoffMetricsError(
                            f"{path}:{line_number}: conflicting duplicate for "
                            f"runtime_id={sample.runtime_id!r}, "
                            f"generation={sample.generation}; first seen at "
                            f"{origins[sample.identity]}"
                        )
        except UnicodeError as exc:
            raise HandoffMetricsError(f"cannot decode {path} as UTF-8: {exc}") from exc

    return tuple(
        samples_by_identity[key]
        for key in sorted(samples_by_identity, key=lambda item: (item[0], item[1]))
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate resident RL/ACT physical handoff samples and report the "
            "20-run acceptance benchmark."
        )
    )
    parser.add_argument("logs", nargs="+", help="resident owner log path(s)")
    args = parser.parse_args(argv)
    source_paths = _source_paths(args.logs)
    try:
        report = analyze_handoff_logs(source_paths)
    except (HandoffMetricsError, OSError, UnicodeError) as exc:
        report = _validation_failure_report(source_paths, str(exc))
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


def _decode_sample(encoded: str, *, location: str) -> ResidentHandoffEvidence:
    try:
        value = json.loads(
            encoded,
            parse_constant=lambda token: _reject_json_constant(token),
            object_pairs_hook=_strict_json_object,
        )
    except (json.JSONDecodeError, ValueError, HandoffMetricsError) as exc:
        raise HandoffMetricsError(f"{location}: invalid handoff JSON: {exc}") from exc
    try:
        return _validate_sample(value)
    except HandoffMetricsError as exc:
        raise HandoffMetricsError(f"{location}: {exc}") from exc


def _validate_sample(value: object) -> ResidentHandoffEvidence:
    if type(value) is not dict:
        raise HandoffMetricsError("handoff sample must be a JSON object")
    keys = frozenset(value)
    if keys != _SAMPLE_KEYS:
        missing = sorted(_SAMPLE_KEYS - keys)
        extra = sorted(keys - _SAMPLE_KEYS)
        raise HandoffMetricsError(
            "handoff sample must have the exact key set; "
            f"missing={missing}, extra={extra}"
        )
    if _string(value, "schema_version") != SAMPLE_SCHEMA_VERSION:
        raise HandoffMetricsError(
            f"schema_version must be {SAMPLE_SCHEMA_VERSION}"
        )

    runtime_id = _string(value, "runtime_id")
    if runtime_id != runtime_id.strip() or not runtime_id or len(runtime_id) > 128:
        raise HandoffMetricsError(
            "runtime_id must be a trimmed non-empty string of at most 128 chars"
        )
    generation = _nonnegative_int(value, "generation")
    from_source = _string(value, "from_source")
    from_mode = _string(value, "from_mode")
    to_source = _string(value, "to_source")
    to_mode = _string(value, "to_mode")
    direction = f"{from_source}/{from_mode}->{to_source}/{to_mode}"
    if direction not in _DIRECTIONS:
        raise HandoffMetricsError(f"unsupported handoff direction: {direction}")

    terminal_seq = _uint32(value, "terminal_zero_command_seq")
    target_seq = _uint32(value, "target_zero_command_seq")
    first_seq = _uint32(value, "first_nonzero_command_seq")
    terminal_ack = _nonnegative_int(
        value, "terminal_zero_ack_monotonic_ns"
    )
    target_ack = _nonnegative_int(value, "target_zero_ack_monotonic_ns")
    first_write = _nonnegative_int(
        value, "first_nonzero_write_monotonic_ns"
    )
    first_ack = _nonnegative_int(value, "first_nonzero_ack_monotonic_ns")
    if not terminal_ack <= target_ack <= first_write <= first_ack:
        raise HandoffMetricsError(
            "handoff timestamps must satisfy terminal ACK <= target ACK <= "
            "first write <= first ACK"
        )

    action_value = value["first_nonzero_action"]
    if type(action_value) is not list or len(action_value) != 4:
        raise HandoffMetricsError(
            "first_nonzero_action must be a four-element JSON array"
        )
    action = tuple(
        _finite_float_item(item, name=f"first_nonzero_action[{index}]")
        for index, item in enumerate(action_value)
    )
    if not any(axis != 0.0 for axis in action):
        raise HandoffMetricsError("first_nonzero_action must be nonzero")

    zero_claim_ms = _finite_duration(value, "zero_claim_ms")
    policy_ready_wait_ms = _finite_duration(value, "policy_ready_wait_ms")
    first_command_ack_ms = _finite_duration(value, "first_command_ack_ms")
    latency_ms = _finite_duration(value, "latency_ms")
    _validate_duration(
        "zero_claim_ms",
        zero_claim_ms,
        start_ns=terminal_ack,
        end_ns=target_ack,
    )
    _validate_duration(
        "policy_ready_wait_ms",
        policy_ready_wait_ms,
        start_ns=target_ack,
        end_ns=first_write,
    )
    _validate_duration(
        "first_command_ack_ms",
        first_command_ack_ms,
        start_ns=first_write,
        end_ns=first_ack,
    )
    _validate_duration(
        "latency_ms",
        latency_ms,
        start_ns=terminal_ack,
        end_ns=first_ack,
    )

    return ResidentHandoffEvidence(
        runtime_id=runtime_id,
        generation=generation,
        from_source=from_source,
        from_mode=from_mode,
        to_source=to_source,
        to_mode=to_mode,
        terminal_zero_command_seq=terminal_seq,
        terminal_zero_ack_monotonic_ns=terminal_ack,
        target_zero_command_seq=target_seq,
        target_zero_ack_monotonic_ns=target_ack,
        first_nonzero_command_seq=first_seq,
        first_nonzero_action=action,  # type: ignore[arg-type]
        first_nonzero_write_monotonic_ns=first_write,
        first_nonzero_ack_monotonic_ns=first_ack,
        zero_claim_ms=zero_claim_ms,
        policy_ready_wait_ms=policy_ready_wait_ms,
        first_command_ack_ms=first_command_ack_ms,
        latency_ms=latency_ms,
    )


def _summarize(
    samples: Sequence[ResidentHandoffEvidence],
    source_paths: Sequence[Path],
) -> dict[str, object]:
    directions: dict[str, dict[str, object]] = {}
    failure_reasons: list[str] = []
    for direction in _DIRECTIONS:
        latencies = sorted(
            sample.latency_ms for sample in samples if sample.direction == direction
        )
        count = len(latencies)
        p50 = _nearest_rank(latencies, 0.50)
        p95 = _nearest_rank(latencies, 0.95)
        maximum = None if not latencies else latencies[-1]
        direction_passed = count >= _MINIMUM_SAMPLES
        if count < _MINIMUM_SAMPLES:
            failure_reasons.append(
                f"{direction} requires at least {_MINIMUM_SAMPLES} samples; "
                f"observed {count}"
            )
        for label, measured, limit in (
            ("p50", p50, _P50_LIMIT_MS),
            ("p95", p95, _P95_LIMIT_MS),
            ("max", maximum, _MAX_LIMIT_MS),
        ):
            if measured is not None and measured >= limit:
                direction_passed = False
                failure_reasons.append(
                    f"{direction} {label} {measured:.6f} ms must be < "
                    f"{limit:.6f} ms"
                )
        directions[direction] = {
            "count": count,
            "p50_ms": p50,
            "p95_ms": p95,
            "max_ms": maximum,
            "passed": direction_passed,
        }

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "passed": not failure_reasons,
        "metric": _metric_description(),
        "thresholds": _thresholds(),
        "source_logs": [str(path) for path in source_paths],
        "sample_count": len(samples),
        "directions": directions,
        "failure_reasons": failure_reasons,
    }


def _validation_failure_report(
    source_paths: Sequence[Path],
    reason: str,
) -> dict[str, object]:
    empty = _summarize((), source_paths)
    return {
        **empty,
        "passed": False,
        "failure_reasons": [reason],
    }


def _metric_description() -> dict[str, str]:
    return {
        "name": "physical_command_gap",
        "start": "terminal_old_source_zero_stm32_ack",
        "end": "first_target_nonzero_stm32_ack",
        "unit": "ms",
        "planning_caveat": (
            "Planning wait may contribute when the target policy is not prepared "
            "before handoff."
        ),
    }


def _thresholds() -> dict[str, float | int]:
    return {
        "minimum_samples_per_direction": _MINIMUM_SAMPLES,
        "p50_ms_lt": _P50_LIMIT_MS,
        "p95_ms_lt": _P95_LIMIT_MS,
        "max_ms_lt": _MAX_LIMIT_MS,
    }


def _source_paths(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    normalized: set[Path] = set()
    for raw_path in paths:
        if not isinstance(raw_path, (str, Path)):
            raise HandoffMetricsError("log paths must be strings or Path objects")
        normalized.add(Path(raw_path).expanduser().resolve())
    if not normalized:
        raise HandoffMetricsError("at least one resident owner log is required")
    return tuple(sorted(normalized, key=str))


def _nearest_rank(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    rank = math.ceil(probability * len(values))
    return values[rank - 1]


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise HandoffMetricsError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(token: str) -> object:
    raise HandoffMetricsError(f"non-finite JSON constant is forbidden: {token}")


def _string(value: dict[str, object], name: str) -> str:
    item = value[name]
    if type(item) is not str:
        raise HandoffMetricsError(f"{name} must be a string")
    return item


def _nonnegative_int(value: dict[str, object], name: str) -> int:
    item = value[name]
    if type(item) is not int or item < 0:
        raise HandoffMetricsError(f"{name} must be a nonnegative integer")
    return item


def _uint32(value: dict[str, object], name: str) -> int:
    item = _nonnegative_int(value, name)
    if item > 0xFFFFFFFF:
        raise HandoffMetricsError(f"{name} must be a uint32")
    return item


def _finite_float_item(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise HandoffMetricsError(f"{name} must be a finite float")
    return value


def _finite_duration(value: dict[str, object], name: str) -> float:
    item = _finite_float_item(value[name], name=name)
    if item < 0.0:
        raise HandoffMetricsError(f"{name} must be nonnegative")
    return item


def _validate_duration(
    name: str,
    measured_ms: float,
    *,
    start_ns: int,
    end_ns: int,
) -> None:
    expected_ms = (end_ns - start_ns) / 1_000_000.0
    if not math.isclose(measured_ms, expected_ms, rel_tol=0.0, abs_tol=1e-9):
        raise HandoffMetricsError(
            f"{name} does not match timestamps: "
            f"reported={measured_ms}, expected={expected_ms}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import pytest

from excavator_il.resident_handoff_metrics import (
    HandoffMetricsError,
    analyze_handoff_logs,
    main,
)


PREFIX = "RESIDENT_HANDOFF_SAMPLE "
RL_TO_ACT = "rl_follow/velocity_reference->act_dig/manual_action"
ACT_TO_RL = "act_dig/manual_action->rl_follow/velocity_reference"


def _sample(
    *,
    runtime_id: str,
    generation: int,
    direction: str,
    latency_ms: float,
) -> dict[str, object]:
    if direction == RL_TO_ACT:
        from_source, from_mode = "rl_follow", "velocity_reference"
        to_source, to_mode = "act_dig", "manual_action"
    elif direction == ACT_TO_RL:
        from_source, from_mode = "act_dig", "manual_action"
        to_source, to_mode = "rl_follow", "velocity_reference"
    else:
        raise AssertionError(f"unsupported test direction: {direction}")

    terminal_ack_ns = generation * 10_000_000_000 + 1_000_000_000
    target_ack_ns = terminal_ack_ns + 10_000_000
    first_write_ns = target_ack_ns + 20_000_000
    first_ack_ns = terminal_ack_ns + round(latency_ms * 1_000_000)
    if first_ack_ns < first_write_ns:
        raise AssertionError("test latency must be at least 30 ms")
    return {
        "schema_version": "resident_handoff_sample.v1",
        "runtime_id": runtime_id,
        "generation": generation,
        "from_source": from_source,
        "from_mode": from_mode,
        "to_source": to_source,
        "to_mode": to_mode,
        "terminal_zero_command_seq": generation * 3,
        "terminal_zero_ack_monotonic_ns": terminal_ack_ns,
        "target_zero_command_seq": generation * 3 + 1,
        "target_zero_ack_monotonic_ns": target_ack_ns,
        "first_nonzero_command_seq": generation * 3 + 2,
        "first_nonzero_action": [0.25, 0.0, -0.5, 0.0],
        "first_nonzero_write_monotonic_ns": first_write_ns,
        "first_nonzero_ack_monotonic_ns": first_ack_ns,
        "zero_claim_ms": 10.0,
        "policy_ready_wait_ms": 20.0,
        "first_command_ack_ms": (first_ack_ns - first_write_ns) / 1_000_000.0,
        "latency_ms": (first_ack_ns - terminal_ack_ns) / 1_000_000.0,
    }


def _write_log(path: Path, samples: list[dict[str, object]]) -> None:
    lines = ["ordinary owner output before evidence\n"]
    lines.extend(
        f"2026-08-21 INFO {PREFIX}{json.dumps(sample, sort_keys=True)}\n"
        for sample in samples
    )
    path.write_text("".join(lines), encoding="utf-8")


def test_twenty_samples_in_each_direction_pass_with_nearest_rank_percentiles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resident-owner.log"
    samples = [
        _sample(
            runtime_id="runtime-pass",
            generation=index,
            direction=RL_TO_ACT,
            latency_ms=float(40 + index),
        )
        for index in range(1, 21)
    ]
    samples.extend(
        _sample(
            runtime_id="runtime-pass",
            generation=100 + index,
            direction=ACT_TO_RL,
            latency_ms=float(80 + index * 2),
        )
        for index in range(1, 21)
    )
    _write_log(path, samples)

    report = analyze_handoff_logs([path])

    assert report["schema_version"] == "resident_handoff_benchmark.v1"
    assert report["passed"] is True
    assert report["sample_count"] == 40
    assert report["source_logs"] == [str(path.resolve())]
    assert report["metric"] == {
        "end": "first_target_nonzero_stm32_ack",
        "name": "physical_command_gap",
        "planning_caveat": (
            "Planning wait may contribute when the target policy is not prepared "
            "before handoff."
        ),
        "start": "terminal_old_source_zero_stm32_ack",
        "unit": "ms",
    }
    assert report["directions"][RL_TO_ACT] == {
        "count": 20,
        "max_ms": 60.0,
        "p50_ms": 50.0,
        "p95_ms": 59.0,
        "passed": True,
    }
    assert report["directions"][ACT_TO_RL] == {
        "count": 20,
        "max_ms": 120.0,
        "p50_ms": 100.0,
        "p95_ms": 118.0,
        "passed": True,
    }
    assert report["failure_reasons"] == []


def test_insufficient_samples_fail_and_cli_returns_two_with_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "short.log"
    _write_log(
        path,
        [
            _sample(
                runtime_id="runtime-short",
                generation=1,
                direction=RL_TO_ACT,
                latency_ms=45.0,
            )
        ],
    )

    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = main([str(path)])

    report = json.loads(output.getvalue())
    assert exit_code == 2
    assert report["passed"] is False
    assert report["directions"][RL_TO_ACT]["count"] == 1
    assert report["directions"][ACT_TO_RL]["count"] == 0
    assert report["directions"][ACT_TO_RL]["p50_ms"] is None
    assert any("requires at least 20 samples" in item for item in report["failure_reasons"])


def test_threshold_is_strict_and_a_maximum_of_1000_ms_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "threshold.log"
    samples = [
        _sample(
            runtime_id="runtime-threshold",
            generation=index,
            direction=RL_TO_ACT,
            latency_ms=1000.0 if index == 20 else 50.0,
        )
        for index in range(1, 21)
    ]
    samples.extend(
        _sample(
            runtime_id="runtime-threshold",
            generation=100 + index,
            direction=ACT_TO_RL,
            latency_ms=60.0,
        )
        for index in range(1, 21)
    )
    _write_log(path, samples)

    report = analyze_handoff_logs([path])

    assert report["passed"] is False
    assert report["directions"][RL_TO_ACT]["passed"] is False
    assert report["directions"][RL_TO_ACT]["max_ms"] == 1000.0
    assert report["directions"][ACT_TO_RL]["passed"] is True
    assert any(
        "max 1000.000000 ms must be < 1000.000000 ms" in item
        for item in report["failure_reasons"]
    )


@pytest.mark.parametrize(
    "mutation, expected_message",
    [
        (lambda value: {**value, "unexpected": 1}, "exact key set"),
        (
            lambda value: {
                **value,
                "to_source": "unknown_policy",
            },
            "unsupported handoff direction",
        ),
        (
            lambda value: {
                **value,
                "first_nonzero_action": [0.0, 0.0, 0.0, 0.0],
            },
            "first_nonzero_action must be nonzero",
        ),
        (
            lambda value: {
                **value,
                "latency_ms": value["latency_ms"] + 1.0,
            },
            "latency_ms does not match timestamps",
        ),
    ],
)
def test_malformed_samples_are_rejected(
    tmp_path: Path,
    mutation,
    expected_message: str,
) -> None:
    path = tmp_path / "malformed.log"
    sample = _sample(
        runtime_id="runtime-malformed",
        generation=1,
        direction=RL_TO_ACT,
        latency_ms=45.0,
    )
    _write_log(path, [mutation(sample)])

    with pytest.raises(HandoffMetricsError, match=expected_message):
        analyze_handoff_logs([path])


def test_exact_duplicate_is_deduplicated_across_logs(tmp_path: Path) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    sample = _sample(
        runtime_id="runtime-dedup",
        generation=7,
        direction=RL_TO_ACT,
        latency_ms=55.0,
    )
    _write_log(first, [sample])
    _write_log(second, [sample])

    report = analyze_handoff_logs([second, first])

    assert report["sample_count"] == 1
    assert report["directions"][RL_TO_ACT]["count"] == 1
    assert report["source_logs"] == sorted(
        [str(first.resolve()), str(second.resolve())]
    )


def test_conflicting_duplicate_fails(tmp_path: Path) -> None:
    path = tmp_path / "conflict.log"
    first = _sample(
        runtime_id="runtime-conflict",
        generation=9,
        direction=RL_TO_ACT,
        latency_ms=55.0,
    )
    second = _sample(
        runtime_id="runtime-conflict",
        generation=9,
        direction=RL_TO_ACT,
        latency_ms=65.0,
    )
    _write_log(path, [first, second])

    with pytest.raises(HandoffMetricsError, match="conflicting duplicate"):
        analyze_handoff_logs([path])


def test_validation_failure_cli_emits_json_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "invalid-json.log"
    path.write_text(f"{PREFIX}{{not-json}}\n", encoding="utf-8")

    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = main([str(path)])

    report = json.loads(output.getvalue())
    assert exit_code == 2
    assert report["schema_version"] == "resident_handoff_benchmark.v1"
    assert report["passed"] is False
    assert "Traceback" not in output.getvalue()
    assert "invalid handoff JSON" in report["failure_reasons"][0]

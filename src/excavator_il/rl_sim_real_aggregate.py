"""Strict held-out aggregation for RL simulation-real tracking pair reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


RL_SIM_REAL_AGGREGATE_SCHEMA_VERSION = "excavator_rl_sim_real_aggregate.v1"
RL_SIM_REAL_PAIR_SCHEMA_VERSION = "excavator_rl_sim_real_pair.v1"
_PAIR_FIELDS = frozenset(
    {
        "schema_version",
        "pair_id",
        "evaluation_scope",
        "trace_semantics",
        "simulation_run_id",
        "real_machine_run_id",
        "simulation_sample_count",
        "real_machine_sample_count",
        "aligned_sample_count",
        "simulation_only_tail_count",
        "simulation_only_tail_sample_ids",
        "real_machine_only_tail_count",
        "real_machine_only_tail_sample_ids",
        "sample_coverage",
        "duration_s",
        "trace_sha256",
        "nonzero_agreement_rate",
        "axes",
        "tracking",
        "binding",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "trajectory_controller_policy_id",
        "trajectory_controller_onnx_sha256",
        "machine_profile_sha256",
        "action_order",
        "trajectory_suite_sha256",
        "trajectory_suite_sample_period_s",
        "trajectory_trace_schema_version",
        "trace_semantics",
    }
)
_ACTION_ORDER = ("boom", "stick", "bucket", "swing")
_SIDES = ("simulation", "real_machine")
_TERMINALS = ("COMPLETED", "TIMEOUT", "REJECTED", "INTERRUPTED")
_EPSILON = 1e-12


def aggregate_rl_sim_real_pair_reports(
    pair_report_paths: Iterable[str | Path],
) -> dict[str, object]:
    """Aggregate an explicit attempted-pair list without dropping failed trials."""

    loaded = tuple(_load_pair_report(Path(path).expanduser()) for path in pair_report_paths)
    if not loaded:
        raise ValueError("at least one attempted pair report is required")
    resolved_paths = tuple(item["resolved_path"] for item in loaded)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("attempted pair report paths must be unique")
    reports = tuple(item["report"] for item in loaded)
    _require_unique_ids(reports, "pair_id")
    _require_unique_ids(reports, "simulation_run_id")
    _require_unique_ids(reports, "real_machine_run_id")
    scopes = {report["evaluation_scope"] for report in reports}
    if scopes != {"held_out_experiment"}:
        raise ValueError(
            "held-out sim-real aggregate accepts only held_out_experiment reports"
        )
    schema_versions = {report["schema_version"] for report in reports}
    if schema_versions != {RL_SIM_REAL_PAIR_SCHEMA_VERSION}:
        raise ValueError("pair report schema versions do not match")
    trace_semantics = {report["trace_semantics"] for report in reports}
    if trace_semantics != {"commanded_normalized_action"}:
        raise ValueError("pair report trace semantics do not match")
    first_binding = reports[0]["binding"]
    if any(report["binding"] != first_binding for report in reports[1:]):
        raise ValueError("pair report binding or hash drift detected")
    suite_counts = {
        report["sample_coverage"][side]["suite_count"]
        for report in reports
        for side in _SIDES
    }


    if len(suite_counts) != 1:
        raise ValueError(
            "trajectory suite sample count drift detected under the same suite hash"
        )

    denominator = len(reports)
    return {
        "schema_version": RL_SIM_REAL_AGGREGATE_SCHEMA_VERSION,
        "evaluation_scope": "held_out_experiment",
        "attempted_pair_count": denominator,
        "binding": dict(first_binding),
        "attempted_pairs": _attempted_pair_evidence(loaded, reports),
        "terminal": _aggregate_terminal(reports),
        "tracking": _aggregate_tracking(reports),
        "duration_s": _aggregate_duration(reports),
        "sample_counts": _aggregate_sample_counts(reports),
        "tails": _aggregate_tails(reports),
        "sample_coverage": _aggregate_sample_coverage(reports),
        "statistical_unit": "pair_run",
    }


def _attempted_pair_evidence(
    loaded: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    return [
        {
            "pair_id": report["pair_id"],
            "simulation_run_id": report["simulation_run_id"],
            "real_machine_run_id": report["real_machine_run_id"],
            "report_path": str(item["resolved_path"]),
            "report_sha256": item["sha256"],
            "simulation_sample_count": report["simulation_sample_count"],
            "real_machine_sample_count": report["real_machine_sample_count"],
            "aligned_sample_count": report["aligned_sample_count"],
            "simulation_only_tail_sample_ids": report[
                "simulation_only_tail_sample_ids"
            ],
            "real_machine_only_tail_sample_ids": report[
                "real_machine_only_tail_sample_ids"
            ],
            "trace_sha256": dict(report["trace_sha256"]),
            "simulation_terminal": report["tracking"]["terminal_result"]["simulation"],
            "real_machine_terminal": report["tracking"]["terminal_result"]["real_machine"],
        }
        for item, report in zip(loaded, reports)
    ]


def _aggregate_terminal(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    denominator = len(reports)
    return {
        "agreement": _count_rate(
            sum(
                bool(report["tracking"]["terminal_result"]["agreement"])
                for report in reports
            ),
            denominator,
        ),
        **{
            side: {
                terminal.lower(): _count_rate(
                    sum(
                        report["tracking"]["terminal_result"][side] == terminal
                        for report in reports
                    ),
                    denominator,
                )
                for terminal in _TERMINALS
            }
            for side in _SIDES
        },
    }


def _aggregate_tracking(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    error_metrics = ("mae", "rmse", "max")
    return {
        "bucket_tip_euclidean_error_m": {
            f"pair_{metric}": _distribution(
                [
                    report["tracking"]["bucket_tip_euclidean_error_m"][metric]
                    for report in reports
                ]
            )
            for metric in error_metrics
        },
        "reference_waypoint_euclidean_error_m": {
            f"pair_{metric}": _distribution(
                [
                    report["tracking"]["reference_waypoint_euclidean_error_m"][
                        metric
                    ]
                    for report in reports
                ]
            )
            for metric in error_metrics
        },
        "waypoint_index_agreement": {
            "pair_rate": _distribution(
                [
                    report["tracking"]["waypoint_index_agreement"]["rate"]
                    for report in reports
                ]
            ),
            "pair_disagreement_rate": _distribution(
                [
                    1.0
                    - report["tracking"]["waypoint_index_agreement"]["rate"]
                    for report in reports
                ]
            ),
        },
        "relative_sample_timing_error_s": {
            f"pair_{metric}": _distribution(
                [
                    report["tracking"]["relative_sample_timing_error_s"][metric]
                    for report in reports
                ]
            )
            for metric in ("mae", "rmse", "max_abs")
        },
        "waypoint_distance_m": {
            side: {
                f"pair_{metric}": _distribution(
                    [
                        report["tracking"]["waypoint_distance_m"][side][metric]
                        for report in reports
                    ]
                )
                for metric in ("final", "p95")
            }
            for side in _SIDES
        },
    }


def _aggregate_duration(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    return {
        side: _distribution([report["duration_s"][side] for report in reports])
        for side in _SIDES
    }


def _aggregate_sample_coverage(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    return {
        **{
            side: {
                "pair_rate": _distribution(
                    [
                        report["sample_coverage"][side]["rate"]
                        for report in reports
                    ]
                ),
                "total_consumed_sample_count": sum(
                    report["sample_coverage"][side]["consumed_count"]
                    for report in reports
                ),
                "total_trajectory_suite_sample_count": sum(
                    report["sample_coverage"][side]["suite_count"]
                    for report in reports
                ),
            }
            for side in _SIDES
        },
        "total_aligned_sample_count": sum(
            report["aligned_sample_count"] for report in reports
        ),
    }


def _aggregate_sample_counts(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    fields = {
        "simulation": "simulation_sample_count",
        "real_machine": "real_machine_sample_count",
        "aligned": "aligned_sample_count",
    }
    return {
        name: _distribution([report[field] for report in reports])
        for name, field in fields.items()
    }


def _aggregate_tails(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    denominator = len(reports)
    return {
        side: {
            "count_per_pair": _distribution(
                [report[f"{side}_only_tail_count"] for report in reports]
            ),
            "total_sample_count": sum(
                report[f"{side}_only_tail_count"] for report in reports
            ),
            "affected_pairs": _count_rate(
                sum(report[f"{side}_only_tail_count"] > 0 for report in reports),
                denominator,
            ),
        }
        for side in _SIDES
    }


def write_rl_sim_real_aggregate(
    output_path: str | Path,
    aggregate: Mapping[str, object],
) -> Path:
    """Atomically create one immutable aggregate JSON artifact."""

    if aggregate.get("schema_version") != RL_SIM_REAL_AGGREGATE_SCHEMA_VERSION:
        raise ValueError(
            f"aggregate schema_version must be {RL_SIM_REAL_AGGREGATE_SCHEMA_VERSION}"
        )
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError(f"aggregate output must be a regular file: {target}")
    if target.exists():
        raise ValueError(f"aggregate output already exists: {target}")
    payload = (
        json.dumps(
            aggregate,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, target)
        except FileExistsError as exc:
            raise ValueError(f"aggregate output already exists: {target}") from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target


def validate_rl_sim_real_binding(value: object) -> dict[str, object]:
    """Return one validated canonical tracking binding snapshot."""

    binding = _mapping(value, "binding")
    if binding.get("trace_semantics") != "commanded_normalized_action":
        raise ValueError("binding.trace_semantics must be commanded_normalized_action")
    _validate_binding(binding, binding.get("trace_semantics"))
    return dict(binding)


def _load_pair_report(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pair report must be a regular file: {path}")
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read pair report {path}: {exc}") from exc
    report = _validate_pair_report(value)
    return {
        "resolved_path": path.resolve(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "report": report,
    }


def _validate_pair_report(value: object) -> Mapping[str, Any]:
    report = _mapping(value, "pair report")
    if set(report) != _PAIR_FIELDS:
        raise ValueError("pair report fields are invalid")
    if report["schema_version"] != RL_SIM_REAL_PAIR_SCHEMA_VERSION:
        raise ValueError(
            f"pair report schema_version must be {RL_SIM_REAL_PAIR_SCHEMA_VERSION}"
        )
    for field in (
        "pair_id",
        "evaluation_scope",
        "trace_semantics",
        "simulation_run_id",
        "real_machine_run_id",
    ):
        _text(report[field], field)
    sample_counts = {
        "simulation": _positive_int(
            report["simulation_sample_count"], "simulation_sample_count"
        ),
        "real_machine": _positive_int(
            report["real_machine_sample_count"], "real_machine_sample_count"
        ),
    }
    aligned_count = _positive_int(
        report["aligned_sample_count"], "aligned_sample_count"
    )
    if aligned_count != min(sample_counts.values()):
        raise ValueError("aligned_sample_count must be the common-prefix length")
    for side in _SIDES:
        _validate_tail(report, side, aligned_count, sample_counts[side])
    _validate_sample_coverage(report["sample_coverage"], sample_counts)
    duration = _exact_mapping(report["duration_s"], "duration_s", set(_SIDES))
    for side in _SIDES:
        _nonnegative_float(duration[side], f"duration_s.{side}")
    trace_sha256 = _exact_mapping(report["trace_sha256"], "trace_sha256", set(_SIDES))
    for side in _SIDES:
        _sha256(trace_sha256[side], f"trace_sha256.{side}")
    _rate(report["nonzero_agreement_rate"], "nonzero_agreement_rate")
    _validate_axes(report["axes"])
    _validate_tracking(report["tracking"], aligned_count=aligned_count)
    _validate_binding(report["binding"], report["trace_semantics"])
    return report


def _validate_tail(
    report: Mapping[str, Any],
    side: str,
    aligned_count: int,
    sample_count: int,
) -> None:
    count_field = f"{side}_only_tail_count"
    ids_field = f"{side}_only_tail_sample_ids"
    tail_count = _nonnegative_int(report[count_field], count_field)
    expected_ids = list(range(aligned_count, sample_count))
    if tail_count != len(expected_ids):
        raise ValueError(f"{count_field} does not match common-prefix tail")
    ids = report[ids_field]
    if not isinstance(ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in ids
    ):
        raise ValueError(f"{ids_field} must be an integer array")
    if ids != expected_ids:
        raise ValueError(f"{ids_field} must be the explicit contiguous tail")


def _validate_sample_coverage(
    value: object,
    sample_counts: Mapping[str, int],
) -> None:
    coverage = _exact_mapping(value, "sample_coverage", set(_SIDES))
    suite_counts: set[int] = set()
    for side in _SIDES:
        side_coverage = _exact_mapping(
            coverage[side],
            f"sample_coverage.{side}",
            {"consumed_count", "suite_count", "rate"},
        )
        consumed = _positive_int(
            side_coverage["consumed_count"],
            f"sample_coverage.{side}.consumed_count",
        )
        suite_count = _positive_int(
            side_coverage["suite_count"], f"sample_coverage.{side}.suite_count"
        )
        if consumed != sample_counts[side]:
            raise ValueError(f"sample_coverage.{side} consumed_count does not match")
        if consumed > suite_count:
            raise ValueError(f"sample_coverage.{side} exceeds trajectory suite")
        rate = _rate(side_coverage["rate"], f"sample_coverage.{side}.rate")
        if abs(rate - consumed / suite_count) > _EPSILON:
            raise ValueError(
                f"sample_coverage.{side} rate does not match consumed and suite counts"
            )
        suite_counts.add(suite_count)
    if len(suite_counts) != 1:
        raise ValueError("simulation and real-machine suite counts do not match")


def _validate_axes(value: object) -> None:
    axes = _exact_mapping(value, "axes", set(_ACTION_ORDER))
    for axis in _ACTION_ORDER:
        metrics = _exact_mapping(
            axes[axis],
            f"axes.{axis}",
            {"mae", "rmse", "max_abs", "sign_agreement_rate"},
        )
        for name in ("mae", "rmse", "max_abs"):
            _nonnegative_float(metrics[name], f"axes.{axis}.{name}")
        _rate(metrics["sign_agreement_rate"], f"axes.{axis}.sign_agreement_rate")


def _validate_tracking(value: object, *, aligned_count: int) -> None:
    tracking = _exact_mapping(
        value,
        "tracking",
        {
            "bucket_tip_euclidean_error_m",
            "reference_waypoint_euclidean_error_m",
            "waypoint_index_agreement",
            "relative_sample_timing_error_s",
            "waypoint_distance_m",
            "terminal_result",
        },
    )
    for group in (
        "bucket_tip_euclidean_error_m",
        "reference_waypoint_euclidean_error_m",
    ):
        metrics = _exact_mapping(
            tracking[group], f"tracking.{group}", {"mae", "rmse", "max"}
        )
        for name in ("mae", "rmse", "max"):
            _nonnegative_float(metrics[name], f"tracking.{group}.{name}")
    agreement = _exact_mapping(
        tracking["waypoint_index_agreement"],
        "tracking.waypoint_index_agreement",
        {"count", "rate"},
    )
    agreement_count = _nonnegative_int(
        agreement["count"], "tracking.waypoint_index_agreement.count"
    )
    if agreement_count > aligned_count:
        raise ValueError("waypoint index agreement count exceeds aligned samples")
    agreement_rate = _rate(
        agreement["rate"], "tracking.waypoint_index_agreement.rate"
    )
    if abs(agreement_rate - agreement_count / aligned_count) > _EPSILON:
        raise ValueError("waypoint index agreement rate does not match count")
    timing = _exact_mapping(
        tracking["relative_sample_timing_error_s"],
        "tracking.relative_sample_timing_error_s",
        {"mae", "rmse", "max_abs"},
    )
    for name in ("mae", "rmse", "max_abs"):
        _nonnegative_float(
            timing[name], f"tracking.relative_sample_timing_error_s.{name}"
        )
    waypoint = _exact_mapping(
        tracking["waypoint_distance_m"],
        "tracking.waypoint_distance_m",
        set(_SIDES),
    )
    for side in _SIDES:
        waypoint_side = _exact_mapping(
            waypoint[side],
            f"tracking.waypoint_distance_m.{side}",
            {"mean", "p95", "final"},
        )
        for name in ("mean", "p95", "final"):
            _nonnegative_float(
                waypoint_side[name], f"tracking.waypoint_distance_m.{side}.{name}"
            )
    _validate_terminal(tracking["terminal_result"])


def _validate_terminal(value: object) -> None:
    count_fields = {f"{terminal.lower()}_count" for terminal in _TERMINALS}
    terminal = _exact_mapping(
        value,
        "tracking.terminal_result",
        {"agreement", "simulation", "real_machine", *count_fields},
    )
    if not isinstance(terminal["agreement"], bool):
        raise ValueError("tracking.terminal_result.agreement must be bool")
    results = {side: terminal[side] for side in _SIDES}
    if any(result not in _TERMINALS for result in results.values()):
        raise ValueError("terminal result is invalid")
    if terminal["agreement"] != (results["simulation"] == results["real_machine"]):
        raise ValueError("terminal agreement does not match terminal results")
    for terminal_name in _TERMINALS:
        field = f"{terminal_name.lower()}_count"
        counts = _exact_mapping(terminal[field], f"tracking.terminal_result.{field}", set(_SIDES))
        for side in _SIDES:
            expected = int(results[side] == terminal_name)
            actual = _nonnegative_int(
                counts[side], f"tracking.terminal_result.{field}.{side}"
            )
            if actual != expected:
                raise ValueError("terminal count does not match terminal result")


def _validate_binding(value: object, top_trace_semantics: object) -> None:
    binding = _exact_mapping(value, "binding", set(_BINDING_FIELDS))
    _text(binding["trajectory_controller_policy_id"], "binding.trajectory_controller_policy_id")
    for field in (
        "trajectory_controller_onnx_sha256",
        "machine_profile_sha256",
        "trajectory_suite_sha256",
    ):
        _sha256(binding[field], f"binding.{field}")
    sample_period_s = _nonnegative_float(
        binding["trajectory_suite_sample_period_s"],
        "binding.trajectory_suite_sample_period_s",
    )
    if not math.isclose(sample_period_s, 0.1, rel_tol=0.0, abs_tol=_EPSILON):
        raise ValueError("binding.trajectory_suite_sample_period_s must be 0.1")
    if binding["trajectory_trace_schema_version"] != "excavator_rl_control_trace.v3":
        raise ValueError("binding.trajectory_trace_schema_version must be excavator_rl_control_trace.v3")
    if binding["action_order"] != list(_ACTION_ORDER):
        raise ValueError("binding.action_order must be [boom, stick, bucket, swing]")
    if binding["trace_semantics"] != top_trace_semantics:
        raise ValueError("binding trace semantics do not match pair report")


def _require_unique_ids(reports: Sequence[Mapping[str, Any]], field: str) -> None:
    values = tuple(report[field] for report in reports)
    if len(set(values)) != len(values):
        raise ValueError(f"attempted pair reports contain duplicate {field}")


def _distribution(values: Sequence[object]) -> dict[str, object]:
    numeric = sorted(_finite_float(value, "distribution value") for value in values)
    if not numeric:
        return {key: 0 if key == "count" else None for key in ("count", "mean", "median", "p95", "min", "max")}
    return {
        "count": len(numeric),
        "mean": _round(sum(numeric) / len(numeric)),
        "median": _round(_percentile(numeric, 0.5)),
        "p95": _round(_percentile(numeric, 0.95)),
        "min": _round(numeric[0]),
        "max": _round(numeric[-1]),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _count_rate(count: int, denominator: int) -> dict[str, object]:
    return {"count": count, "rate": None if denominator == 0 else _round(count / denominator)}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_mapping(
    value: object, label: str, fields: set[str]
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    if set(result) != fields:
        raise ValueError(f"{label} fields are invalid")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return text


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _nonnegative_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _rate(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative int")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _round(value: float) -> float:
    return round(value, 12)


__all__ = [
    "RL_SIM_REAL_AGGREGATE_SCHEMA_VERSION",
    "aggregate_rl_sim_real_pair_reports",
    "validate_rl_sim_real_binding",
    "write_rl_sim_real_aggregate",
]

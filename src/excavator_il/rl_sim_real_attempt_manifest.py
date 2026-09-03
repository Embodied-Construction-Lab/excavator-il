"""Pre-frozen attempt denominator for formal RL simulation-real studies."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from .rl_sim_real_aggregate import (
    RL_SIM_REAL_AGGREGATE_SCHEMA_VERSION,
    aggregate_rl_sim_real_pair_reports,
    validate_rl_sim_real_binding,
)


RL_SIM_REAL_ATTEMPT_MANIFEST_SCHEMA_VERSION = (
    "excavator_rl_sim_real_attempt_manifest.v1"
)
_MANIFEST_FIELDS = frozenset(
    {"schema_version", "evaluation_scope", "study_id", "binding", "attempts"}
)
_ATTEMPT_FIELDS = frozenset(
    {
        "attempt_id",
        "pair_id",
        "simulation_run_id",
        "real_machine_run_id",
        "pair_report_path",
    }
)
_ID_FIELDS = (
    "attempt_id",
    "pair_id",
    "simulation_run_id",
    "real_machine_run_id",
)
_SIDES = ("simulation", "real_machine")
_TERMINALS = ("completed", "timeout", "rejected", "interrupted")


def aggregate_rl_sim_real_attempt_manifest(
    attempt_manifest_path: str | Path,
) -> dict[str, object]:
    """Aggregate all frozen attempts while retaining absent pair evidence."""

    manifest_path = Path(attempt_manifest_path).expanduser()
    manifest, manifest_sha256 = _load_attempt_manifest(manifest_path)
    attempts = manifest["attempts"]
    present_attempts = tuple(item for item in attempts if item["present"])
    if present_attempts:
        base = aggregate_rl_sim_real_pair_reports(
            item["resolved_report_path"] for item in present_attempts
        )
        if base["binding"] != manifest["binding"]:
            raise ValueError("pair report binding does not match attempt manifest")
    else:
        base = _empty_trace_aggregate(manifest["binding"])

    paired_evidence = tuple(zip(present_attempts, base["attempted_pairs"]))
    for attempt, actual in paired_evidence:
        for field in ("pair_id", "simulation_run_id", "real_machine_run_id"):
            if actual[field] != attempt[field]:
                raise ValueError(f"pair report {field} does not match attempt manifest")
    actual_by_attempt = {
        attempt["attempt_id"]: actual for attempt, actual in paired_evidence
    }

    evidence = tuple(
        _attempt_evidence(attempt, actual_by_attempt.get(attempt["attempt_id"]))
        for attempt in attempts
    )
    denominator = len(attempts)
    trace_bearing_count = len(present_attempts)
    missing_count = denominator - trace_bearing_count
    return {
        **base,
        "study_id": manifest["study_id"],
        "attempt_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
        },
        "attempted_pair_count": denominator,
        "trace_bearing_pair_count": trace_bearing_count,
        "missing_pair_report_count": missing_count,
        "not_evaluable_attempt_count": missing_count,
        "evidence_complete": missing_count == 0,
        "attempted_pairs": list(evidence),
        "terminal": _terminal_with_missing(base["terminal"], denominator, missing_count),
    }


def _load_attempt_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"attempt manifest must be a regular file: {path}")
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read attempt manifest {path}: {exc}") from exc
    manifest = _exact_mapping(value, "attempt manifest", _MANIFEST_FIELDS)
    if manifest["schema_version"] != RL_SIM_REAL_ATTEMPT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("attempt manifest schema_version is invalid")
    if manifest["evaluation_scope"] != "held_out_experiment":
        raise ValueError("attempt manifest must use held_out_experiment scope")
    study_id = _text(manifest["study_id"], "study_id")
    binding = validate_rl_sim_real_binding(manifest["binding"])
    raw_attempts = manifest["attempts"]
    if not isinstance(raw_attempts, list) or not raw_attempts:
        raise ValueError("attempt manifest attempts must be a non-empty array")
    root = path.parent.resolve()
    attempts = tuple(_normalize_attempt(item, root) for item in raw_attempts)
    for field in _ID_FIELDS:
        _require_unique(attempts, field)
    _require_unique(attempts, "resolved_report_path")
    return (
        {**manifest, "study_id": study_id, "binding": binding, "attempts": attempts},
        hashlib.sha256(payload).hexdigest(),
    )


def _normalize_attempt(value: object, root: Path) -> dict[str, object]:
    attempt = _exact_mapping(value, "attempt", _ATTEMPT_FIELDS)
    identifiers = {field: _text(attempt[field], field) for field in _ID_FIELDS}
    raw_path = Path(_text(attempt["pair_report_path"], "pair_report_path"))
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ValueError("pair_report_path must be relative and without traversal")
    candidate = root / raw_path
    if candidate.is_symlink():
        raise ValueError(f"pair report must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("pair_report_path must be relative and without traversal")
    return {
        **identifiers,
        "pair_report_path": str(raw_path),
        "resolved_report_path": resolved,
        "present": candidate.exists(),
    }


def _attempt_evidence(
    attempt: Mapping[str, Any], actual: Mapping[str, Any] | None
) -> dict[str, object]:
    if actual is not None:
        return {
            **actual,
            "attempt_id": attempt["attempt_id"],
            "evidence_status": "trace_bearing",
        }
    return {
        **{field: attempt[field] for field in _ID_FIELDS},
        "report_path": str(attempt["resolved_report_path"]),
        "report_sha256": None,
        "trace_sha256": None,
        "simulation_terminal": "UNKNOWN",
        "real_machine_terminal": "UNKNOWN",
        "evidence_status": "missing_pair_report",
    }


def _terminal_with_missing(
    known: Mapping[str, Any], denominator: int, missing_count: int
) -> dict[str, object]:
    known_count = denominator - missing_count
    agreement_count = int(known.get("agreement", {}).get("count", 0))
    return {
        "agreement": _count_rate(agreement_count, denominator),
        "evaluable": _count_rate(known_count, denominator),
        "not_evaluable": _count_rate(missing_count, denominator),
        **{
            side: {
                **{
                    terminal: _count_rate(
                        int(known.get(side, {}).get(terminal, {}).get("count", 0)),
                        denominator,
                    )
                    for terminal in _TERMINALS
                },
                "unknown": _count_rate(missing_count, denominator),
            }
            for side in _SIDES
        },
    }


def _empty_trace_aggregate(binding: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema_version": RL_SIM_REAL_AGGREGATE_SCHEMA_VERSION,
        "evaluation_scope": "held_out_experiment",
        "binding": dict(binding),
        "attempted_pairs": [],
        "terminal": {},
        "tracking": None,
        "duration_s": None,
        "sample_counts": None,
        "tails": None,
        "sample_coverage": None,
        "statistical_unit": "pair_run",
    }


def _count_rate(count: int, denominator: int) -> dict[str, object]:
    return {"count": count, "rate": round(count / denominator, 12)}


def _require_unique(attempts: tuple[Mapping[str, Any], ...], field: str) -> None:
    values = tuple(item[field] for item in attempts)
    if len(set(values)) != len(values):
        raise ValueError(f"attempt manifest contains duplicate {field}")


def _exact_mapping(
    value: object, label: str, fields: frozenset[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


__all__ = [
    "RL_SIM_REAL_ATTEMPT_MANIFEST_SCHEMA_VERSION",
    "aggregate_rl_sim_real_attempt_manifest",
]

"""Strict paired evaluation for RL simulation and real-machine trajectory traces."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .experiment_run import load_experiment_run
from .rl_sim_experiment_run import (
    RL_CONTROL_TRACE_SCHEMA_VERSION,
    RL_CONTROL_TRACE_SEMANTICS,
    load_machine_profile_action_order,
    load_rl_control_trace_snapshot,
    load_trajectory_suite_snapshot,
)


RL_SIM_REAL_PAIR_SCHEMA_VERSION = "excavator_rl_sim_real_pair.v1"
_PAIR_FIELDS = frozenset(
    {
        "schema_version",
        "pair_id",
        "evaluation_scope",
        "parent_runs",
        "binding",
        "artifacts",
    }
)
_PARENT_KEYS = frozenset({"simulation", "real_machine"})
_PARENT_FIELDS = frozenset({"run_id", "run_kind", "run_path", "manifest_sha256"})
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
_ARTIFACT_FIELDS = frozenset({"artifact_id", "role", "snapshot_path", "sha256"})
_ACTION_ORDER = ("boom", "stick", "bucket", "swing")
_AXES = ("boom", "stick", "bucket", "swing")
_EPSILON = 1e-9


def write_rl_sim_real_pair_manifest(
    output_path: str | Path,
    *,
    simulation_run_path: str | Path,
    real_run_path: str | Path,
    pair_id: str,
) -> Path:
    sim_binding = _parent_binding(load_experiment_run(simulation_run_path), parent_label="simulation")
    real_binding = _parent_binding(load_experiment_run(real_run_path), parent_label="real_machine")
    if sim_binding["policy_id"] != real_binding["policy_id"]:
        raise ValueError("trajectory controller policy binding does not match parent runs")
    if sim_binding["onnx_sha256"] != real_binding["onnx_sha256"]:
        raise ValueError("trajectory controller ONNX binding does not match parent runs")
    if sim_binding["machine_profile_sha256"] != real_binding["machine_profile_sha256"]:
        raise ValueError("machine profile binding does not match parent runs")
    if sim_binding["action_order"] != real_binding["action_order"]:
        raise ValueError("action order binding does not match parent runs")
    if sim_binding["trajectory_suite_sha256"] != real_binding["trajectory_suite_sha256"]:
        raise ValueError("trajectory suite binding does not match parent runs")
    if (
        sim_binding["trajectory_suite_sample_period_s"]
        != real_binding["trajectory_suite_sample_period_s"]
    ):
        raise ValueError("trajectory suite sample period does not match parent runs")
    if sim_binding["trace_semantics"] != real_binding["trace_semantics"]:
        raise ValueError("trajectory trace semantics do not match parent runs")
    if sim_binding["trace_schema_version"] != real_binding["trace_schema_version"]:
        raise ValueError("trajectory trace schema does not match parent runs")
    if sim_binding["evaluation_scope"] != real_binding["evaluation_scope"]:
        raise ValueError("evaluation scope does not match parent runs")
    manifest = {
        "schema_version": RL_SIM_REAL_PAIR_SCHEMA_VERSION,
        "pair_id": _text(pair_id, "pair_id"),
        "evaluation_scope": sim_binding["evaluation_scope"],
        "parent_runs": {
            "simulation": sim_binding["parent_ref"],
            "real_machine": real_binding["parent_ref"],
        },
        "binding": {
            "trajectory_controller_policy_id": sim_binding["policy_id"],
            "trajectory_controller_onnx_sha256": sim_binding["onnx_sha256"],
            "machine_profile_sha256": sim_binding["machine_profile_sha256"],
            "action_order": list(sim_binding["action_order"]),
            "trajectory_suite_sha256": sim_binding["trajectory_suite_sha256"],
            "trajectory_suite_sample_period_s": sim_binding[
                "trajectory_suite_sample_period_s"
            ],
            "trajectory_trace_schema_version": sim_binding[
                "trace_schema_version"
            ],
            "trace_semantics": sim_binding["trace_semantics"],
        },
        "artifacts": {
            "simulation": sim_binding["trace_artifact"],
            "real_machine": real_binding["trace_artifact"],
        },
    }
    target = Path(output_path).expanduser()
    _atomic_write_json(
        target,
        manifest,
    )
    return target


def evaluate_rl_sim_real_pair(manifest_path: str | Path) -> dict[str, object]:
    manifest = _load_manifest(Path(manifest_path).expanduser())
    sim_binding = _parent_binding(
        load_experiment_run(manifest["parent_runs"]["simulation"]["run_path"]),
        parent_label="simulation",
    )
    real_binding = _parent_binding(
        load_experiment_run(manifest["parent_runs"]["real_machine"]["run_path"]),
        parent_label="real_machine",
    )
    _require_manifest_match(
        manifest["parent_runs"]["simulation"],
        sim_binding["parent_ref"],
    )
    _require_manifest_match(
        manifest["parent_runs"]["real_machine"],
        real_binding["parent_ref"],
    )
    _require_trace_artifact_match(
        manifest["artifacts"]["simulation"],
        sim_binding["trace_artifact"],
    )
    _require_trace_artifact_match(
        manifest["artifacts"]["real_machine"],
        real_binding["trace_artifact"],
    )
    if (
        manifest["evaluation_scope"] != sim_binding["evaluation_scope"]
        or manifest["evaluation_scope"] != real_binding["evaluation_scope"]
    ):
        raise ValueError("evaluation scope does not match parent runs")
    binding = manifest["binding"]
    if binding["trajectory_controller_policy_id"] != sim_binding["policy_id"] or binding["trajectory_controller_policy_id"] != real_binding["policy_id"]:
        raise ValueError("trajectory controller policy binding does not match parent runs")
    if binding["trajectory_controller_onnx_sha256"] != sim_binding["onnx_sha256"] or binding["trajectory_controller_onnx_sha256"] != real_binding["onnx_sha256"]:
        raise ValueError("trajectory controller ONNX binding does not match parent runs")
    if binding["machine_profile_sha256"] != sim_binding["machine_profile_sha256"] or binding["machine_profile_sha256"] != real_binding["machine_profile_sha256"]:
        raise ValueError("machine profile binding does not match parent runs")
    if tuple(binding["action_order"]) != sim_binding["action_order"] or tuple(binding["action_order"]) != real_binding["action_order"]:
        raise ValueError("action order binding does not match parent runs")
    if binding["trajectory_suite_sha256"] != sim_binding["trajectory_suite_sha256"]:
        raise ValueError("trajectory suite binding does not match simulation parent run")
    if binding["trajectory_suite_sha256"] != real_binding["trajectory_suite_sha256"]:
        raise ValueError("trajectory suite binding does not match real-machine parent run")
    if (
        binding["trajectory_suite_sample_period_s"]
        != sim_binding["trajectory_suite_sample_period_s"]
        or binding["trajectory_suite_sample_period_s"]
        != real_binding["trajectory_suite_sample_period_s"]
    ):
        raise ValueError("trajectory suite sample period does not match parent runs")
    if binding["trace_semantics"] != sim_binding["trace_semantics"] or binding["trace_semantics"] != real_binding["trace_semantics"]:
        raise ValueError("trajectory trace semantics do not match parent runs")
    if (
        binding["trajectory_trace_schema_version"]
        != sim_binding["trace_schema_version"]
        or binding["trajectory_trace_schema_version"]
        != real_binding["trace_schema_version"]
    ):
        raise ValueError("trajectory trace schema does not match parent runs")
    sim_document = sim_binding["trace_document"]
    real_document = real_binding["trace_document"]
    sim_trace = {sample.sample_id: sample for sample in sim_document.samples}
    real_trace = {sample.sample_id: sample for sample in real_document.samples}
    aligned_count = min(len(sim_trace), len(real_trace))
    aligned = list(range(aligned_count))
    if not aligned:
        raise ValueError("sim-real pair has no aligned trajectory samples")
    simulation_only_tail = list(range(aligned_count, len(sim_trace)))
    real_machine_only_tail = list(range(aligned_count, len(real_trace)))
    axes: dict[str, object] = {}
    nonzero_matches = 0
    for axis_index, axis_name in enumerate(_AXES):
        diffs: list[float] = []
        sign_matches = 0
        for sample_id in aligned:
            sim_value = sim_trace[sample_id].action[axis_index]
            real_value = real_trace[sample_id].action[axis_index]
            diffs.append(real_value - sim_value)
            if _sign(sim_value) == _sign(real_value):
                sign_matches += 1
        axes[axis_name] = {
            "mae": _round(sum(abs(value) for value in diffs) / len(diffs)),
            "rmse": _round(math.sqrt(sum(value * value for value in diffs) / len(diffs))),
            "max_abs": _round(max(abs(value) for value in diffs)),
            "sign_agreement_rate": _round(sign_matches / len(diffs)),
        }
    waypoint_index_agreement_count = 0
    for sample_id in aligned:
        if sim_trace[sample_id].waypoint_index == real_trace[sample_id].waypoint_index:
            waypoint_index_agreement_count += 1
        if _is_nonzero(sim_trace[sample_id].action) == _is_nonzero(real_trace[sample_id].action):
            nonzero_matches += 1
    bucket_tip_errors = [
        math.dist(
            sim_trace[sample_id].bucket_tip_ros_m,
            real_trace[sample_id].bucket_tip_ros_m,
        )
        for sample_id in aligned
    ]
    reference_waypoint_errors = [
        math.dist(
            sim_trace[sample_id].reference_waypoint_ros_m,
            real_trace[sample_id].reference_waypoint_ros_m,
        )
        for sample_id in aligned
    ]
    sim_first_stamp_s = sim_trace[0].stamp_s
    real_first_stamp_s = real_trace[0].stamp_s
    relative_timing_errors = [
        (real_trace[sample_id].stamp_s - real_first_stamp_s)
        - (sim_trace[sample_id].stamp_s - sim_first_stamp_s)
        for sample_id in aligned
    ]
    tracking = {
        "bucket_tip_euclidean_error_m": _error_summary(bucket_tip_errors),
        "reference_waypoint_euclidean_error_m": _error_summary(
            reference_waypoint_errors
        ),
        "waypoint_index_agreement": {
            "count": waypoint_index_agreement_count,
            "rate": _round(waypoint_index_agreement_count / len(aligned)),
        },
        "relative_sample_timing_error_s": _signed_error_summary(
            relative_timing_errors
        ),
        "waypoint_distance_m": {
            "simulation": _series_summary(
                [sample.waypoint_distance_m for sample in sim_document.samples]
            ),
            "real_machine": _series_summary(
                [sample.waypoint_distance_m for sample in real_document.samples]
            ),
        },
        "terminal_result": _terminal_result_summary(
            sim_document.terminal.result,
            real_document.terminal.result,
        ),
    }
    suite_sample_count = sim_binding["trajectory_suite_sample_count"]
    return {
        "schema_version": RL_SIM_REAL_PAIR_SCHEMA_VERSION,
        "pair_id": manifest["pair_id"],
        "evaluation_scope": manifest["evaluation_scope"],
        "trace_semantics": binding["trace_semantics"],
        "simulation_run_id": sim_binding["snapshot"].run_id,
        "real_machine_run_id": real_binding["snapshot"].run_id,
        "simulation_sample_count": len(sim_trace),
        "real_machine_sample_count": len(real_trace),
        "aligned_sample_count": len(aligned),
        "simulation_only_tail_count": len(simulation_only_tail),
        "simulation_only_tail_sample_ids": simulation_only_tail,
        "real_machine_only_tail_count": len(real_machine_only_tail),
        "real_machine_only_tail_sample_ids": real_machine_only_tail,
        "sample_coverage": {
            "simulation": {
                "consumed_count": len(sim_trace),
                "suite_count": suite_sample_count,
                "rate": _round(len(sim_trace) / suite_sample_count),
            },
            "real_machine": {
                "consumed_count": len(real_trace),
                "suite_count": suite_sample_count,
                "rate": _round(len(real_trace) / suite_sample_count),
            },
        },
        "duration_s": {
            "simulation": _round(sim_document.terminal.elapsed_s),
            "real_machine": _round(real_document.terminal.elapsed_s),
        },
        "trace_sha256": {
            "simulation": sim_binding["trace_artifact"]["sha256"],
            "real_machine": real_binding["trace_artifact"]["sha256"],
        },
        "nonzero_agreement_rate": _round(nonzero_matches / len(aligned)),
        "axes": axes,
        "tracking": tracking,
        "binding": dict(binding),
    }


def _parent_binding(snapshot: Any, *, parent_label: str) -> dict[str, Any]:
    if getattr(snapshot, "manifest", None) is None:
        raise ValueError(f"{parent_label} parent run must be finalized")
    start = snapshot.start
    final = snapshot.final or {}
    expected_run_kind = "evaluation" if parent_label == "simulation" else "hybrid_live"
    if start.get("run_kind") != expected_run_kind:
        raise ValueError(
            f"{parent_label} parent run_kind must be {expected_run_kind}"
        )
    policy_ids = _mapping(start.get("policy_ids"), "start.policy_ids")
    policy_id = _text(policy_ids.get("trajectory_controller"), "trajectory_controller policy")
    machine_profile = _mapping(start.get("machine_profile"), "start.machine_profile")
    machine_profile_sha256 = _text(machine_profile.get("sha256"), "machine_profile.sha256")
    machine_profile_snapshot = snapshot.run_dir / _text(
        machine_profile.get("snapshot_path"), "machine_profile.snapshot_path"
    )
    action_order = load_machine_profile_action_order(machine_profile_snapshot)
    onnx_artifact = _single_artifact(snapshot.artifacts, role="rl_onnx_model")
    trace_artifact = _single_artifact(snapshot.artifacts, role="trajectory_trace")
    suite_artifact = _single_artifact(snapshot.artifacts, role="trajectory_suite")
    trajectory_suite_sha256 = _text(
        suite_artifact.get("sha256"),
        "trajectory_suite.sha256",
    )
    suite_snapshot_path = snapshot.run_dir / _text(
        suite_artifact.get("snapshot_path"), "trajectory_suite.snapshot_path"
    )
    trajectory_suite, actual_suite_sha256 = load_trajectory_suite_snapshot(
        suite_snapshot_path
    )
    if actual_suite_sha256 != trajectory_suite_sha256:
        raise ValueError("trajectory suite artifact SHA does not match parent run")
    trace_snapshot_path = snapshot.run_dir / _text(
        trace_artifact.get("snapshot_path"), "trace snapshot_path"
    )
    trace_document, actual_trace_sha256 = load_rl_control_trace_snapshot(
        trace_snapshot_path
    )
    if actual_trace_sha256 != _text(trace_artifact.get("sha256"), "trace sha256"):
        raise ValueError("trajectory trace artifact SHA does not match parent run")
    trace_samples = trace_document.samples
    if {sample.trajectory_suite_sha256 for sample in trace_samples} != {
        trajectory_suite_sha256
    }:
        raise ValueError(
            f"{parent_label} trajectory trace suite binding does not match parent run"
        )
    trace_metadata = _mapping(
        trace_artifact.get("metadata"), "trajectory_trace.metadata"
    )
    if trace_metadata.get("trajectory_suite_sha256") != trajectory_suite_sha256:
        raise ValueError(
            f"{parent_label} trajectory trace artifact suite binding does not "
            "match parent run"
        )
    if {sample.sample_id for sample in trace_samples} - set(
        trajectory_suite["sample_ids"]
    ):
        raise ValueError(
            f"{parent_label} trajectory trace contains sample IDs outside its "
            "trajectory suite"
        )
    trace_semantics = {sample.trace_semantics for sample in trace_samples}
    if trace_semantics != {RL_CONTROL_TRACE_SEMANTICS}:
        raise ValueError(
            "trajectory trace semantics must be commanded_normalized_action"
        )
    manifest_path = snapshot.run_dir / "manifest.json"
    evaluation_scope = _evaluation_scope(
        _mapping(final.get("metrics", {}), "final.metrics").get("evaluation_scope"),
    )
    if parent_label == "real_machine":
        audit_artifact = _single_artifact(
            snapshot.artifacts,
            role="rl_control_audit",
        )
        _validate_real_audit_binding(
            audit_artifact,
            trace_artifact,
            trajectory_suite_sha256=trajectory_suite_sha256,
        )
    return {
        "snapshot": snapshot,
        "policy_id": policy_id,
        "onnx_sha256": _text(onnx_artifact.get("sha256"), "rl_onnx_model.sha256"),
        "machine_profile_sha256": machine_profile_sha256,
        "action_order": action_order,
        "trajectory_suite_sha256": trajectory_suite_sha256,
        "trace_semantics": RL_CONTROL_TRACE_SEMANTICS,
        "trace_schema_version": RL_CONTROL_TRACE_SCHEMA_VERSION,
        "evaluation_scope": evaluation_scope,
        "trajectory_suite_sample_count": len(trajectory_suite["sample_ids"]),
        "trajectory_suite_sample_period_s": trajectory_suite["sample_period_s"],
        "trace_document": trace_document,
        "trace_artifact": {
            "artifact_id": _text(trace_artifact.get("artifact_id"), "trace artifact_id"),
            "role": "trajectory_trace",
            "snapshot_path": _text(trace_artifact.get("snapshot_path"), "trace snapshot_path"),
            "sha256": _text(trace_artifact.get("sha256"), "trace sha256"),
        },
        "parent_ref": {
            "run_id": snapshot.run_id,
            "run_kind": _text(start.get("run_kind"), "run_kind"),
            "run_path": str(snapshot.run_dir),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    value = _read_json_object(path, label="sim-real pair manifest")
    if set(value) != _PAIR_FIELDS:
        raise ValueError(
            "sim-real pair manifest fields are invalid; expected "
            f"{sorted(_PAIR_FIELDS)}"
        )
    if value["schema_version"] != RL_SIM_REAL_PAIR_SCHEMA_VERSION:
        raise ValueError(
            f"sim-real pair manifest schema_version must be {RL_SIM_REAL_PAIR_SCHEMA_VERSION}"
        )
    value["pair_id"] = _text(value["pair_id"], "pair_id")
    value["evaluation_scope"] = _text(value["evaluation_scope"], "evaluation_scope")
    parents = _mapping(value["parent_runs"], "parent_runs")
    if set(parents) != _PARENT_KEYS:
        raise ValueError("parent_runs must contain simulation and real_machine")
    for name in _PARENT_KEYS:
        parent = _mapping(parents[name], f"parent_runs.{name}")
        if set(parent) != _PARENT_FIELDS:
            raise ValueError(f"parent_runs.{name} fields are invalid")
    binding = _mapping(value["binding"], "binding")
    if set(binding) != _BINDING_FIELDS:
        raise ValueError("binding fields are invalid")
    action_order = binding["action_order"]
    if (
        not isinstance(action_order, list)
        or len(action_order) != 4
        or any(not isinstance(item, str) or not item.strip() for item in action_order)
    ):
        raise ValueError("binding.action_order must contain four non-empty axis names")
    artifacts = _mapping(value["artifacts"], "artifacts")
    if set(artifacts) != _PARENT_KEYS:
        raise ValueError("artifacts must contain simulation and real_machine")
    for name in _PARENT_KEYS:
        artifact = _mapping(artifacts[name], f"artifacts.{name}")
        if set(artifact) != _ARTIFACT_FIELDS:
            raise ValueError(f"artifacts.{name} fields are invalid")
    return value


def _single_artifact(artifacts: Any, *, role: str) -> Mapping[str, object]:
    matches = [artifact for artifact in artifacts if artifact.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {role} artifact")
    return _mapping(matches[0], f"artifact[{role}]")


def _validate_real_audit_binding(
    audit_artifact: Mapping[str, object],
    trace_artifact: Mapping[str, object],
    *,
    trajectory_suite_sha256: str,
) -> None:
    audit_metadata = _mapping(
        audit_artifact.get("metadata"), "rl_control_audit.metadata"
    )
    trace_metadata = _mapping(
        trace_artifact.get("metadata"), "trajectory_trace.metadata"
    )
    if audit_metadata.get("schema_version") != "orin_edge_control_audit.v1":
        raise ValueError("real-machine audit schema binding is invalid")
    if audit_metadata.get("trajectory_controller_backend") != "onnx_rl":
        raise ValueError("real-machine audit backend binding is invalid")
    if (
        audit_metadata.get("trace_semantics") != RL_CONTROL_TRACE_SEMANTICS
        or trace_metadata.get("trace_semantics") != RL_CONTROL_TRACE_SEMANTICS
    ):
        raise ValueError("real-machine audit trace semantics binding is invalid")
    trace_run_id = _text(audit_metadata.get("trace_run_id"), "audit trace_run_id")
    if trace_metadata.get("trace_run_id") != trace_run_id:
        raise ValueError("real-machine trace_run_id binding is invalid")
    if trace_metadata.get("source_audit_sha256") != audit_artifact.get("sha256"):
        raise ValueError("real-machine trace source audit SHA binding is invalid")
    if (
        audit_metadata.get("trajectory_suite_sha256")
        != trajectory_suite_sha256
        or trace_metadata.get("trajectory_suite_sha256")
        != trajectory_suite_sha256
    ):
        raise ValueError("real-machine trajectory suite SHA binding is invalid")


def _require_manifest_match(
    manifest_parent: Mapping[str, object],
    actual_parent: Mapping[str, object],
) -> None:
    if _text(manifest_parent.get("run_id"), "parent run_id") != _text(actual_parent.get("run_id"), "actual run_id"):
        raise ValueError("parent run_id does not match finalized run")
    if _text(manifest_parent.get("manifest_sha256"), "parent manifest_sha256") != _text(
        actual_parent.get("manifest_sha256"), "actual manifest_sha256"
    ):
        raise ValueError("parent manifest SHA does not match finalized run")


def _require_trace_artifact_match(
    manifest_artifact: Mapping[str, object],
    parent_artifact: Mapping[str, object],
) -> None:
    if dict(manifest_artifact) != dict(parent_artifact):
        raise ValueError(
            "trajectory trace artifact does not match finalized parent run"
        )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _evaluation_scope(value: object) -> str:
    scope = _text(value, "evaluation_scope")
    if scope not in {"training_internal", "held_out_experiment"}:
        raise ValueError(
            "evaluation_scope must be training_internal or held_out_experiment"
        )
    return scope


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"sim-real pair manifest output must be a regular file: {path}")
    if path.exists():
        raise ValueError(f"sim-real pair manifest output already exists: {path}")
    payload = (
        json.dumps(
            value,
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
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise ValueError(
                f"sim-real pair manifest output already exists: {path}"
            ) from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _is_nonzero(values: tuple[float, float, float, float]) -> bool:
    return any(abs(value) > _EPSILON for value in values)


def _sign(value: float) -> int:
    if value > _EPSILON:
        return 1
    if value < -_EPSILON:
        return -1
    return 0


def _round(value: float) -> float:
    return round(value, 12)


def _error_summary(values: list[float]) -> dict[str, float]:
    return {
        "mae": _round(sum(values) / len(values)),
        "rmse": _round(math.sqrt(sum(value * value for value in values) / len(values))),
        "max": _round(max(values)),
    }


def _signed_error_summary(values: list[float]) -> dict[str, float]:
    return {
        "mae": _round(sum(abs(value) for value in values) / len(values)),
        "rmse": _round(
            math.sqrt(sum(value * value for value in values) / len(values))
        ),
        "max_abs": _round(max(abs(value) for value in values)),
    }


def _series_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": _round(sum(values) / len(values)),
        "p95": _round(_percentile(values, 0.95)),
        "final": _round(values[-1]),
    }
def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _terminal_result_summary(
    simulation: str, real_machine: str
) -> dict[str, object]:
    result = {
        "agreement": simulation == real_machine,
        "simulation": simulation,
        "real_machine": real_machine,
    }
    for label in ("completed", "timeout", "rejected", "interrupted"):
        expected = label.upper()
        result[f"{label}_count"] = {
            "simulation": int(simulation == expected),
            "real_machine": int(real_machine == expected),
        }
    return result


__all__ = [
    "RL_SIM_REAL_PAIR_SCHEMA_VERSION",
    "evaluate_rl_sim_real_pair",
    "write_rl_sim_real_pair_manifest",
]

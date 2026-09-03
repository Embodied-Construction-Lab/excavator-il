#!/usr/bin/env python3
"""Inspect ICRA 2027 experiment readiness without touching hardware.

This entry point opens versioned JSON/Python assets only.  It intentionally has
no SSH, Docker, camera, serial, ROS, or motion-runtime integration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Callable

_REPOSITORY = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from excavator_il.hybrid_experiment_run import HybridExperimentRunConfig
from excavator_il.icra2027_experiment_profile import (
    Icra2027ExperimentProfile,
    Icra2027ExperimentSuite,
)
from excavator_il.resident_fixed_cycle_system import ResidentFixedCyclePcConfig
from excavator_il.rl_sim_real_pair import evaluate_rl_sim_real_pair


SCHEMA_VERSION = "excavator_icra2027_experiment_readiness_report.v1"
_LIVE_GATE_INPUTS = (
    (
        "engine_off_gate_evidence",
        "发动机关闭条件下完成启动、取消和终态归零验收。",
    ),
    ("single_scoop_gate_evidence", "保留一次完整单铲 commissioning 证据。"),
    ("multi_scoop_gate_evidence", "保留连续多铲 commissioning 证据。"),
    (
        "held_out_experiment_evidence",
        "建立独立 held-out 配置和正式 soil-reset 配对 block。",
    ),
)


def _binding(
    binding_id: str,
    path: Path | None,
    *,
    strict_load: str,
) -> dict[str, object]:
    return {
        "binding_id": binding_id,
        "path": None if path is None else str(path),
        "exists": False if path is None else path.exists(),
        "strict_load": strict_load,
    }


def _blocked(input_id: str, reason: str) -> dict[str, str]:
    return {"input_id": input_id, "reason": reason}


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if not (path.is_file() or path.is_dir()):
        raise ValueError(f"{label} must be a file or directory: {path}")


def _load_live_bindings(
    profile: Icra2027ExperimentProfile,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    profile.preflight()
    bindings = [
        _binding("experiment_profile", profile.source_path, strict_load="passed")
    ]
    runtime_path = profile.bindings["resident_fixed_cycle_config"]
    evidence_path = profile.bindings["hybrid_evidence_config"]
    if runtime_path is None or evidence_path is None:
        raise ValueError("runnable profile bindings are incomplete")
    ResidentFixedCyclePcConfig.load(runtime_path)
    evidence = HybridExperimentRunConfig.load(evidence_path)
    bindings.extend(
        (
            _binding(
                "resident_fixed_cycle_config",
                runtime_path,
                strict_load="passed",
            ),
            _binding(
                "hybrid_evidence_config", evidence_path, strict_load="passed"
            ),
        )
    )
    assets: list[dict[str, object]] = []
    _require_path(evidence.machine_profile_path, "machine_profile")
    assets.append(
        _binding(
            "machine_profile",
            evidence.machine_profile_path,
            strict_load="passed",
        )
    )
    for label, path in sorted(evidence.repository_paths.items()):
        _require_path(path, f"repository_paths.{label}")
        assets.append(
            _binding(f"repository:{label}", path, strict_load="path_checked")
        )
    for label, path in sorted(evidence.config_paths.items()):
        _require_path(path, f"config_paths.{label}")
        assets.append(
            _binding(f"config:{label}", path, strict_load="path_checked")
        )
    for artifact in sorted(evidence.artifacts, key=lambda item: item.artifact_id):
        _require_path(artifact.source_path, f"artifact:{artifact.artifact_id}")
        assets.append(
            _binding(
                f"artifact:{artifact.artifact_id}",
                artifact.source_path,
                strict_load="path_checked",
            )
        )
    return bindings, assets


def _planned_profile_report(
    profile: Icra2027ExperimentProfile,
) -> dict[str, object]:
    bindings = [
        _binding("experiment_profile", profile.source_path, strict_load="passed")
    ]
    for label, path in sorted(profile.bindings.items()):
        bindings.append(
            _binding(
                label,
                path,
                strict_load="not_bound" if path is None else "not_runnable",
            )
        )
    blocked_inputs = []
    if profile.profile_id == "proposed_hybrid":
        blocked_inputs.extend(
            (
                _blocked(
                    "tadps_live_resident_binding",
                    "TADPS 尚未绑定到同一 Resident Mission。",
                ),
                _blocked(
                    "held_out_experiment_evidence",
                    "完整主方法尚无 held-out evidence 配置。",
                ),
            )
        )
    return {
        "study_kind": profile.study_kind,
        "declared_readiness": profile.readiness,
        "formal_ready": False,
        "static_preflight_passed": True,
        "bindings": bindings,
        "assets": [],
        "blocked_inputs": blocked_inputs,
        "failure_reasons": [],
    }


def _load_tadps_evaluator(workspace_root: Path) -> Callable[..., dict[str, Any]]:
    airy_root = workspace_root / "AiryLidar"
    if str(airy_root) not in sys.path:
        sys.path.insert(0, str(airy_root))
    from mission.tadps_benchmark import evaluate_tadps_replay

    return evaluate_tadps_replay


def _load_tadps_exporter(workspace_root: Path) -> Callable[..., object]:
    airy_root = workspace_root / "AiryLidar"
    if str(airy_root) not in sys.path:
        sys.path.insert(0, str(airy_root))
    from mission.tadps_replay_export import export_tadps_candidate_replay

    return export_tadps_candidate_replay


def _load_tadps_live_capture(
    workspace_root: Path,
) -> tuple[type[object], Callable[[Path], dict[str, Any]]]:
    airy_root = workspace_root / "AiryLidar"
    if str(airy_root) not in sys.path:
        sys.path.insert(0, str(airy_root))
    from mission.tadps_live_capture import (
        TadpsLiveCaptureSession,
        load_selector_bridge_descriptor,
    )

    return TadpsLiveCaptureSession, load_selector_bridge_descriptor


def _strict_load_python_cli(path: Path, label: str) -> None:
    try:
        namespace = runpy.run_path(
            str(path),
            run_name="_icra2027_static_cli_preflight",
        )
    except Exception as exc:
        raise ValueError(f"{label} cannot be imported: {path}") from exc
    if not callable(namespace.get("build_parser")) or not callable(
        namespace.get("main")
    ):
        raise ValueError(f"{label} public CLI API is incomplete: {path}")


def _tadps_report(
    *,
    profile: Icra2027ExperimentProfile,
    workspace_root: Path,
    replay_path: Path | None,
) -> dict[str, object]:
    config_path = workspace_root / "AiryLidar/mission/config/tadps_benchmark.v1.json"
    evaluator_path = workspace_root / "AiryLidar/mission/tadps_benchmark.py"
    exporter_path = workspace_root / "AiryLidar/mission/tadps_replay_export.py"
    capture_module_path = workspace_root / "AiryLidar/mission/tadps_live_capture.py"
    capture_cli_path = (
        workspace_root
        / "AiryLidar/mission/scripts/capture_tadps_candidate_frames.py"
    )
    bridge_descriptor_path = (
        workspace_root
        / "AiryLidar/mission/bridges/"
        "excavator_dig_point_tadps_candidate_trace.v1.json"
    )
    candidate_example_path = (
        workspace_root
        / "AiryLidar/mission/config/tadps_selector_candidate_frame.v1.example.jsonl"
    )
    _require_path(config_path, "TADPS benchmark config")
    _require_path(evaluator_path, "TADPS evaluator")
    _require_path(exporter_path, "TADPS replay exporter")
    _require_path(capture_module_path, "TADPS live capture module")
    _require_path(capture_cli_path, "TADPS live capture CLI")
    _require_path(bridge_descriptor_path, "TADPS selector bridge descriptor")
    _require_path(candidate_example_path, "TADPS candidate-frame schema example")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evaluator = _load_tadps_evaluator(workspace_root)
    _load_tadps_exporter(workspace_root)
    capture_session_type, load_bridge_descriptor = _load_tadps_live_capture(
        workspace_root
    )
    if not callable(getattr(capture_session_type, "record_json", None)) or not callable(
        getattr(capture_session_type, "close", None)
    ):
        raise ValueError("TADPS live capture public API is incomplete")
    descriptor = load_bridge_descriptor(bridge_descriptor_path)
    patch_relative = descriptor.get("patch_file")
    if not isinstance(patch_relative, str):
        raise ValueError("TADPS selector bridge patch_file is invalid")
    bridge_patch_path = bridge_descriptor_path.parent / patch_relative
    _require_path(bridge_patch_path, "TADPS selector bridge patch")
    _strict_load_python_cli(capture_cli_path, "TADPS live capture CLI")
    smoke_replay = {
        "schema_version": "tadps_candidate_replay.v1",
        "frame_id": config.get("expected_frame_id"),
        "sequences": [
            {
                "sequence_id": "static-preflight",
                "frames": [
                    {
                        "frame_index": 0,
                        "stamp_s": 0.0,
                        "map_sha256": "0" * 64,
                        "candidates": [],
                    }
                ],
            }
        ],
    }
    evaluator(smoke_replay, config)
    assets = [
        _binding("benchmark_config", config_path, strict_load="passed"),
        _binding(
            "candidate_frame_schema_example",
            candidate_example_path,
            strict_load="path_checked",
        ),
        _binding(
            "candidate_replay_exporter", exporter_path, strict_load="imported"
        ),
        _binding(
            "live_candidate_frame_capture_module",
            capture_module_path,
            strict_load="imported",
        ),
        _binding(
            "live_candidate_frame_capture_cli",
            capture_cli_path,
            strict_load="imported",
        ),
        _binding(
            "selector_bridge_descriptor",
            bridge_descriptor_path,
            strict_load="passed",
        ),
        _binding(
            "selector_bridge_patch",
            bridge_patch_path,
            strict_load="sha256_verified",
        ),
        _binding("offline_evaluator", evaluator_path, strict_load="imported"),
    ]
    blocked_inputs: list[dict[str, str]] = [
        _blocked(
            "frozen_real_candidate_replay",
            "需要带真实 logger/export provenance 的冻结 replay；仅通过 schema 校验不足以关闭阻塞。",
        ),
        _blocked(
            "held_out_tadps_split",
            "需要冻结独立 held-out replay、标签与评估 split。",
        ),
    ]
    if replay_path is not None:
        replay = replay_path.expanduser().resolve()
        _require_path(replay, "TADPS candidate replay")
        evaluator(json.loads(replay.read_text(encoding="utf-8")), config)
        assets.append(
            _binding("frozen_candidate_replay", replay, strict_load="passed")
        )
    return {
        "study_kind": profile.study_kind,
        "declared_readiness": profile.readiness,
        "formal_ready": False,
        "static_preflight_passed": True,
        "bindings": [
            _binding(
                "experiment_profile", profile.source_path, strict_load="passed"
            )
        ],
        "assets": assets,
        "blocked_inputs": blocked_inputs,
        "failure_reasons": [],
    }


def _rl_sim_real_report(
    *, repository_root: Path, pair_manifest: Path | None
) -> dict[str, object]:
    from excavator_il.rl_real_control_trace import RlRealExperimentRunRequest
    from excavator_il.rl_sim_real_aggregate import (
        RL_SIM_REAL_AGGREGATE_SCHEMA_VERSION,
    )
    from excavator_il.rl_sim_real_attempt_manifest import (
        RL_SIM_REAL_ATTEMPT_MANIFEST_SCHEMA_VERSION,
    )
    from excavator_il.rl_sim_experiment_run import (
        RL_CONTROL_TRACE_SCHEMA_VERSION,
        RlControlTraceSample,
        RlControlTraceTerminal,
    )

    if "trace_run_id" not in RlRealExperimentRunRequest.__dataclass_fields__:
        raise ValueError("real trace producer must require trace_run_id")
    expected_sample_fields = {
        "sample_id",
        "stamp_s",
        "action_order",
        "action",
        "trace_semantics",
        "trajectory_suite_sha256",
        "bucket_tip_ros_m",
        "reference_waypoint_ros_m",
        "waypoint_index",
        "waypoint_distance_m",
        "episode_progress",
        "result",
    }
    expected_terminal_fields = {
        "stamp_s",
        "elapsed_s",
        "trace_semantics",
        "trajectory_suite_sha256",
        "result",
    }
    if RL_CONTROL_TRACE_SCHEMA_VERSION != "excavator_rl_control_trace.v3":
        raise ValueError("tracking trace producer must use canonical schema v3")
    if set(RlControlTraceSample.__dataclass_fields__) != expected_sample_fields:
        raise ValueError("tracking trace policy_sample contract is incomplete")
    if set(RlControlTraceTerminal.__dataclass_fields__) != expected_terminal_fields:
        raise ValueError("tracking trace terminal contract is incomplete")
    workspace_root = repository_root.parent
    source_paths = {
        "simulation_control_audit_producer": workspace_root
        / "RLExcavator/Assets/Scripts/Tasks/OpenLoopVelocityRecorder.cs",
        "simulation_control_audit_writer": workspace_root
        / "RLExcavator/Assets/Scripts/Tasks/StrictRlSimControlAudit.cs",
        "simulation_run_recorder": repository_root
        / "src/excavator_il/rl_sim_experiment_run.py",
        "pair_evaluator": repository_root / "src/excavator_il/rl_sim_real_pair.py",
        "held_out_pair_aggregator": repository_root
        / "src/excavator_il/rl_sim_real_aggregate.py",
        "attempt_manifest_aggregator": repository_root
        / "src/excavator_il/rl_sim_real_attempt_manifest.py",
        "real_control_trace_producer": repository_root
        / "src/excavator_il/rl_real_control_trace.py",
        "real_control_audit_producer": workspace_root
        / "excavator-orin-runtime/edge_runtime/control.py",
        "real_control_audit_writer": workspace_root
        / "excavator-orin-runtime/edge_runtime/audit_writer.py",
        "simulation_control_trace_converter": repository_root
        / "src/excavator_il/rl_sim_trace_converter.py",
        "convert_simulation_trace_cli": repository_root
        / "scripts/convert_rl_sim_trace.py",
        "trajectory_suite_generator": repository_root
        / "scripts/create_rl_trajectory_suite.py",
        "record_cli": repository_root / "scripts/record_rl_sim_experiment_run.py",
        "record_real_parent_cli": repository_root
        / "scripts/record_rl_real_experiment_run.py",
        "evaluate_cli": repository_root / "scripts/evaluate_rl_sim_real_pair.py",
        "aggregate_pairs_cli": repository_root
        / "scripts/aggregate_rl_sim_real_pairs.py",
    }
    assets = []
    for label, path in sorted(source_paths.items()):
        _require_path(path, f"RL sim-real {label}")
        assets.append(_binding(label, path, strict_load="path_checked"))
    blocked_inputs: list[dict[str, str]] = [
        _blocked(
            "simulation_tracking_trace",
            "需要 Unity 真实策略决策边界产生的 suite-bound tracking trace。",
        ),
        _blocked(
            "real_tracking_trace",
            "需要 Orin Follow ACTIVE/terminal 证据产生的 suite-bound tracking trace。",
        ),
        _blocked(
            "frozen_attempt_manifest",
            "需要在任何 held-out run 前冻结并保留有序 attempt manifest。",
        ),
        _blocked(
            "held_out_paired_tracking_runs",
            "需要多个 held-out parent Run 配对，并由冻结 attempt manifest "
            "生成 evidence_complete=true 的 aggregate。",
        ),
    ]
    if pair_manifest is not None:
        resolved = pair_manifest.expanduser().resolve()
        _require_path(resolved, "RL sim-real pair manifest")
        evaluate_rl_sim_real_pair(resolved)
        assets.append(_binding("pair_manifest", resolved, strict_load="passed"))
    return {
        "study_kind": "tracking_parity_pipeline",
        "declared_readiness": "planned",
        "formal_ready": False,
        "tracking_trace_schema_version": RL_CONTROL_TRACE_SCHEMA_VERSION,
        "trajectory_suite_contract": {
            "exact_fields": ["suite_id", "sample_period_s", "sample_ids"],
            "suite_id": "nonempty_text",
            "sample_id_semantics": "elapsed_policy_decision_index",
            "sample_ids": "nonempty_contiguous_prefix_starting_at_zero",
            "sample_period_s": 0.1,
        },
        "pair_alignment_contract": {
            "alignment": "nonempty_common_prefix",
            "aggregate_denominator": "prefrozen_attempt_manifest_entries",
            "missing_tail_policy": "report_without_imputation",
            "zero_sample_attempt_policy": (
                "manifest_missing_pair_report_without_tracking_metrics"
            ),
            "required_outputs": [
                "simulation_only_tail_sample_ids",
                "real_machine_only_tail_sample_ids",
                "sample_coverage",
                "duration_s",
                "tracking.terminal_result",
                "tracking.bucket_tip_euclidean_error_m",
                "tracking.reference_waypoint_euclidean_error_m",
                "tracking.waypoint_index_agreement",
                "tracking.relative_sample_timing_error_s",
                "tracking.waypoint_distance_m",
            ],
        },
        "formal_evidence_contract": {
            "attempt_manifest_schema_version": (
                RL_SIM_REAL_ATTEMPT_MANIFEST_SCHEMA_VERSION
            ),
            "attempt_order": "prefrozen_and_preserved",
            "aggregate_schema_version": RL_SIM_REAL_AGGREGATE_SCHEMA_VERSION,
            "required_aggregate_evidence_complete": True,
            "missing_pair_report": (
                "retained_in_attempt_denominator_without_tracking_metrics"
            ),
            "pair_trace_binding": {
                "trajectory_trace_schema_version": (
                    "excavator_rl_control_trace.v3"
                ),
                "trace_sha256": ["simulation", "real_machine"],
            },
        },
        "static_preflight_passed": True,
        "bindings": [],
        "assets": assets,
        "blocked_inputs": blocked_inputs,
        "failure_reasons": [],
    }


def build_report(
    *,
    suite_path: Path,
    tadps_replay: Path | None = None,
    rl_pair_manifest: Path | None = None,
) -> tuple[dict[str, object], int]:
    suite_root = suite_path.expanduser().resolve()
    repository_root = suite_root.parents[2]
    workspace_root = repository_root.parent
    failures: list[str] = []
    experiments: dict[str, dict[str, object]] = {}
    try:
        suite = Icra2027ExperimentSuite.load_directory(suite_root)
        for profile_id, profile in suite.profiles.items():
            try:
                if profile_id == "tadps":
                    experiments[profile_id] = _tadps_report(
                        profile=profile,
                        workspace_root=workspace_root,
                        replay_path=tadps_replay,
                    )
                elif profile.readiness == "planned":
                    experiments[profile_id] = _planned_profile_report(profile)
                else:
                    bindings, assets = _load_live_bindings(profile)
                    experiments[profile_id] = {
                        "study_kind": profile.study_kind,
                        "declared_readiness": profile.readiness,
                        "formal_ready": profile.readiness == "ready",
                        "static_preflight_passed": True,
                        "bindings": bindings,
                        "assets": assets,
                        "blocked_inputs": (
                            []
                            if profile.readiness == "ready"
                            else [
                                _blocked(input_id, reason)
                                for input_id, reason in _LIVE_GATE_INPUTS
                            ]
                        ),
                        "failure_reasons": [],
                    }
            except (ImportError, OSError, TypeError, ValueError) as exc:
                message = f"experiment {profile_id}: {exc}"
                failures.append(message)
                experiments[profile_id] = {
                    "study_kind": profile.study_kind,
                    "declared_readiness": profile.readiness,
                    "formal_ready": False,
                    "static_preflight_passed": False,
                    "bindings": [],
                    "assets": [],
                    "blocked_inputs": [],
                    "failure_reasons": [str(exc)],
                }
        try:
            experiments["rl_sim_real"] = _rl_sim_real_report(
                repository_root=repository_root,
                pair_manifest=rl_pair_manifest,
            )
        except (ImportError, OSError, TypeError, ValueError) as exc:
            message = f"experiment rl_sim_real: {exc}"
            failures.append(message)
            experiments["rl_sim_real"] = {
                "study_kind": "paired_offline_evaluator",
                "declared_readiness": "planned",
                "formal_ready": False,
                "static_preflight_passed": False,
                "bindings": [],
                "assets": [],
                "blocked_inputs": [],
                "failure_reasons": [str(exc)],
            }
    except (OSError, TypeError, ValueError) as exc:
        failures.append(str(exc))

    ordered = dict(sorted(experiments.items()))
    readiness_counts = {"planned": 0, "commissioning": 0, "ready": 0}
    for experiment in ordered.values():
        readiness = experiment["declared_readiness"]
        if readiness in readiness_counts:
            readiness_counts[str(readiness)] += 1
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "suite_root": str(suite_root),
        "summary": {
            "commissioning": readiness_counts["commissioning"],
            "planned": readiness_counts["planned"],
            "ready": readiness_counts["ready"],
            "total": len(ordered),
        },
        "experiments": ordered,
        "failure_reasons": failures,
        "passed": not failures,
    }
    return report, 0 if not failures else 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repository = _REPOSITORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=repository / "config/experiments/icra2027",
    )
    parser.add_argument("--tadps-replay", type=Path)
    parser.add_argument("--rl-pair-manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    report, exit_code = build_report(
        suite_path=arguments.suite,
        tadps_replay=arguments.tadps_replay,
        rl_pair_manifest=arguments.rl_pair_manifest,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

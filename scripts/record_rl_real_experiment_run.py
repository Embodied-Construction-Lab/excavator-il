#!/usr/bin/env python3
"""Post-process an Orin RL audit and record a strict real-machine parent run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

_REPOSITORY = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from excavator_il.rl_real_control_trace import (
    RlRealExperimentRunRequest,
    record_rl_real_experiment_run,
)


def _kv_pairs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"expected label=path, got {value!r}")
        label, raw_path = value.split("=", 1)
        if not label or not raw_path or label in result:
            raise argparse.ArgumentTypeError(
                f"expected one unique non-empty label=path, got {value!r}"
            )
        result[label] = Path(raw_path).expanduser()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--machine-profile", required=True, type=Path)
    parser.add_argument("--trajectory-suite", required=True, type=Path)
    parser.add_argument(
        "--trajectory-suite-sha256",
        required=True,
        help=(
            "explicit SHA-256 of the frozen suite used by the selected real trace; "
            "the recorder validates it but never infers it"
        ),
    )
    parser.add_argument("--trajectory-controller-onnx", required=True, type=Path)
    parser.add_argument(
        "--control-audit",
        required=True,
        type=Path,
        help="offline orin_edge_control_audit.v1 JSONL; this command never contacts hardware",
    )
    parser.add_argument("--trace-output", required=True, type=Path)
    parser.add_argument(
        "--trace-run-id",
        required=True,
        help="explicit EdgeControlRunner segment identity from the append-only audit",
    )
    parser.add_argument("--policy-id", required=True)
    parser.add_argument(
        "--evaluation-scope",
        required=True,
        choices=("training_internal", "held_out_experiment"),
    )
    parser.add_argument("--task-variant", required=True)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--material-id", required=True)
    parser.add_argument("--soil-reset-block-id")
    parser.add_argument("--dig-point-id")
    parser.add_argument("--run-id")
    parser.add_argument("--repository-path", action="append", default=[])
    parser.add_argument("--config-path", action="append", default=[])
    args = parser.parse_args(argv)
    snapshot = record_rl_real_experiment_run(
        RlRealExperimentRunRequest(
            experiment_run_root=args.evidence_root,
            machine_profile_path=args.machine_profile,
            trajectory_suite_path=args.trajectory_suite,
            trajectory_controller_onnx_path=args.trajectory_controller_onnx,
            control_audit_path=args.control_audit,
            trace_output_path=args.trace_output,
            trace_run_id=args.trace_run_id,
            trajectory_suite_sha256=args.trajectory_suite_sha256,
            policy_id=args.policy_id,
            evaluation_scope=args.evaluation_scope,
            task_variant=args.task_variant,
            operator_id=args.operator_id,
            material_id=args.material_id,
            soil_reset_block_id=args.soil_reset_block_id,
            dig_point_id=args.dig_point_id,
            repository_paths=_kv_pairs(args.repository_path),
            config_paths=_kv_pairs(args.config_path),
            run_id=args.run_id,
        )
    )
    print(
        json.dumps(
            {
                "run_id": snapshot.run_id,
                "run_dir": str(snapshot.run_dir),
                "trace_output": str(args.trace_output.expanduser()),
                "trace_run_id": args.trace_run_id,
                "manifest_sha256": hashlib.sha256(
                    (snapshot.run_dir / "manifest.json").read_bytes()
                ).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

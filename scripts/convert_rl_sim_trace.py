#!/usr/bin/env python3
"""Convert one explicit Unity RL control audit segment into a paired trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from excavator_il.rl_sim_trace_converter import export_rl_sim_control_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-audit", required=True, type=Path)
    parser.add_argument("--trajectory-suite", required=True, type=Path)
    parser.add_argument("--trace-output", required=True, type=Path)
    parser.add_argument("--trace-run-id", required=True)
    args = parser.parse_args(argv)
    result = export_rl_sim_control_trace(
        args.control_audit,
        args.trace_output,
        trajectory_suite_path=args.trajectory_suite,
        trace_run_id=args.trace_run_id,
    )
    print(
        json.dumps(
            {
                "first_policy_action_seq": result.first_policy_action_seq,
                "first_sample_id": result.first_sample_id,
                "last_policy_action_seq": result.last_policy_action_seq,
                "last_sample_id": result.last_sample_id,
                "sample_count": result.sample_count,
                "terminal_result": result.terminal_result,
                "trace_output": str(result.output_path),
                "trace_run_id": result.trace_run_id,
                "trace_semantics": result.trace_semantics,
                "trajectory_suite_sha256": result.trajectory_suite_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

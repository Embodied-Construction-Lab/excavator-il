#!/usr/bin/env python3
"""Evaluate finalized Experiment Runs into deterministic JSON and CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from excavator_il.evaluation_harness import (
    EvaluationError,
    evaluate_experiment_runs,
    write_evaluation_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--aggregate-mode",
        choices=("homogeneous", "live_task", "collection_dataset"),
        default="homogeneous",
    )
    args = parser.parse_args(argv)
    try:
        aggregate = evaluate_experiment_runs(
            args.runs,
            aggregate_mode=args.aggregate_mode,
        )
        json_path, csv_path = write_evaluation_outputs(
            aggregate,
            json_path=args.output_dir / "evaluation_aggregate.json",
            csv_path=args.output_dir / "evaluation_aggregate.csv",
        )
    except (EvaluationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "json": str(json_path.resolve()),
                "csv": str(csv_path.resolve()),
                "run_count": aggregate["run_count"],
                "failed_run_count": aggregate["failed_run_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if aggregate["failed_run_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

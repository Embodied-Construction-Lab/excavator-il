#!/usr/bin/env python3
"""Benchmark one LeRobot ACT dataset representation without training."""

from __future__ import annotations

import argparse
import json

from excavator_il.training_input_benchmark import benchmark_training_input


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("repo_id")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measured-batches", type=int, default=20)
    args = parser.parse_args()
    try:
        result = benchmark_training_input(
            args.dataset_root,
            args.repo_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup_batches=args.warmup_batches,
            measured_batches=args.measured_batches,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {"passed": True, **result.to_dict()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

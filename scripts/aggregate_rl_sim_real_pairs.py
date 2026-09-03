#!/usr/bin/env python3
"""Aggregate a pre-frozen held-out RL sim-real attempt manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_REPOSITORY = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from excavator_il.rl_sim_real_aggregate import write_rl_sim_real_aggregate
from excavator_il.rl_sim_real_attempt_manifest import (
    aggregate_rl_sim_real_attempt_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        aggregate = aggregate_rl_sim_real_attempt_manifest(args.attempt_manifest)
        output = write_rl_sim_real_aggregate(args.output, aggregate)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "attempted_pair_count": aggregate["attempted_pair_count"],
                "evidence_complete": aggregate["evidence_complete"],
                "evaluation_scope": aggregate["evaluation_scope"],
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

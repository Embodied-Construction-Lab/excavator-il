#!/usr/bin/env python3
"""Build and/or evaluate a strict RL sim-real pair manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from excavator_il.rl_sim_real_pair import (
    evaluate_rl_sim_real_pair,
    write_rl_sim_real_pair_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--simulation-run", type=Path)
    parser.add_argument("--real-run", type=Path)
    parser.add_argument("--pair-id")
    args = parser.parse_args(argv)
    if (args.simulation_run is None) != (args.real_run is None):
        parser.error("--simulation-run and --real-run must be provided together")
    if args.simulation_run is not None:
        if not args.pair_id:
            parser.error("--pair-id is required when building a pair manifest")
        write_rl_sim_real_pair_manifest(
            args.pair_manifest,
            simulation_run_path=args.simulation_run,
            real_run_path=args.real_run,
            pair_id=args.pair_id,
        )
    report = evaluate_rl_sim_real_pair(args.pair_manifest)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

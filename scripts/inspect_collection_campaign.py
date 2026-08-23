#!/usr/bin/env python3
"""Inspect the deterministic 200-Episode collection campaign without writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from excavator_il.collection_campaign import inspect_collection_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_root",
        type=Path,
        nargs="?",
        help="directory containing raw episode_* directories",
    )
    parser.add_argument(
        "--collection-config",
        type=Path,
        help="collector config whose data_root is authoritative",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="print only the next expected slot and completion state",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.raw_root is None) == (args.collection_config is None):
        raise ValueError(
            "provide exactly one of raw_root or --collection-config"
        )
    if args.collection_config is not None:
        from excavator_il.collector.config import load_collection_config

        raw_root = load_collection_config(args.collection_config).data_root
    else:
        assert args.raw_root is not None
        raw_root = args.raw_root
    report = inspect_collection_campaign(raw_root)
    output: object = report
    if args.next:
        summary = report["summary"]
        output = {
            "schema_version": report["schema_version"],
            "raw_root": report["raw_root"],
            "complete_and_valid": summary["complete_and_valid"],
            "summary": {
                "planned": summary["planned"],
                "completed": summary["completed"],
                "ignored_diagnostics": summary["ignored_diagnostics"],
                "complete_and_valid": summary["complete_and_valid"],
            },
            "next_expected_slot": report["next_expected_slot"],
        }
    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"]["complete_and_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create one frozen 10 Hz decision-index suite for RL sim-real tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


_SUITE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SAMPLE_PERIOD_S = 0.1
_MAX_SAMPLE_COUNT = 1_000_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--sample-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _validate(parser: argparse.ArgumentParser, suite_id: str, count: int) -> None:
    if _SUITE_ID.fullmatch(suite_id) is None:
        parser.error(
            "--suite-id must use 1-128 ASCII letters, digits, '.', '_' or '-'"
        )
    if not 1 <= count <= _MAX_SAMPLE_COUNT:
        parser.error(
            f"--sample-count must be in [1, {_MAX_SAMPLE_COUNT}]"
        )


def _write_new_file(path: Path, payload: bytes) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {resolved.parent}")
    try:
        with resolved.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError(f"output already exists: {resolved}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(parser, args.suite_id, args.sample_count)
    output = args.output.expanduser().resolve()
    suite = {
        "sample_ids": list(range(args.sample_count)),
        "sample_period_s": _SAMPLE_PERIOD_S,
        "suite_id": args.suite_id,
    }
    payload = (
        json.dumps(suite, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        _write_new_file(output, payload)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = {
        "duration_s": round(args.sample_count * _SAMPLE_PERIOD_S, 10),
        "output": str(output),
        "sample_count": args.sample_count,
        "sample_period_s": _SAMPLE_PERIOD_S,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "suite_id": args.suite_id,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

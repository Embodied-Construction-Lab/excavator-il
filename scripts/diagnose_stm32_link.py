#!/usr/bin/env python3
"""Run the read-only USART2 integrity probe on Orin from the PC."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

from excavator_il.guided_episode import GuidedEpisodeConfig


_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/guided_episode.pc.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--duration-s", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or not 0.0 < args.duration_s <= 60.0:
        parser.error("--duration-s must be finite and within (0, 60]")

    config = GuidedEpisodeConfig.load(args.config)
    remote_timeout_s = args.duration_s + 5.0
    local_timeout_s = args.duration_s + 10.0
    argv = [
        "timeout",
        "--signal=TERM",
        "--kill-after=2s",
        f"{remote_timeout_s:g}s",
        str(config.orin_executable),
        "diagnose-stm32-link",
        "--config",
        str(config.orin_collection_config),
        "--duration-s",
        str(args.duration_s),
    ]
    remote_command = (
        f"cd {shlex.quote(str(config.orin_repo))} && {shlex.join(argv)}"
    )
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                config.orin_ssh_host,
                remote_command,
            ],
            check=False,
            timeout=local_timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(
            "error: STM32 diagnostic exceeded its local hard deadline; "
            "verify that /dev/ttyTHS1 was released on Orin",
            file=sys.stderr,
        )
        return 3
    if result.returncode == 124:
        print(
            "error: STM32 diagnostic timed out without completing; "
            "the remote timeout terminated it",
            file=sys.stderr,
        )
        return 3
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run deadman-gated PC teleoperation without recording an Episode."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from excavator_il.guided_episode import (
    GuidedEpisodeConfig,
    SystemGuidedEpisodeOperations,
    run_standalone_teleop,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/guided_episode.pc.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        config = GuidedEpisodeConfig.load(args.config)
        operations = SystemGuidedEpisodeOperations(config)
        run_standalone_teleop(config, operations, wait_fn=signal.pause)
    except KeyboardInterrupt:
        print("仅遥操作已停止；teleop 与 Collector 已退出，命令已回零。")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

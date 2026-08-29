"""Command-line lifecycle for the resident ACT worker."""

from __future__ import annotations

import argparse
import logging
import signal as _signal
import sys
from typing import Any, Callable


def run_cli(
    argv: list[str] | None,
    *,
    build_worker: Callable[..., Any],
    signal_module: Any = _signal,
) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Run the resident ACT policy worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--operator-observation-config")
    args = parser.parse_args(argv)
    worker = build_worker(
        args.config,
        socket_path=args.socket_path,
        operator_observation_config=args.operator_observation_config,
    )
    previous: dict[int, Any] = {}

    def stop(_signum: int, _frame: Any) -> None:
        worker.request_stop()

    try:
        for signum in (signal_module.SIGINT, signal_module.SIGTERM):
            previous[signum] = signal_module.signal(signum, stop)
        worker.run()
    finally:
        for signum, handler in previous.items():
            signal_module.signal(signum, handler)
    return 0

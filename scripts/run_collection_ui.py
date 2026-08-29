#!/usr/bin/env python3
"""Start the PC-local guided Demonstration Episode UI."""

from __future__ import annotations

import argparse
from pathlib import Path

from excavator_il.collection_ui_config import load_collection_ui_config
from excavator_il.collection_ui_process import collection_ui_process_lease
from excavator_il.collection_ui_runtime import run_collection_ui


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/collection_ui.pc.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="print the local URL without opening the default browser",
    )
    args = parser.parse_args()
    try:
        config_path = args.config.expanduser().resolve()
        config = load_collection_ui_config(config_path)
        with collection_ui_process_lease(
            config_path=config_path,
            host=config.host,
            port=config.port,
        ):
            run_collection_ui(config_path, open_browser=not args.no_browser)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

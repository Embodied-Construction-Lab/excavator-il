#!/usr/bin/env python3
"""Run the camera-only dual-RGB field preflight on Orin."""

from pathlib import Path

from excavator_il.camera_diagnostics import main


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[1]
    raise SystemExit(
        main(default_config_path=_ROOT / "config" / "collection.orin.json")
    )

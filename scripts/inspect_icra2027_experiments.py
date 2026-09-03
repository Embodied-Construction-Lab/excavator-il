#!/usr/bin/env python3
"""Validate the versioned ICRA 2027 experiment matrix."""

from pathlib import Path
import sys

_REPOSITORY = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from excavator_il.icra2027_experiment_profile import main


if __name__ == "__main__":
    raise SystemExit(main())

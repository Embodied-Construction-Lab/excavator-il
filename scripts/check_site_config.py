#!/usr/bin/env python3
"""Print the validated site topology used by active experiment configs."""

from __future__ import annotations

import json
from pathlib import Path

from excavator_il.site_config import check_site_config


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(check_site_config(root / "config"), indent=2))

#!/usr/bin/env python3
"""Create the isolated three-phase ACT training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="派生 DIG/TRANSPORT/DUMP 三阶段条件 ACT 数据集",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT
        / "data/lerobot/icra2027_transport_dump_dual_rgb_split_video",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPO_ROOT
        / "config/training/icra2027_transport_dump_three_phase_annotations.v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "data/lerobot/icra2027_transport_dump_dual_rgb_three_phase_video",
    )
    return parser.parse_args()


def main() -> None:
    from excavator_il.phase_conditioned_dataset import (
        derive_phase_conditioned_video_split,
    )

    args = parse_args()
    result = derive_phase_conditioned_video_split(
        source_root=args.source_root,
        annotation_path=args.annotations,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "output_root": str(result.root),
                "train_dataset_sha256": result.train_dataset_sha256,
                "validation_dataset_sha256": result.validation_dataset_sha256,
                "provenance_path": str(result.provenance_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

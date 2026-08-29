#!/usr/bin/env python3
"""Create the constant-image ACT state-only ablation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from excavator_il.observation_dataset_transform import derive_constant_image_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive an ACT ablation that preserves the visual architecture while "
            "replacing every RGB observation with the same black image."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--image-feature", default="observation.images.front"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = derive_constant_image_split(
        source_root=args.source_root,
        output_root=args.output_root,
        image_feature=args.image_feature,
    )
    print(
        json.dumps(
            {
                "schema_version": "excavator_state_only_derivation_result.v1",
                "root": str(result.root),
                "train_root": str(result.train_root),
                "validation_root": str(result.validation_root),
                "train_repo_id": result.train_repo_id,
                "validation_repo_id": result.validation_repo_id,
                "provenance_path": str(result.provenance_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

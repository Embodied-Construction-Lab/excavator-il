#!/usr/bin/env python3
"""Create an immutable video-backed derivative of one ACT train/val split."""

from __future__ import annotations

import argparse
import json

from excavator_il.video_training_dataset import derive_video_training_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_split_root")
    parser.add_argument("output_split_root")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames-per-batch", type=int, default=3000)
    args = parser.parse_args()
    try:
        result = derive_video_training_split(
            args.source_split_root,
            args.output_split_root,
            num_workers=args.num_workers,
            max_frames_per_batch=args.max_frames_per_batch,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "passed": True,
                "output_root": str(result.output_root),
                "train_dataset_sha256": result.train_dataset_sha256,
                "validation_dataset_sha256": result.validation_dataset_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

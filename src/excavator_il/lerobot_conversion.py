from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .raw_episode import ACTION_FIELDS, validate_episode


STATE_FIELDS = (
    "boom_pos_m",
    "stick_pos_m",
    "bucket_pos_m",
    "boom_vel_mps",
    "stick_vel_mps",
    "bucket_vel_mps",
    "boom_angle_rad",
    "arm_angle_rad",
    "bucket_angle_rad",
    "swing_angle_rad",
    "swing_vel_radps",
)

@dataclass(frozen=True)
class ConversionSummary:
    episode_count: int
    frame_count: int
    fps: int
    output_root: Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _causal_image_paths(
    episode_path: Path,
    steps: list[dict[str, str]],
    camera_rows: list[dict[str, str]],
) -> list[Path]:
    camera_timestamps = [int(row["camera_stamp_monotonic_ns"]) for row in camera_rows]
    camera_index = 0
    paths: list[Path] = []
    for step in steps:
        state_timestamp = int(step["state_receive_monotonic_ns"])
        while camera_index + 1 < len(camera_rows) and camera_timestamps[camera_index + 1] <= state_timestamp:
            camera_index += 1
        paths.append(episode_path / camera_rows[camera_index]["image_path"])
    return paths


def convert_episodes(
    episode_paths: list[str | Path],
    output_root: str | Path,
    repo_id: str,
    *,
    fps: int = 10,
) -> ConversionSummary:
    """Convert validated raw RGB episodes into a local LeRobotDataset v3."""
    if not episode_paths:
        raise ValueError("at least one episode path is required")
    validated_paths = [Path(path) for path in episode_paths]
    reports = [validate_episode(path) for path in validated_paths]
    image_shape = reports[0].image_shape
    if any(report.image_shape != image_shape for report in reports[1:]):
        raise ValueError("all episodes must use the same RGB image shape")

    root = Path(output_root)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_FIELDS),),
            "names": list(STATE_FIELDS),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_FIELDS),),
            "names": list(ACTION_FIELDS),
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="scale_excavator_v1",
        use_videos=True,
        image_writer_threads=1,
        vcodec="h264",
    )

    frame_count = 0
    try:
        for episode_path in validated_paths:
            metadata = json.loads((episode_path / "episode.json").read_text(encoding="utf-8"))
            steps = _read_rows(episode_path / "steps.csv")
            camera_rows = _read_rows(episode_path / "camera_front_timestamps.csv")
            image_paths = _causal_image_paths(episode_path, steps, camera_rows)
            task = f"excavate {metadata['material_id']} at configured dig target"

            for step, image_path in zip(steps, image_paths, strict=True):
                with Image.open(image_path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                dataset.add_frame(
                    {
                        "observation.state": np.asarray(
                            [float(step[field]) for field in STATE_FIELDS], dtype=np.float32
                        ),
                        "action": np.asarray(
                            [float(step[field]) for field in ACTION_FIELDS], dtype=np.float32
                        ),
                        "observation.images.front": rgb,
                        "task": task,
                    }
                )
                frame_count += 1
            dataset.save_episode()
    finally:
        dataset.finalize()

    return ConversionSummary(
        episode_count=len(validated_paths),
        frame_count=frame_count,
        fps=fps,
        output_root=root,
    )

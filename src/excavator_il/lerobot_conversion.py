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
    source_episode_count: int
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
    allow_synthetic: bool = False,
) -> ConversionSummary:
    """Convert validated raw RGB episodes into a local LeRobotDataset v3."""
    if not episode_paths:
        raise ValueError("at least one episode path is required")
    validated_paths = [Path(path) for path in episode_paths]
    resolved_paths = [path.resolve() for path in validated_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("duplicate Episode path is not allowed")
    metadata_by_path = {
        path: json.loads((path / "episode.json").read_text(encoding="utf-8"))
        for path in validated_paths
    }
    episode_ids = [metadata.get("episode_id") for metadata in metadata_by_path.values()]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate episode_id is not allowed")
    synthetic_paths = [
        path
        for path, metadata in metadata_by_path.items()
        if "synthetic_provenance" in metadata
    ]
    for path in synthetic_paths:
        provenance = metadata_by_path[path]["synthetic_provenance"]
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("source_episode_id"), str)
            or not provenance["source_episode_id"]
            or provenance.get("method") != "exact_duplicate_for_pipeline_validation"
            or provenance.get("training_eligible") is not False
        ):
            raise ValueError(f"invalid synthetic provenance: {path}")
    if synthetic_paths and not allow_synthetic:
        raise ValueError(
            f"synthetic Episode requires explicit --allow-synthetic: {synthetic_paths[0]}"
        )
    if synthetic_paths and len(synthetic_paths) != len(validated_paths):
        raise ValueError("real and synthetic Episodes must not be mixed in one dataset")
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
        "source.episode_id": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "source.segment_id": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "source.frame_index": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
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
        for episode_path, report in zip(validated_paths, reports, strict=True):
            metadata = json.loads((episode_path / "episode.json").read_text(encoding="utf-8"))
            steps = _read_rows(episode_path / "steps.csv")
            camera_rows = _read_rows(episode_path / "camera_front_timestamps.csv")
            image_paths = _causal_image_paths(episode_path, steps, camera_rows)
            task = f"excavate {metadata['material_id']} at configured dig target"

            for segment in report.training_segments:
                start = segment.start_frame_index
                end = segment.end_frame_index_exclusive
                for step, image_path in zip(
                    steps[start:end], image_paths[start:end], strict=True
                ):
                    with Image.open(image_path) as image:
                        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    dataset.add_frame(
                        {
                            "observation.state": np.asarray(
                                [float(step[field]) for field in STATE_FIELDS],
                                dtype=np.float32,
                            ),
                            "action": np.asarray(
                                [float(step[field]) for field in ACTION_FIELDS],
                                dtype=np.float32,
                            ),
                            "observation.images.front": rgb,
                            "source.episode_id": metadata["episode_id"],
                            "source.segment_id": segment.segment_id,
                            "source.frame_index": step["frame_index"],
                            "task": task,
                        }
                    )
                    frame_count += 1
                dataset.save_episode()
    finally:
        dataset.finalize()

    if synthetic_paths:
        (root / "pipeline_validation.json").write_text(
            json.dumps(
                {
                    "contains_synthetic_episodes": True,
                    "training_eligible": False,
                    "synthetic_episode_ids": [
                        metadata_by_path[path]["episode_id"] for path in synthetic_paths
                    ],
                    "source_episode_ids": sorted(
                        {
                            metadata_by_path[path]["synthetic_provenance"][
                                "source_episode_id"
                            ]
                            for path in synthetic_paths
                        }
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return ConversionSummary(
        source_episode_count=len(validated_paths),
        episode_count=sum(report.training_segment_count for report in reports),
        frame_count=frame_count,
        fps=fps,
        output_root=root,
    )

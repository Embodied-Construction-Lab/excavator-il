from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .raw_episode import ACTION_FIELDS, EpisodeValidationReport, validate_episode


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


def _load_lerobot_dataset_class():
    """Import the optional training dependency only for dataset conversion."""

    override = globals().get("LeRobotDataset")
    if override is not None:
        return override
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def __getattr__(name: str):
    if name == "LeRobotDataset":
        return _load_lerobot_dataset_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _verify_converted_dataset(
    root: Path,
    repo_id: str,
    *,
    expected_episode_count: int,
    expected_frame_count: int,
    expected_source_episode_ids: set[str],
) -> None:
    converted = _load_lerobot_dataset_class()(repo_id=repo_id, root=root)
    if converted.num_episodes != expected_episode_count:
        raise RuntimeError(
            "converted dataset episode count mismatch: "
            f"expected {expected_episode_count}, got {converted.num_episodes}"
        )
    if converted.num_frames != expected_frame_count:
        raise RuntimeError(
            "converted dataset frame count mismatch: "
            f"expected {expected_frame_count}, got {converted.num_frames}"
        )
    actual_source_episode_ids = set(converted.hf_dataset["source.episode_id"])
    if actual_source_episode_ids != expected_source_episode_ids:
        raise RuntimeError(
            "converted dataset source Episode IDs mismatch: "
            f"expected {sorted(expected_source_episode_ids)}, "
            f"got {sorted(actual_source_episode_ids)}"
        )


def _resolve_camera_roles(
    reports: list[EpisodeValidationReport],
    requested: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if requested is not None:
        if requested not in {("front",), ("front", "dump")}:
            raise ValueError(
                "camera_roles must be None, ('front',) or ('front', 'dump')"
            )
        return requested

    role_sets = {frozenset(report.cameras) for report in reports}
    if role_sets == {frozenset({"front"})}:
        return ("front",)
    if role_sets == {frozenset({"front", "dump"})}:
        return ("front", "dump")
    if len(role_sets) > 1:
        raise ValueError(
            "mixed camera contracts require explicit camera_roles"
        )
    raise ValueError("unsupported camera contract in validated Episodes")


def convert_episodes(
    episode_paths: list[str | Path],
    output_root: str | Path,
    repo_id: str,
    *,
    fps: int = 10,
    allow_synthetic: bool = False,
    camera_roles: tuple[str, ...] | None = None,
    task_variant_override: str | None = None,
) -> ConversionSummary:
    """Convert validated raw RGB episodes into a local LeRobotDataset v3."""
    if not episode_paths:
        raise ValueError("at least one episode path is required")
    if camera_roles is not None and camera_roles not in {
        ("front",),
        ("front", "dump"),
    }:
        raise ValueError(
            "camera_roles must be None, ('front',) or ('front', 'dump')"
        )
    if task_variant_override not in {
        None,
        "dig_only",
        "dig_transport_dump",
    }:
        raise ValueError(
            "task_variant_override must be dig_only or dig_transport_dump"
        )
    validated_paths = [Path(path) for path in episode_paths]
    resolved_paths = [path.resolve() for path in validated_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("duplicate Episode path is not allowed")
    metadata_by_path = {
        path: json.loads((path / "episode.json").read_text(encoding="utf-8"))
        for path in validated_paths
    }
    for path, metadata in metadata_by_path.items():
        recording_purpose = metadata.get(
            "recording_purpose", "demonstration"
        )
        if recording_purpose != "demonstration":
            raise ValueError(
                f"{recording_purpose} Episode is not eligible for training "
                f"conversion: {path}"
            )
        if (
            metadata.get("status") != "complete"
            or metadata.get("success") is not True
        ):
            raise ValueError(
                "only a successful complete demonstration is eligible for "
                f"training conversion: {path}"
            )
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
    camera_roles = _resolve_camera_roles(reports, camera_roles)
    image_shapes: dict[str, tuple[int, int, int]] = {}
    for camera_role in camera_roles:
        missing = [
            report.episode_id
            for report in reports
            if camera_role not in report.cameras
        ]
        if missing:
            raise ValueError(
                f"camera role {camera_role!r} is missing from Episode {missing[0]}"
            )
        image_shape = reports[0].cameras[camera_role].image_shape
        if any(
            report.cameras[camera_role].image_shape != image_shape
            for report in reports[1:]
        ):
            raise ValueError(
                f"all episodes must use the same {camera_role} RGB image shape"
            )
        image_shapes[camera_role] = image_shape

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
        **{
            f"observation.images.{camera_role}": {
                "dtype": "image",
                "shape": image_shapes[camera_role],
                "names": ["height", "width", "channel"],
            }
            for camera_role in camera_roles
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
        "source.task_variant": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "source.soil_reset_block_id": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "source.dig_point_id": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "source.collection_zone_id": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "source.dig_repeat_index": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    }
    dataset = _load_lerobot_dataset_class().create(
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
            metadata = json.loads(
                (episode_path / "episode.json").read_text(encoding="utf-8")
            )
            steps = _read_rows(episode_path / "steps.csv")
            image_paths_by_role = {
                camera_role: _causal_image_paths(
                    episode_path,
                    steps,
                    _read_rows(
                        episode_path
                        / f"camera_{camera_role}_timestamps.csv"
                    ),
                )
                for camera_role in camera_roles
            }
            protocol = metadata.get("collection_protocol", {})
            if not isinstance(protocol, dict):
                raise ValueError("episode.json collection_protocol must be an object")
            task_variant = task_variant_override or str(
                protocol.get("task_variant", "unknown")
            )
            soil_reset_block_id = str(
                protocol.get("soil_reset_block_id", "unknown")
            )
            dig_point_id = str(protocol.get("dig_point_id", "unknown"))
            collection_labels = metadata.get("collection_labels", {})
            if not isinstance(collection_labels, dict):
                raise ValueError("episode.json collection_labels must be an object")
            collection_zone_id = str(
                collection_labels.get("collection_zone_id", "unknown")
            )
            dig_repeat_index = str(
                collection_labels.get("dig_repeat_index", "unknown")
            )
            task = f"excavate {metadata['material_id']} at configured dig target"

            for segment in report.training_segments:
                start = segment.start_frame_index
                end = segment.end_frame_index_exclusive
                for row_offset, step in enumerate(steps[start:end], start=start):
                    rgb_by_role: dict[str, np.ndarray] = {}
                    for camera_role in camera_roles:
                        with Image.open(
                            image_paths_by_role[camera_role][row_offset]
                        ) as image:
                            rgb_by_role[camera_role] = np.asarray(
                                image.convert("RGB"), dtype=np.uint8
                            )
                    frame = {
                        "observation.state": np.asarray(
                            [float(step[field]) for field in STATE_FIELDS],
                            dtype=np.float32,
                        ),
                        "action": np.asarray(
                            [float(step[field]) for field in ACTION_FIELDS],
                            dtype=np.float32,
                        ),
                        "source.episode_id": metadata["episode_id"],
                        "source.segment_id": segment.segment_id,
                        "source.frame_index": step["frame_index"],
                        "source.task_variant": task_variant,
                        "source.soil_reset_block_id": soil_reset_block_id,
                        "source.dig_point_id": dig_point_id,
                        "source.collection_zone_id": collection_zone_id,
                        "source.dig_repeat_index": dig_repeat_index,
                        "task": task,
                        **{
                            f"observation.images.{camera_role}": rgb
                            for camera_role, rgb in rgb_by_role.items()
                        },
                    }
                    dataset.add_frame(frame)
                    frame_count += 1
                dataset.save_episode()
    finally:
        dataset.finalize()

    expected_episode_count = sum(report.training_segment_count for report in reports)
    _verify_converted_dataset(
        root,
        repo_id,
        expected_episode_count=expected_episode_count,
        expected_frame_count=frame_count,
        expected_source_episode_ids={str(episode_id) for episode_id in episode_ids},
    )

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
        episode_count=expected_episode_count,
        frame_count=frame_count,
        fps=fps,
        output_root=root,
    )

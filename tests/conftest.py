import csv
import json

import numpy as np
import pytest
from PIL import Image

from excavator_il.raw_episode import STEP_FIELDS


@pytest.fixture
def rgb_episode_factory(tmp_path):
    def create(
        episode_id: str = "episode_0001",
        step_count: int = 3,
        *,
        dual_camera: bool = False,
    ):
        episode = tmp_path / episode_id
        frames = episode / "camera_front"
        frames.mkdir(parents=True)

        metadata = {
            "episode_id": episode_id,
            "schema_version": (
                "excavator_demo_raw.v2"
                if dual_camera
                else "excavator_demo_raw.v1"
            ),
            "task": "ExecuteDig",
            "operator_id": "operator_01",
            "dig_target_m": [0.8, 0.1, -0.2],
            "material_id": "dry_soil_01",
            "status": "complete",
            "success": True,
            "failure_reason": "",
            "intervention": False,
            "firmware_commit": "abc1234",
            "urdf_hash": "urdf-sha256",
            "machine_profile_hash": "profile-sha256",
            "valve_calibration_id": "valve-v1",
            "pump_setting": "fixed_30_percent",
            "camera_front": {
                "device_id": "fixture-camera",
                "width": 32,
                "height": 24,
                "nominal_fps": 10,
                "pixel_format": "RGB8",
                "timestamp_clock": "CLOCK_MONOTONIC",
            },
        }
        if dual_camera:
            camera_front = metadata.pop("camera_front")
            metadata["cameras"] = {
                "front": camera_front,
                "dump": {
                    **camera_front,
                    "device_id": "fixture-dump-camera",
                },
            }
            metadata["collection_protocol"] = {
                "task_variant": "dig_transport_dump",
                "soil_reset_block_id": "block_01",
                "dig_point_id": "dig_01",
            }
        (episode / "episode.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        rows = []
        camera_rows = []
        for index in range(step_count):
            state_ns = 1_000_000_000 + index * 100_000_000
            row = {name: "0" for name in STEP_FIELDS}
            row.update(
                {
                    "episode_id": episode_id,
                    "frame_index": str(index),
                    "state_seq": str(100 + index),
                    "state_stamp_ms": str(5000 + index * 100),
                    "state_receive_monotonic_ns": str(state_ns),
                    "action_stamp_monotonic_ns": str(state_ns - 10_000_000),
                    "boom_pos_m": str(0.15 + index * 0.001),
                    "stick_pos_m": "0.14",
                    "bucket_pos_m": "0.13",
                    "boom_angle_rad": "0.5",
                    "arm_angle_rad": "1.0",
                    "bucket_angle_rad": "1.5",
                    "action_boom": "0.2",
                    "action_stick": "-0.3",
                    "action_bucket": "0.4",
                    "action_swing": "0.0",
                    "pump_percent": "30",
                    "sensor_valid": "1",
                    "control_mode": "manual_joystick",
                }
            )
            rows.append(row)

            image_name = f"{index:06d}.png"
            Image.fromarray(
                np.full((24, 32, 3), index * 20, dtype=np.uint8)
            ).save(frames / image_name)
            camera_rows.append(
                {
                    "camera_frame_index": str(index),
                    "camera_stamp_monotonic_ns": str(state_ns - 5_000_000),
                    "image_path": f"camera_front/{image_name}",
                }
            )

        with (episode / "steps.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        with (episode / "camera_front_timestamps.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=camera_rows[0].keys())
            writer.writeheader()
            writer.writerows(camera_rows)

        if dual_camera:
            dump_frames = episode / "camera_dump"
            dump_frames.mkdir()
            dump_rows = []
            for index in range(step_count):
                image_name = f"{index:06d}.png"
                Image.fromarray(
                    np.full((24, 32, 3), 100 + index * 20, dtype=np.uint8)
                ).save(dump_frames / image_name)
                state_ns = 1_000_000_000 + index * 100_000_000
                dump_rows.append(
                    {
                        "camera_frame_index": str(index),
                        "camera_stamp_monotonic_ns": str(state_ns - 8_000_000),
                        "image_path": f"camera_dump/{image_name}",
                    }
                )
            with (episode / "camera_dump_timestamps.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=dump_rows[0].keys())
                writer.writeheader()
                writer.writerows(dump_rows)

        return episode

    return create

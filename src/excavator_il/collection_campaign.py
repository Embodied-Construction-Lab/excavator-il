"""Deterministic planning and read-only auditing for the 200-Episode campaign."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any


COLLECTION_CAMPAIGN_SCHEMA_VERSION = "excavator_collection_campaign.v1"
RAW_EPISODE_SCHEMA_VERSION = "excavator_demo_raw.v2"
TASK_VARIANTS = ("dig_only", "dig_transport_dump")
DIG_POINT_IDS = ("dig_01", "dig_02", "dig_03")
COLLECTION_PROTOCOL_FIELDS = frozenset(
    {"task_variant", "soil_reset_block_id", "dig_point_id"}
)
RAW_EPISODE_STATUSES = frozenset(
    {"recording", "pending_review", "complete", "failed", "aborted"}
)
RECORDING_PURPOSES = frozenset({"demonstration", "diagnostic"})
_EPISODE_DIR = re.compile(r"episode_(\d+)$")


def build_default_campaign() -> dict[str, object]:
    """Return the frozen 20-block, 200-slot collection plan."""

    slots: list[dict[str, str]] = []
    for block_index in range(1, 21):
        task_variant = TASK_VARIANTS[(block_index - 1) % len(TASK_VARIANTS)]
        for within_block_index in range(10):
            slots.append(
                {
                    "slot_id": f"slot_{len(slots) + 1:03d}",
                    "task_variant": task_variant,
                    "soil_reset_block_id": f"block_{block_index:02d}",
                    "dig_point_id": DIG_POINT_IDS[
                        within_block_index % len(DIG_POINT_IDS)
                    ],
                }
            )
    return {
        "schema_version": COLLECTION_CAMPAIGN_SCHEMA_VERSION,
        "planned_episode_count": len(slots),
        "slots": slots,
    }


def _episode_entries(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        return []
    candidates: list[tuple[int, str, Path]] = []
    for child in raw_root.iterdir():
        match = _EPISODE_DIR.fullmatch(child.name)
        if match is not None:
            candidates.append((int(match.group(1)), child.name, child))
    return [item[2] for item in sorted(candidates)]


def _malformed_entry(episode_dir: Path, reason: str) -> dict[str, str]:
    return {
        "episode_id": episode_dir.name,
        "episode_path": str(episode_dir.resolve()),
        "reason": reason,
    }


def _load_authoritative_json(
    episode_dir: Path,
    filename: str,
    *,
    directory_fd: int,
) -> tuple[Any | None, dict[str, str] | None]:
    path = episode_dir / filename
    try:
        path_status = path.lstat()
    except FileNotFoundError:
        return None, _malformed_entry(episode_dir, f"missing {filename}")
    except OSError as exc:
        return None, _malformed_entry(
            episode_dir,
            f"invalid {filename}: {exc}",
        )
    if not stat.S_ISREG(path_status.st_mode):
        return None, _malformed_entry(
            episode_dir,
            f"{filename} must be a regular non-symbolic-link file",
        )
    file_fd = -1
    try:
        file_fd = os.open(
            filename,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        return None, _malformed_entry(episode_dir, f"missing {filename}")
    except OSError:
        return None, _malformed_entry(
            episode_dir,
            f"{filename} changed or could not be opened safely",
        )
    try:
        opened_status = os.fstat(file_fd)
    except OSError as exc:
        os.close(file_fd)
        return None, _malformed_entry(
            episode_dir,
            f"invalid {filename}: {exc}",
        )
    if (
        not stat.S_ISREG(opened_status.st_mode)
        or opened_status.st_dev != path_status.st_dev
        or opened_status.st_ino != path_status.st_ino
    ):
        os.close(file_fd)
        return None, _malformed_entry(
            episode_dir,
            f"{filename} changed or is not a stable regular file",
        )
    try:
        with os.fdopen(file_fd, "r", encoding="utf-8") as stream:
            file_fd = -1
            return json.load(stream), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _malformed_entry(
            episode_dir,
            f"invalid {filename}: {exc}",
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)


def _load_episode(
    episode_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        entry_status = episode_dir.lstat()
    except OSError as exc:
        return None, _malformed_entry(
            episode_dir,
            f"invalid Episode entry: {exc}",
        )
    if stat.S_ISLNK(entry_status.st_mode):
        return None, {
            "episode_id": episode_dir.name,
            "episode_path": str(episode_dir.absolute()),
            "reason": (
                "Episode entry must be a real directory, not a symbolic link"
            ),
        }
    if not stat.S_ISDIR(entry_status.st_mode):
        return None, _malformed_entry(
            episode_dir,
            "Episode entry must be a real directory",
        )
    try:
        directory_fd = os.open(
            episode_dir,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError:
        return None, {
            "episode_id": episode_dir.name,
            "episode_path": str(episode_dir.absolute()),
            "reason": "Episode entry changed or could not be opened safely",
        }
    try:
        opened_status = os.fstat(directory_fd)
    except OSError:
        os.close(directory_fd)
        return None, {
            "episode_id": episode_dir.name,
            "episode_path": str(episode_dir.absolute()),
            "reason": "Episode entry could not be inspected safely",
        }
    if (
        opened_status.st_dev != entry_status.st_dev
        or opened_status.st_ino != entry_status.st_ino
    ):
        os.close(directory_fd)
        return None, {
            "episode_id": episode_dir.name,
            "episode_path": str(episode_dir.absolute()),
            "reason": "Episode entry changed during inspection",
        }
    try:
        return _load_episode_from_directory(episode_dir, directory_fd)
    finally:
        os.close(directory_fd)


def _load_episode_from_directory(
    episode_dir: Path,
    directory_fd: int,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    metadata, malformed_entry = _load_authoritative_json(
        episode_dir,
        "episode.json",
        directory_fd=directory_fd,
    )
    if malformed_entry is not None:
        return None, malformed_entry
    if not isinstance(metadata, dict):
        return None, _malformed_entry(episode_dir, "episode.json must be an object")
    if metadata.get("schema_version") != RAW_EPISODE_SCHEMA_VERSION:
        return None, _malformed_entry(
            episode_dir,
            f"schema_version must be {RAW_EPISODE_SCHEMA_VERSION}",
        )
    if metadata.get("episode_id") != episode_dir.name:
        return None, _malformed_entry(
            episode_dir, "episode_id must match its Episode directory name"
        )
    recording_purpose = metadata.get("recording_purpose", "demonstration")
    if recording_purpose not in RECORDING_PURPOSES:
        return None, _malformed_entry(
            episode_dir,
            "recording_purpose must be demonstration or diagnostic",
        )
    status = metadata.get("status")
    success = metadata.get("success")
    if status not in RAW_EPISODE_STATUSES:
        return None, _malformed_entry(episode_dir, "episode status is invalid")
    if success is not None and not isinstance(success, bool):
        return None, _malformed_entry(
            episode_dir, "episode success must be true, false or null"
        )
    if recording_purpose == "diagnostic":
        return {
            "episode_id": episode_dir.name,
            "episode_path": str(episode_dir.resolve()),
            "recording_purpose": recording_purpose,
            "status": status,
            "success": success,
        }, None
    protocol = metadata.get("collection_protocol")
    if not isinstance(protocol, dict) or set(protocol) != COLLECTION_PROTOCOL_FIELDS:
        return None, _malformed_entry(
            episode_dir,
            "collection_protocol must contain exactly task_variant, "
            "soil_reset_block_id and dig_point_id",
        )
    if not all(isinstance(protocol[field], str) for field in COLLECTION_PROTOCOL_FIELDS):
        return None, _malformed_entry(
            episode_dir, "collection_protocol values must be text"
        )
    if status == "complete" and success is True:
        quality, malformed_entry = _load_authoritative_json(
            episode_dir,
            "quality_report.json",
            directory_fd=directory_fd,
        )
        if malformed_entry is not None:
            return None, malformed_entry
        if not isinstance(quality, dict):
            return None, _malformed_entry(
                episode_dir, "quality_report.json must be an object"
            )
        if quality.get("episode_id") != episode_dir.name:
            return None, _malformed_entry(
                episode_dir,
                "quality_report.json episode_id must match the Episode",
            )
        if quality.get("passed") is not True:
            return None, _malformed_entry(
                episode_dir, "quality_report.json passed must be true"
            )
    return {
        "episode_id": episode_dir.name,
        "episode_path": str(episode_dir.resolve()),
        **protocol,
        "status": status,
        "success": success,
    }, None


def inspect_collection_campaign(raw_root: str | Path) -> dict[str, object]:
    """Read raw Episode metadata and deterministically audit campaign progress."""

    root = Path(raw_root).expanduser()
    campaign = build_default_campaign()
    slots = campaign["slots"]
    assert isinstance(slots, list)
    slot_ids_by_key: dict[tuple[str, str, str], list[str]] = {}
    for slot in slots:
        assert isinstance(slot, dict)
        key = (
            slot["task_variant"],
            slot["soil_reset_block_id"],
            slot["dig_point_id"],
        )
        slot_ids_by_key.setdefault(key, []).append(slot["slot_id"])

    completed: list[dict[str, str]] = []
    failed: list[dict[str, Any]] = []
    duplicate: list[dict[str, Any]] = []
    unplanned: list[dict[str, Any]] = []
    malformed: list[dict[str, str]] = []
    ignored_diagnostics: list[dict[str, Any]] = []
    completed_slot_ids: set[str] = set()
    for episode_dir in _episode_entries(root):
        record, malformed_entry = _load_episode(episode_dir)
        if malformed_entry is not None:
            malformed.append(malformed_entry)
            continue
        if record is None:
            continue
        if record.get("recording_purpose") == "diagnostic":
            ignored_diagnostics.append(record)
            continue
        key = (
            record["task_variant"],
            record["soil_reset_block_id"],
            record["dig_point_id"],
        )
        if key not in slot_ids_by_key:
            unplanned.append(record)
            continue
        if record["status"] != "complete" or record["success"] is not True:
            failed.append(record)
            continue
        available = slot_ids_by_key[key]
        slot_id = next(
            (candidate for candidate in available if candidate not in completed_slot_ids),
            None,
        )
        if slot_id is None:
            duplicate.append(record)
            continue
        completed_slot_ids.add(slot_id)
        completed.append(
            {
                "slot_id": slot_id,
                "episode_id": record["episode_id"],
                "episode_path": record["episode_path"],
            }
        )

    missing = [
        dict(slot) for slot in slots if slot["slot_id"] not in completed_slot_ids
    ]
    summary = {
        "planned": len(slots),
        "completed": len(completed),
        "missing": len(missing),
        "failed": len(failed),
        "duplicate": len(duplicate),
        "unplanned": len(unplanned),
        "malformed": len(malformed),
        "ignored_diagnostics": len(ignored_diagnostics),
        "complete_and_valid": False,
    }
    summary["complete_and_valid"] = (
        summary["completed"] == summary["planned"]
        and summary["duplicate"] == 0
        and summary["unplanned"] == 0
        and summary["malformed"] == 0
    )
    return {
        "schema_version": COLLECTION_CAMPAIGN_SCHEMA_VERSION,
        "raw_root": str(root.resolve()),
        "planned_episode_count": len(slots),
        "slots": slots,
        "summary": summary,
        "completed": completed,
        "missing": missing,
        "failed": failed,
        "duplicate": duplicate,
        "unplanned": unplanned,
        "malformed": malformed,
        "ignored_diagnostics": ignored_diagnostics,
        "next_expected_slot": missing[0] if missing else None,
    }

"""Create isolated exact-duplicate Episodes for offline pipeline validation."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SyntheticEpisodeSummary:
    source_episode_id: str
    episode_ids: tuple[str, ...]
    output_root: Path
    image_storage: str
    training_eligible: bool


def _replace_episode_id(value: Any, source_id: str, target_id: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_episode_id(item, source_id, target_id)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_episode_id(item, source_id, target_id) for item in value]
    if isinstance(value, str):
        return value.replace(source_id, target_id)
    return value


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir()
        elif relative.parts[0] == "camera_front":
            os.link(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)


def _rewrite_json(path: Path, source_id: str, target_id: str) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value = _replace_episode_id(value, source_id, target_id)
    if path.name == "episode.json":
        value["synthetic_provenance"] = {
            "source_episode_id": source_id,
            "method": "exact_duplicate_for_pipeline_validation",
            "training_eligible": False,
        }
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_jsonl(path: Path, source_id: str, target_id: str) -> None:
    output = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                output.append(
                    json.dumps(
                        _replace_episode_id(value, source_id, target_id),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _rewrite_csv(path: Path, source_id: str, target_id: str) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "episode_id" not in fieldnames:
        return
    for row in rows:
        row["episode_id"] = row["episode_id"].replace(source_id, target_id)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def synthesize_episodes(
    source_episode: str | Path,
    output_root: str | Path,
    *,
    count: int,
) -> SyntheticEpisodeSummary:
    """Duplicate one Episode under unique synthetic IDs without changing the source."""
    source = Path(source_episode)
    root = Path(output_root)
    if count <= 0:
        raise ValueError("synthetic Episode count must be positive")
    if not source.is_dir():
        raise ValueError(f"source Episode does not exist: {source}")
    source_resolved = source.resolve()
    root_resolved = root.resolve()
    if root_resolved == source_resolved or root_resolved.is_relative_to(source_resolved):
        raise ValueError("synthetic output root must not be inside the source Episode")
    metadata_path = source / "episode.json"
    if not metadata_path.is_file():
        raise ValueError("source Episode is missing episode.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_id = metadata.get("episode_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source Episode has an invalid episode_id")

    episode_ids = tuple(f"synthetic_episode_{index:04d}" for index in range(1, count + 1))
    destinations = [root / episode_id for episode_id in episode_ids]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise ValueError(f"synthetic Episode destination already exists: {existing[0]}")

    root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".synthesize-", dir=root))
    published: list[Path] = []
    try:
        for target_id, destination in zip(episode_ids, destinations, strict=True):
            staging = staging_root / target_id
            _copy_tree(source, staging)
            for path in staging.iterdir():
                if path.suffix == ".json":
                    _rewrite_json(path, source_id, target_id)
                elif path.suffix == ".jsonl":
                    _rewrite_jsonl(path, source_id, target_id)
                elif path.suffix == ".csv":
                    _rewrite_csv(path, source_id, target_id)
            staging.rename(destination)
            published.append(destination)
    except BaseException:
        for destination in published:
            if destination.is_dir():
                shutil.rmtree(destination)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return SyntheticEpisodeSummary(
        source_episode_id=source_id,
        episode_ids=episode_ids,
        output_root=root,
        image_storage="hardlink",
        training_eligible=False,
    )

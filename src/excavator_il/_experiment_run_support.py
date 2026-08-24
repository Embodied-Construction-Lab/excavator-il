"""Private schema validation and atomic storage for Experiment Run evidence."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ._experiment_run_types import (
    EXPERIMENT_ARTIFACT_SCHEMA_VERSION,
    EXPERIMENT_EVENT_SCHEMA_VERSION,
    EXPERIMENT_INDEX_SCHEMA_VERSION,
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    EXPERIMENT_RUN_KINDS,
    EXPERIMENT_RUN_SCHEMA_VERSION,
    EvidenceRequirement,
    ExperimentRunError,
    ExperimentRunValidationError,
    TaskContext,
)


_TASK_CONTEXT_FIELDS: Final = frozenset(
    "task_variant soil_reset_block_id dig_point_id operator_id material_id".split())
_START_FIELDS: Final = frozenset(
    "schema_version run_id run_kind task_context policy_ids host_topology repositories "
    "config_snapshots machine_profile evidence_requirements started_at_utc "
    "started_monotonic_ns".split())
_EVENT_FIELDS: Final = frozenset(
    "schema_version sequence event_type wall_time_utc monotonic_ns payload".split())
_ARTIFACT_FIELDS: Final = frozenset(
    "schema_version sequence artifact_id role registered_at_utc monotonic_ns "
    "source_path snapshot_path snapshot_method object_type sha256 size_bytes "
    "file_count metadata".split())
_FINAL_FIELDS: Final = frozenset(
    "status finished_at_utc finished_monotonic_ns metrics summary".split())
_MANIFEST_FIELDS: Final = frozenset("schema_version start final evidence".split())
_INDEX_FIELDS: Final = frozenset(
    "schema_version run_id run_kind status task_context policy_ids started_at_utc "
    "finished_at_utc manifest_sha256".split()
)
_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40,64}$")
_MAX_CONFIG_BYTES: Final = 16 * 1024 * 1024
_MAX_JSONL_RECORD_BYTES: Final = 1024 * 1024


def _prepare_root(root: str | Path) -> Path:
    candidate = Path(root).expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ExperimentRunValidationError("experiment evidence root must not be a symlink")
    candidate.mkdir(parents=True, exist_ok=True)
    if not candidate.is_dir():
        raise ExperimentRunValidationError("experiment evidence root must be a directory")
    resolved = candidate.resolve(strict=True)
    for name in ("active", "runs", ".locks"):
        child = resolved / name
        if child.exists() and child.is_symlink():
            raise ExperimentRunValidationError(f"evidence {name} directory is a symlink")
        child.mkdir(mode=0o700, exist_ok=True)
        if not child.is_dir():
            raise ExperimentRunValidationError(f"evidence {name} is not a directory")
    return resolved


def _open_root(root: str | Path) -> Path:
    candidate = Path(root).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ExperimentRunValidationError(
            "experiment evidence root is unavailable or a symlink"
        )
    resolved = candidate.resolve(strict=True)
    for name in ("active", "runs", ".locks"):
        child = resolved / name
        if child.is_symlink() or not child.is_dir():
            raise ExperimentRunValidationError(
                f"evidence {name} directory is unavailable or a symlink"
            )
    return resolved


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    if path.exists() and path.is_symlink():
        raise ExperimentRunValidationError("experiment lock must not be a symlink")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _new_run_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"run_{timestamp}_{uuid.uuid4().hex[:10]}"


def _validate_run_id(value: str | None) -> str:
    if not isinstance(value, str) or not _RUN_ID_RE.fullmatch(value):
        raise ExperimentRunValidationError("run_id is invalid or contains path traversal")
    if value in {".", ".."} or ".." in value:
        raise ExperimentRunValidationError("run_id is invalid or contains path traversal")
    return value


def _validate_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ExperimentRunValidationError(f"{label} must be a safe non-empty identifier")
    if value in {".", ".."} or ".." in value:
        raise ExperimentRunValidationError(f"{label} contains path traversal")
    return value


def _validate_run_kind(value: Any) -> str:
    if value not in EXPERIMENT_RUN_KINDS:
        raise ExperimentRunValidationError(
            f"run_kind must be one of {sorted(EXPERIMENT_RUN_KINDS)}"
        )
    return str(value)


def _validate_text(
    value: Any, label: str, *, allow_empty: bool, max_length: int = 512
) -> str:
    if not isinstance(value, str):
        raise ExperimentRunValidationError(f"{label} must be a string")
    if not allow_empty and not value.strip():
        raise ExperimentRunValidationError(f"{label} must not be empty")
    if len(value) > max_length or any(ord(character) < 32 for character in value):
        raise ExperimentRunValidationError(f"{label} contains invalid text")
    return value


def _normalize_task_context(
    context: TaskContext | Mapping[str, Any],
) -> dict[str, str | None]:
    if isinstance(context, TaskContext):
        raw: Mapping[str, Any] = dataclasses.asdict(context)
    elif isinstance(context, Mapping):
        raw = context
    else:
        raise ExperimentRunValidationError("task_context must be TaskContext or mapping")
    if set(raw) != _TASK_CONTEXT_FIELDS:
        raise ExperimentRunValidationError(
            "task_context must contain exactly the five experiment context fields"
        )
    normalized: dict[str, str | None] = {}
    for key in sorted(_TASK_CONTEXT_FIELDS):
        value = raw[key]
        if value is None:
            if key in {"task_variant", "operator_id"}:
                raise ExperimentRunValidationError(f"task_context {key} is required")
            normalized[key] = None
        else:
            normalized[key] = _validate_text(
                value, f"task_context {key}", allow_empty=False
            )
    return normalized


def _normalize_string_mapping(values: Mapping[str, str], label: str) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise ExperimentRunValidationError(f"{label} must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in values.items():
        name = _validate_name(key, f"{label} key")
        normalized[name] = _validate_text(
            value, f"{label} value for {name}", allow_empty=False
        )
    return dict(sorted(normalized.items()))


def _normalize_path_mapping(
    values: Mapping[str, str | Path], label: str
) -> dict[str, Path]:
    if not isinstance(values, Mapping):
        raise ExperimentRunValidationError(f"{label} must be a mapping")
    normalized: dict[str, Path] = {}
    for key, value in values.items():
        normalized[_validate_name(key, f"{label} key")] = Path(value).expanduser()
    return dict(sorted(normalized.items()))


def _normalize_evidence_requirements(
    values: Mapping[str, EvidenceRequirement | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping):
        raise ExperimentRunValidationError("evidence_requirements must be a mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in values.items():
        role = _validate_name(key, "evidence role")
        if isinstance(value, EvidenceRequirement):
            requirement = value
        elif isinstance(value, Mapping) and set(value) == {"required", "min_count"}:
            requirement = EvidenceRequirement(
                required=value["required"], min_count=value["min_count"]
            )
        else:
            raise ExperimentRunValidationError(
                "evidence requirement must contain required and min_count"
            )
        normalized[role] = dataclasses.asdict(requirement)
    return dict(sorted(normalized.items()))


def _copy_json_value(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExperimentRunValidationError(f"{label} keys must be strings")
            copied[key] = _copy_json_value(item, label)
        return copied
    if isinstance(value, (list, tuple)):
        return [_copy_json_value(item, label) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ExperimentRunValidationError(f"{label} must contain finite JSON values")


def _normalize_json_mapping(values: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ExperimentRunValidationError(f"{label} must be a mapping")
    try:
        encoded = json.dumps(
            _copy_json_value(values, label),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ExperimentRunValidationError(
            f"{label} must contain finite JSON values"
        ) from exc


def _snapshot_config_file(
    snapshot_dir: Path, label: str, source_path: str | Path
) -> dict[str, Any]:
    source = _existing_regular_file(source_path, label=f"config {label}")
    if source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ExperimentRunValidationError(
            f"config {label} exceeds {_MAX_CONFIG_BYTES} bytes"
        )
    data = _read_regular_file_bytes(source)
    relative = Path("config_snapshots") / f"{label}.snapshot"
    _atomic_write_bytes(snapshot_dir.parent / relative, data, read_only=True)
    return {
        "source_path": str(source),
        "snapshot_path": relative.as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _existing_directory(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ExperimentRunValidationError(f"{label} must not be a symlink")
    if not candidate.is_dir():
        raise ExperimentRunValidationError(f"{label} directory does not exist: {candidate}")
    return candidate.resolve(strict=True)


def _existing_regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ExperimentRunValidationError(f"{label} must not be a symlink")
    if not candidate.is_file():
        raise ExperimentRunValidationError(f"{label} file does not exist: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ExperimentRunValidationError(f"{label} must be a regular file")
    return resolved


def _hash_regular_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExperimentRunValidationError(f"cannot safely open regular file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExperimentRunValidationError(f"path is not a regular file: {path}")
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or size != after.st_size
        ):
            raise ExperimentRunValidationError(f"file changed while hashing: {path}")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _read_regular_file_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExperimentRunValidationError(f"path is not a regular file: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or size != after.st_size
        ):
            raise ExperimentRunValidationError(f"file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise ExperimentRunValidationError(
            f"{label} fields are invalid; expected {sorted(expected)}"
        )


def _validate_start_manifest(start: Mapping[str, Any], run_dir: Path) -> None:
    if not isinstance(start, Mapping):
        raise ExperimentRunValidationError("start manifest must be an object")
    _require_exact_fields(start, _START_FIELDS, "start manifest")
    if start["schema_version"] != EXPERIMENT_RUN_SCHEMA_VERSION:
        raise ExperimentRunValidationError("start manifest schema_version is invalid")
    _validate_run_id(start["run_id"])
    _validate_run_kind(start["run_kind"])
    _normalize_task_context(start["task_context"])
    _normalize_string_mapping(start["policy_ids"], "policy_ids")
    _normalize_json_mapping(start["host_topology"], "host_topology")
    _normalize_evidence_requirements(start["evidence_requirements"])
    _validate_timestamp(start["started_at_utc"], "started_at_utc")
    _validate_nonnegative_int(start["started_monotonic_ns"], "started_monotonic_ns")

    repositories = start["repositories"]
    if not isinstance(repositories, Mapping):
        raise ExperimentRunValidationError("repositories must be an object")
    for name, record in repositories.items():
        _validate_name(name, "repository name")
        if not isinstance(record, Mapping):
            raise ExperimentRunValidationError("repository record must be an object")
        _require_exact_fields(
            record, frozenset({"source_path", "commit", "dirty"}), "repository"
        )
        if not Path(str(record["source_path"])).is_absolute():
            raise ExperimentRunValidationError("repository source_path must be absolute")
        if not isinstance(record["commit"], str) or not _GIT_COMMIT_RE.fullmatch(
            record["commit"]
        ):
            raise ExperimentRunValidationError("repository commit is invalid")
        if not isinstance(record["dirty"], bool):
            raise ExperimentRunValidationError("repository dirty must be bool")

    snapshots = start["config_snapshots"]
    if not isinstance(snapshots, Mapping):
        raise ExperimentRunValidationError("config_snapshots must be an object")
    for name, record in snapshots.items():
        _validate_name(name, "config snapshot name")
        _validate_snapshot_record(record, run_dir, f"config snapshot {name}")
    _validate_snapshot_record(start["machine_profile"], run_dir, "machine profile")


def _validate_snapshot_record(record: Any, run_dir: Path, label: str) -> None:
    if not isinstance(record, Mapping):
        raise ExperimentRunValidationError(f"{label} must be an object")
    _require_exact_fields(
        record,
        frozenset({"source_path", "snapshot_path", "sha256", "size_bytes"}),
        label,
    )
    if not Path(str(record["source_path"])).is_absolute():
        raise ExperimentRunValidationError(f"{label} source_path must be absolute")
    snapshot = run_dir / _safe_internal_relative_path(record["snapshot_path"], label)
    if snapshot.is_symlink() or not snapshot.is_file():
        raise ExperimentRunValidationError(f"{label} snapshot is unavailable or a symlink")
    _validate_sha(record["sha256"], f"{label} sha256")
    _validate_nonnegative_int(record["size_bytes"], f"{label} size_bytes")
    actual_digest, actual_size = _hash_regular_file(snapshot)
    if actual_digest != record["sha256"] or actual_size != record["size_bytes"]:
        raise ExperimentRunValidationError(f"{label} snapshot fingerprint mismatch")


def _validate_event_record(
    record: Mapping[str, Any], expected_sequence: int | None = None
) -> None:
    if not isinstance(record, Mapping):
        raise ExperimentRunValidationError("event record must be an object")
    _require_exact_fields(record, _EVENT_FIELDS, "event record")
    if record["schema_version"] != EXPERIMENT_EVENT_SCHEMA_VERSION:
        raise ExperimentRunValidationError("event schema_version is invalid")
    sequence = _validate_nonnegative_int(record["sequence"], "event sequence")
    if expected_sequence is not None and sequence != expected_sequence:
        raise ExperimentRunValidationError("event sequence is not contiguous")
    _validate_name(record["event_type"], "event_type")
    _validate_timestamp(record["wall_time_utc"], "event wall_time_utc")
    _validate_nonnegative_int(record["monotonic_ns"], "event monotonic_ns")
    _normalize_json_mapping(record["payload"], "event payload")


def _validate_artifact_record(
    record: Mapping[str, Any], expected_sequence: int | None = None
) -> None:
    if not isinstance(record, Mapping):
        raise ExperimentRunValidationError("artifact record must be an object")
    _require_exact_fields(record, _ARTIFACT_FIELDS, "artifact record")
    if record["schema_version"] != EXPERIMENT_ARTIFACT_SCHEMA_VERSION:
        raise ExperimentRunValidationError("artifact schema_version is invalid")
    sequence = _validate_nonnegative_int(record["sequence"], "artifact sequence")
    if expected_sequence is not None and sequence != expected_sequence:
        raise ExperimentRunValidationError("artifact sequence is not contiguous")
    _validate_name(record["artifact_id"], "artifact_id")
    _validate_name(record["role"], "artifact role")
    _validate_timestamp(record["registered_at_utc"], "artifact registered_at_utc")
    _validate_nonnegative_int(record["monotonic_ns"], "artifact monotonic_ns")
    if not isinstance(record["source_path"], str) or not Path(
        record["source_path"]
    ).is_absolute():
        raise ExperimentRunValidationError("artifact source_path must be absolute")
    _safe_artifact_snapshot_relative_path(record["snapshot_path"], "artifact")
    if record["snapshot_method"] not in {"reflink", "copy", "mixed"}:
        raise ExperimentRunValidationError("artifact snapshot_method is invalid")
    if record["object_type"] not in {"file", "directory"}:
        raise ExperimentRunValidationError("artifact object_type is invalid")
    _validate_sha(record["sha256"], "artifact sha256")
    _validate_nonnegative_int(record["size_bytes"], "artifact size_bytes")
    file_count = _validate_nonnegative_int(record["file_count"], "artifact file_count")
    if record["object_type"] == "file" and file_count != 1:
        raise ExperimentRunValidationError("file artifact file_count must equal one")
    _normalize_json_mapping(record["metadata"], "artifact metadata")


def _validate_final(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ExperimentRunValidationError("final record must be an object")
    _require_exact_fields(record, _FINAL_FIELDS, "final record")
    if record["status"] not in {"success", "failure"}:
        raise ExperimentRunValidationError("final status is invalid")
    _validate_timestamp(record["finished_at_utc"], "finished_at_utc")
    _validate_nonnegative_int(record["finished_monotonic_ns"], "finished_monotonic_ns")
    _normalize_json_mapping(record["metrics"], "final metrics")
    if record["summary"] is not None:
        _validate_text(
            record["summary"], "final summary", allow_empty=False, max_length=4096
        )


def _validate_manifest(manifest: Mapping[str, Any], run_dir: Path) -> None:
    if not isinstance(manifest, Mapping):
        raise ExperimentRunValidationError("final manifest must be an object")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "final manifest")
    if manifest["schema_version"] != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        raise ExperimentRunValidationError("final manifest schema_version is invalid")
    _validate_start_manifest(manifest["start"], run_dir)
    _validate_final(manifest["final"])
    evidence = manifest["evidence"]
    if not isinstance(evidence, Mapping):
        raise ExperimentRunValidationError("manifest evidence must be an object")
    evidence_fields = frozenset(
        {
            "start_sha256",
            "events_jsonl_sha256",
            "artifacts_jsonl_sha256",
            "event_count",
            "artifact_count",
        }
    )
    _require_exact_fields(evidence, evidence_fields, "manifest evidence")
    for key in ("start_sha256", "events_jsonl_sha256", "artifacts_jsonl_sha256"):
        _validate_sha(evidence[key], f"manifest {key}")
    _validate_nonnegative_int(evidence["event_count"], "manifest event_count")
    _validate_nonnegative_int(evidence["artifact_count"], "manifest artifact_count")
    expected = {
        "start_sha256": _file_sha256(run_dir / "start.json"),
        "events_jsonl_sha256": _file_sha256(run_dir / "events.jsonl"),
        "artifacts_jsonl_sha256": _file_sha256(run_dir / "artifacts.jsonl"),
    }
    for key, digest in expected.items():
        if evidence[key] != digest:
            raise ExperimentRunValidationError(
                f"{key.replace('_sha256', '')} evidence fingerprint mismatch"
            )


def _validate_index(index: Mapping[str, Any], manifest_path: Path) -> None:
    if not isinstance(index, Mapping):
        raise ExperimentRunValidationError("run index must be an object")
    _require_exact_fields(index, _INDEX_FIELDS, "run index")
    if index["schema_version"] != EXPERIMENT_INDEX_SCHEMA_VERSION:
        raise ExperimentRunValidationError("run index schema_version is invalid")
    _validate_run_id(index["run_id"])
    _validate_run_kind(index["run_kind"])
    if index["status"] not in {"success", "failure"}:
        raise ExperimentRunValidationError("run index status is invalid")
    _normalize_task_context(index["task_context"])
    _normalize_string_mapping(index["policy_ids"], "run index policy_ids")
    _validate_timestamp(index["started_at_utc"], "run index started_at_utc")
    _validate_timestamp(index["finished_at_utc"], "run index finished_at_utc")
    _validate_sha(index["manifest_sha256"], "run index manifest_sha256")
    if index["manifest_sha256"] != _file_sha256(manifest_path):
        raise ExperimentRunValidationError("run index manifest fingerprint mismatch")


def _require_evidence(
    requirements: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]
) -> None:
    counts = Counter(str(artifact["role"]) for artifact in artifacts)
    missing = [
        f"{role} ({counts[role]}/{requirement['min_count']})"
        for role, requirement in requirements.items()
        if requirement["required"] and counts[role] < requirement["min_count"]
    ]
    if missing:
        raise ExperimentRunValidationError(
            "required evidence is missing: " + ", ".join(missing)
        )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunValidationError(f"{label} is unavailable or a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentRunValidationError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ExperimentRunValidationError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, validator: Any) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunValidationError(f"{path.name} is unavailable or a symlink")
    records: list[dict[str, Any]] = []
    previous_monotonic = -1
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if len(raw_line.encode("utf-8")) > _MAX_JSONL_RECORD_BYTES:
                    raise ExperimentRunValidationError(
                        f"{path.name} line {line_number} is too large"
                    )
                if not raw_line.endswith("\n"):
                    raise ExperimentRunValidationError(
                        f"{path.name} has an incomplete final record"
                    )
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ExperimentRunValidationError(
                        f"{path.name} line {line_number} is invalid JSON"
                    ) from exc
                try:
                    validator(record, expected_sequence=len(records))
                except ExperimentRunValidationError as exc:
                    raise ExperimentRunValidationError(
                        f"{path.name} line {line_number}: {exc}"
                    ) from exc
                monotonic_ns = int(record["monotonic_ns"])
                if monotonic_ns < previous_monotonic:
                    raise ExperimentRunValidationError(
                        f"{path.name} monotonic timestamp regressed"
                    )
                previous_monotonic = monotonic_ns
                records.append(record)
    except (OSError, UnicodeError) as exc:
        raise ExperimentRunValidationError(f"{path.name} cannot be read") from exc
    return records


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    data = _canonical_json_bytes(value) + b"\n"
    if len(data) > _MAX_JSONL_RECORD_BYTES:
        raise ExperimentRunValidationError("JSONL record exceeds the one MiB limit")
    if path.is_symlink():
        raise ExperimentRunValidationError(f"{path.name} must not be a symlink")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        written = os.write(descriptor, data)
        if written != len(data):  # pragma: no cover - rare filesystem failure
            raise ExperimentRunError(f"short append to {path.name}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_json_lines(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    """Atomically replace one JSONL file with a complete validated record set."""

    encoded_records = []
    for value in values:
        data = _canonical_json_bytes(value) + b"\n"
        if len(data) > _MAX_JSONL_RECORD_BYTES:
            raise ExperimentRunValidationError("JSONL record exceeds the one MiB limit")
        encoded_records.append(data)
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunValidationError(
            f"{path.name} must be an existing regular file"
        )
    _atomic_replace_bytes(path, b"".join(encoded_records))


def _create_empty_regular_file(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, read_only: bool) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value) + b"\n", read_only=read_only)


def _atomic_write_bytes(path: Path, data: bytes, *, read_only: bool) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable evidence file: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:  # pragma: no cover - filesystem failure
                    raise ExperimentRunError(f"short write to {temporary.name}")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if read_only:
            temporary.chmod(0o444)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():  # pragma: no cover - failed replace cleanup
            temporary.unlink()


def _atomic_replace_bytes(path: Path, data: bytes) -> None:
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunValidationError(
            f"{path.name} must be an existing regular file"
        )
    temporary = path.with_name(f".{path.name}.replace-{uuid.uuid4().hex}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:  # pragma: no cover - filesystem failure
                    raise ExperimentRunError(f"short write to {temporary.name}")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentRunValidationError("value must contain finite JSON data") from exc


def _file_sha256(path: Path) -> str:
    return _hash_regular_file(path)[0]


def _safe_internal_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExperimentRunValidationError(f"{label} snapshot_path is invalid")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ExperimentRunValidationError(f"{label} snapshot_path contains traversal")
    if relative.parts[0] != "config_snapshots":
        raise ExperimentRunValidationError(f"{label} snapshot_path leaves config_snapshots")
    return relative


def _safe_artifact_snapshot_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExperimentRunValidationError(f"{label} snapshot_path is invalid")
    relative = Path(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ExperimentRunValidationError(f"{label} snapshot_path contains traversal")
    if len(relative.parts) != 2 or relative.parts[0] != "artifact_snapshots":
        raise ExperimentRunValidationError(
            f"{label} snapshot_path leaves artifact_snapshots"
        )
    return relative


def _validate_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ExperimentRunValidationError(f"{label} must be a UTC ISO-8601 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ExperimentRunValidationError(f"{label} is invalid") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise ExperimentRunValidationError(f"{label} must be UTC")


def _wall_time_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _validate_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExperimentRunValidationError(f"{label} must be a non-negative integer")
    return value


def _validate_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ExperimentRunValidationError(f"{label} must be a lowercase SHA-256")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value

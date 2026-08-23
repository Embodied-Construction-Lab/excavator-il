"""Reproducible, append-only evidence for one excavation experiment run.

This public module owns the evidence lifecycle.  Active runs live below
``active/``.  Finalization writes the complete manifest and compact index,
then atomically renames that directory below ``runs/``.  That rename is the
publication boundary, so readers never observe a partly published final run.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._experiment_artifact_store import (
    discard_artifact_snapshot,
    fingerprint_path,
    recover_artifact_store,
    reopen_artifact_store,
    seal_artifact_store,
    snapshot_artifact,
    verify_registered_artifacts,
)
from ._experiment_run_support import (
    _GIT_COMMIT_RE,
    _append_json_line,
    _atomic_write_json,
    _create_empty_regular_file,
    _deep_freeze,
    _exclusive_lock,
    _existing_directory,
    _file_sha256,
    _fsync_directory,
    _new_run_id,
    _normalize_evidence_requirements,
    _normalize_json_mapping,
    _normalize_path_mapping,
    _normalize_string_mapping,
    _normalize_task_context,
    _open_root,
    _prepare_root,
    _read_json_object,
    _read_jsonl,
    _replace_json_lines,
    _require_evidence,
    _snapshot_config_file,
    _validate_artifact_record,
    _validate_event_record,
    _validate_final,
    _validate_index,
    _validate_manifest,
    _validate_name,
    _validate_run_id,
    _validate_run_kind,
    _validate_start_manifest,
    _validate_text,
    _wall_time_utc,
)
from ._experiment_run_types import (
    EXPERIMENT_ARTIFACT_SCHEMA_VERSION,
    EXPERIMENT_EVENT_SCHEMA_VERSION,
    EXPERIMENT_INDEX_SCHEMA_VERSION,
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    EXPERIMENT_RUN_KINDS,
    EXPERIMENT_RUN_SCHEMA_VERSION,
    EvidenceRequirement,
    ExperimentRunError,
    ExperimentRunFinalizedError,
    ExperimentRunSnapshot,
    ExperimentRunValidationError,
    PathFingerprint,
    RepositoryState,
    TaskContext,
)

__all__ = [
    "EXPERIMENT_RUN_SCHEMA_VERSION",
    "EXPERIMENT_RUN_KINDS",
    "EvidenceRequirement",
    "ExperimentRun",
    "ExperimentRunError",
    "ExperimentRunFinalizedError",
    "ExperimentRunSnapshot",
    "ExperimentRunValidationError",
    "PathFingerprint",
    "RepositoryState",
    "TaskContext",
    "capture_repository_state",
    "fingerprint_path",
    "load_experiment_run",
]


def capture_repository_state(repository_path: str | Path) -> RepositoryState:
    """Capture the exact Git commit and dirty bit for one repository."""

    path = _existing_directory(repository_path, label="repository")
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        porcelain = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExperimentRunValidationError(
            f"repository is not a readable Git worktree: {path}"
        ) from exc
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise ExperimentRunValidationError("repository commit is invalid")
    return RepositoryState(
        source_path=str(path), commit=commit, dirty=bool(porcelain.strip())
    )


class ExperimentRun:
    """Mutable handle for one active run; final snapshots are read-only."""

    def __init__(self, root: Path, run_id: str) -> None:
        self._root = root
        self._run_id = run_id

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        run_kind: str,
        task_context: TaskContext | Mapping[str, Any],
        policy_ids: Mapping[str, str],
        host_topology: Mapping[str, Any],
        repository_paths: Mapping[str, str | Path],
        config_paths: Mapping[str, str | Path],
        machine_profile_path: str | Path,
        evidence_requirements: Mapping[
            str, EvidenceRequirement | Mapping[str, Any]
        ]
        | None = None,
        run_id: str | None = None,
    ) -> "ExperimentRun":
        evidence_root = _prepare_root(root)
        normalized_run_kind = _validate_run_kind(run_kind)
        normalized_context = _normalize_task_context(task_context)
        normalized_policies = _normalize_string_mapping(policy_ids, "policy_ids")
        normalized_hosts = _normalize_json_mapping(host_topology, "host_topology")
        normalized_requirements = _normalize_evidence_requirements(
            evidence_requirements or {}
        )
        repositories = _normalize_path_mapping(repository_paths, "repository_paths")
        configs = _normalize_path_mapping(config_paths, "config_paths")
        profile_path = Path(machine_profile_path).expanduser()

        with _exclusive_lock(evidence_root / ".locks" / "create.lock"):
            selected_run_id = _validate_run_id(run_id) if run_id else _new_run_id()
            active_dir = evidence_root / "active" / selected_run_id
            final_dir = evidence_root / "runs" / selected_run_id
            if active_dir.exists() or final_dir.exists():
                raise FileExistsError(f"experiment run already exists: {selected_run_id}")

            staging = evidence_root / "active" / f".creating-{uuid.uuid4().hex}"
            staging.mkdir(mode=0o700)
            try:
                snapshot_dir = staging / "config_snapshots"
                snapshot_dir.mkdir(mode=0o700)
                (staging / "artifact_snapshots").mkdir(mode=0o700)
                config_records = {
                    label: _snapshot_config_file(snapshot_dir, label, source)
                    for label, source in configs.items()
                }
                profile_record = _snapshot_config_file(
                    snapshot_dir, "machine_profile", profile_path
                )
                repository_records = {
                    label: dataclasses.asdict(capture_repository_state(path))
                    for label, path in repositories.items()
                }
                start = {
                    "schema_version": EXPERIMENT_RUN_SCHEMA_VERSION,
                    "run_id": selected_run_id,
                    "run_kind": normalized_run_kind,
                    "task_context": normalized_context,
                    "policy_ids": normalized_policies,
                    "host_topology": normalized_hosts,
                    "repositories": repository_records,
                    "config_snapshots": config_records,
                    "machine_profile": profile_record,
                    "evidence_requirements": normalized_requirements,
                    "started_at_utc": _wall_time_utc(),
                    "started_monotonic_ns": time.monotonic_ns(),
                }
                _validate_start_manifest(start, staging)
                _atomic_write_json(staging / "start.json", start, read_only=True)
                _create_empty_regular_file(staging / "events.jsonl")
                _create_empty_regular_file(staging / "artifacts.jsonl")
                _fsync_directory(staging)
                os.replace(staging, active_dir)
                _fsync_directory(active_dir.parent)
            except BaseException:
                if staging.exists() and not staging.is_symlink():
                    shutil.rmtree(staging)
                raise
        return cls(evidence_root, selected_run_id)

    @classmethod
    def load(cls, root: str | Path, run_id: str) -> "ExperimentRun":
        evidence_root = _open_root(root)
        normalized_id = _validate_run_id(run_id)
        with _exclusive_lock(evidence_root / ".locks" / f"{normalized_id}.lock"):
            active_dir = evidence_root / "active" / normalized_id
            if active_dir.is_dir() and not active_dir.is_symlink():
                _recover_active_finalization(active_dir)
            load_experiment_run(evidence_root, normalized_id)
        return cls(evidence_root, normalized_id)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        final_dir = self._root / "runs" / self._run_id
        if final_dir.is_dir() and not final_dir.is_symlink():
            return final_dir
        active_dir = self._root / "active" / self._run_id
        if active_dir.is_dir() and not active_dir.is_symlink():
            return active_dir
        raise FileNotFoundError(f"experiment run does not exist: {self._run_id}")

    @property
    def state(self) -> str:
        if (self._root / "runs" / self._run_id).is_dir():
            return load_experiment_run(self._root, self._run_id).state
        if (self._root / "active" / self._run_id).is_dir():
            return "active"
        raise FileNotFoundError(f"experiment run does not exist: {self._run_id}")

    def snapshot(self) -> ExperimentRunSnapshot:
        return load_experiment_run(self._root, self._run_id)

    def append_event(
        self, event_type: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        normalized_type = _validate_name(event_type, "event_type")
        normalized_payload = _normalize_json_mapping(payload or {}, "event payload")
        with _exclusive_lock(self._lock_path):
            run_dir = self._require_active_dir()
            events = _read_jsonl(run_dir / "events.jsonl", _validate_event_record)
            monotonic_ns = time.monotonic_ns()
            if events:
                monotonic_ns = max(monotonic_ns, int(events[-1]["monotonic_ns"]))
            record = {
                "schema_version": EXPERIMENT_EVENT_SCHEMA_VERSION,
                "sequence": len(events),
                "event_type": normalized_type,
                "wall_time_utc": _wall_time_utc(),
                "monotonic_ns": monotonic_ns,
                "payload": normalized_payload,
            }
            _validate_event_record(record, expected_sequence=len(events))
            _append_json_line(run_dir / "events.jsonl", record)
            return record

    def register_artifact(
        self,
        artifact_id: str,
        source_path: str | Path,
        *,
        role: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_id = _validate_name(artifact_id, "artifact_id")
        normalized_role = _validate_name(role, "artifact role")
        normalized_metadata = _normalize_json_mapping(metadata or {}, "artifact metadata")
        with _exclusive_lock(self._lock_path):
            run_dir = self._require_active_dir()
            artifacts = _read_jsonl(
                run_dir / "artifacts.jsonl", _validate_artifact_record
            )
            recover_artifact_store(run_dir, artifacts)
            if any(item["artifact_id"] == normalized_id for item in artifacts):
                raise ExperimentRunValidationError(
                    f"artifact_id is already registered: {normalized_id}"
                )
            monotonic_ns = time.monotonic_ns()
            if artifacts:
                monotonic_ns = max(monotonic_ns, int(artifacts[-1]["monotonic_ns"]))
            snapshot = snapshot_artifact(
                run_dir,
                source_path,
                artifact_id=normalized_id,
                sequence=len(artifacts),
            )
            fingerprint = snapshot.fingerprint
            record = {
                "schema_version": EXPERIMENT_ARTIFACT_SCHEMA_VERSION,
                "sequence": len(artifacts),
                "artifact_id": normalized_id,
                "role": normalized_role,
                "registered_at_utc": _wall_time_utc(),
                "monotonic_ns": monotonic_ns,
                "source_path": snapshot.source_path,
                "snapshot_path": snapshot.snapshot_path,
                "snapshot_method": snapshot.snapshot_method,
                "object_type": fingerprint.object_type,
                "sha256": fingerprint.sha256,
                "size_bytes": fingerprint.size_bytes,
                "file_count": fingerprint.file_count,
                "metadata": normalized_metadata,
            }
            try:
                _validate_artifact_record(record, expected_sequence=len(artifacts))
                _replace_json_lines(
                    run_dir / "artifacts.jsonl",
                    (*artifacts, record),
                )
            except BaseException:
                committed = _read_jsonl(
                    run_dir / "artifacts.jsonl", _validate_artifact_record
                )
                if len(committed) == len(artifacts) + 1 and committed[-1] == record:
                    verify_registered_artifacts((record,), run_dir)
                    return record
                discard_artifact_snapshot(run_dir, snapshot.snapshot_path)
                raise
            return record

    def finalize(
        self,
        status: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        summary: str | None = None,
    ) -> ExperimentRunSnapshot:
        if status not in {"success", "failure"}:
            raise ExperimentRunValidationError("final status must be success or failure")
        normalized_metrics = _normalize_json_mapping(metrics or {}, "final metrics")
        if summary is not None:
            _validate_text(summary, "final summary", allow_empty=False, max_length=4096)

        with _exclusive_lock(self._lock_path):
            run_dir = self._require_active_dir()
            _recover_active_finalization(run_dir)
            start = _read_json_object(run_dir / "start.json", "start manifest")
            _validate_start_manifest(start, run_dir)
            events = _read_jsonl(run_dir / "events.jsonl", _validate_event_record)
            artifacts = _read_jsonl(
                run_dir / "artifacts.jsonl", _validate_artifact_record
            )
            recover_artifact_store(run_dir, artifacts)
            verify_registered_artifacts(artifacts, run_dir)
            if status == "success":
                _require_evidence(start["evidence_requirements"], artifacts)
            final_dir = self._root / "runs" / self._run_id
            if final_dir.exists():
                raise ExperimentRunFinalizedError(
                    f"experiment run is already finalized: {self._run_id}"
                )

            final = {
                "status": status,
                "finished_at_utc": _wall_time_utc(),
                "finished_monotonic_ns": time.monotonic_ns(),
                "metrics": normalized_metrics,
                "summary": summary,
            }
            _validate_final(final)
            evidence = {
                "start_sha256": _file_sha256(run_dir / "start.json"),
                "events_jsonl_sha256": _file_sha256(run_dir / "events.jsonl"),
                "artifacts_jsonl_sha256": _file_sha256(run_dir / "artifacts.jsonl"),
                "event_count": len(events),
                "artifact_count": len(artifacts),
            }
            manifest = {
                "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
                "start": start,
                "final": final,
                "evidence": evidence,
            }
            _validate_manifest(manifest, run_dir)
            metadata_staging = run_dir / f".finalizing-{uuid.uuid4().hex}"
            metadata_staging.mkdir(mode=0o700)
            staged_manifest_path = metadata_staging / "manifest.json"
            staged_index_path = metadata_staging / "index.json"
            index = {
                "schema_version": EXPERIMENT_INDEX_SCHEMA_VERSION,
                "run_id": self._run_id,
                "run_kind": start["run_kind"],
                "status": status,
                "task_context": start["task_context"],
                "policy_ids": start["policy_ids"],
                "started_at_utc": start["started_at_utc"],
                "finished_at_utc": final["finished_at_utc"],
                "manifest_sha256": "",
            }
            try:
                _atomic_write_json(
                    staged_manifest_path,
                    manifest,
                    read_only=True,
                )
                index["manifest_sha256"] = _file_sha256(staged_manifest_path)
                _validate_index(index, staged_manifest_path)
                _atomic_write_json(
                    staged_index_path,
                    index,
                    read_only=True,
                )
                _validate_manifest(
                    _read_json_object(staged_manifest_path, "staged final manifest"),
                    run_dir,
                )
                _validate_index(
                    _read_json_object(staged_index_path, "staged run index"),
                    staged_manifest_path,
                )
                published_paths: list[Path] = []
                try:
                    for staged_path in (staged_manifest_path, staged_index_path):
                        published_path = run_dir / staged_path.name
                        os.replace(staged_path, published_path)
                        published_paths.append(published_path)
                    _fsync_directory(run_dir)
                except BaseException:
                    for published_path in published_paths:
                        published_path.unlink(missing_ok=True)
                    _fsync_directory(run_dir)
                    raise
            finally:
                if metadata_staging.exists() and not metadata_staging.is_symlink():
                    shutil.rmtree(metadata_staging)
            try:
                for evidence_file in (
                    run_dir / "events.jsonl",
                    run_dir / "artifacts.jsonl",
                ):
                    evidence_file.chmod(0o444)
                seal_artifact_store(run_dir)
                _fsync_directory(run_dir)
                os.replace(run_dir, final_dir)
            except BaseException:
                if final_dir.is_dir() and not final_dir.is_symlink():
                    return load_experiment_run(self._root, self._run_id)
                if run_dir.is_dir() and not run_dir.is_symlink():
                    _recover_active_finalization(run_dir)
                raise
            try:
                _fsync_directory(run_dir.parent)
                _fsync_directory(final_dir.parent)
            except BaseException:
                return load_experiment_run(self._root, self._run_id)
        return load_experiment_run(self._root, self._run_id)

    @property
    def _lock_path(self) -> Path:
        return self._root / ".locks" / f"{self._run_id}.lock"

    def _require_active_dir(self) -> Path:
        if (self._root / "runs" / self._run_id).exists():
            raise ExperimentRunFinalizedError(
                f"experiment run is finalized: {self._run_id}"
            )
        active_dir = self._root / "active" / self._run_id
        if active_dir.is_symlink():
            raise ExperimentRunValidationError("active run directory must not be a symlink")
        if not active_dir.is_dir():
            raise FileNotFoundError(f"active experiment run does not exist: {self._run_id}")
        return active_dir


def _recover_active_finalization(run_dir: Path) -> None:
    """Abort any unpublished finalize transaction left below ``active/``."""

    for metadata_path in (run_dir / "manifest.json", run_dir / "index.json"):
        if metadata_path.exists() and not metadata_path.is_file():
            raise ExperimentRunValidationError(
                f"stale {metadata_path.name} is not a regular file"
            )
        metadata_path.unlink(missing_ok=True)
    for staging in tuple(run_dir.glob(".finalizing-*")):
        if staging.is_symlink() or not staging.is_dir():
            raise ExperimentRunValidationError(
                "stale finalization staging path is not a safe directory"
            )
        shutil.rmtree(staging)
    for evidence_file in (run_dir / "events.jsonl", run_dir / "artifacts.jsonl"):
        if evidence_file.is_symlink() or not evidence_file.is_file():
            raise ExperimentRunValidationError(
                f"{evidence_file.name} is unavailable or not a regular file"
            )
        evidence_file.chmod(0o600)
    reopen_artifact_store(run_dir)
    _fsync_directory(run_dir)


def load_experiment_run(
    root_or_run_dir: str | Path, run_id: str | None = None
) -> ExperimentRunSnapshot:
    """Strictly load an active or finalized run by root/id or concrete path."""

    if run_id is None:
        candidate = Path(root_or_run_dir).expanduser()
        if candidate.is_symlink() or not candidate.is_dir():
            raise ExperimentRunValidationError("run directory is unavailable or a symlink")
        candidate = candidate.resolve(strict=True)
        if candidate.parent.name not in {"active", "runs"}:
            raise ExperimentRunValidationError(
                "run directory must be directly below active/ or runs/"
            )
        normalized_id = _validate_run_id(candidate.name)
        run_dir = candidate
        finalized = candidate.parent.name == "runs"
    else:
        root = _open_root(root_or_run_dir)
        normalized_id = _validate_run_id(run_id)
        active = root / "active" / normalized_id
        final = root / "runs" / normalized_id
        if active.exists() and final.exists():
            raise ExperimentRunValidationError(
                "run exists in both active and finalized indexes"
            )
        if final.is_dir() and not final.is_symlink():
            run_dir, finalized = final, True
        elif active.is_dir() and not active.is_symlink():
            run_dir, finalized = active, False
        else:
            raise FileNotFoundError(f"experiment run does not exist: {normalized_id}")

    start = _read_json_object(run_dir / "start.json", "start manifest")
    _validate_start_manifest(start, run_dir)
    if start["run_id"] != normalized_id:
        raise ExperimentRunValidationError("run directory and start run_id disagree")
    events = _read_jsonl(run_dir / "events.jsonl", _validate_event_record)
    artifacts = _read_jsonl(run_dir / "artifacts.jsonl", _validate_artifact_record)

    final_record: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    state = "active"
    if finalized:
        manifest = _read_json_object(run_dir / "manifest.json", "final manifest")
        _validate_manifest(manifest, run_dir)
        if manifest["start"] != start:
            raise ExperimentRunValidationError("final manifest start snapshot mismatch")
        if manifest["evidence"]["event_count"] != len(events):
            raise ExperimentRunValidationError("events.jsonl count does not match manifest")
        if manifest["evidence"]["artifact_count"] != len(artifacts):
            raise ExperimentRunValidationError("artifacts.jsonl count does not match manifest")
        final_record = manifest["final"]
        state = str(final_record["status"])
        index = _read_json_object(run_dir / "index.json", "run index")
        _validate_index(index, run_dir / "manifest.json")
        if (
            index["run_id"] != normalized_id
            or index["status"] != state
            or index["run_kind"] != start["run_kind"]
            or index["task_context"] != start["task_context"]
            or index["policy_ids"] != start["policy_ids"]
        ):
            raise ExperimentRunValidationError("run index does not match manifest")
    elif (run_dir / "manifest.json").exists() or (run_dir / "index.json").exists():
        raise ExperimentRunValidationError("active run contains unpublished final files")

    return ExperimentRunSnapshot(
        run_id=normalized_id,
        run_dir=run_dir,
        state=state,
        start=_deep_freeze(start),
        events=tuple(_deep_freeze(record) for record in events),
        artifacts=tuple(_deep_freeze(record) for record in artifacts),
        final=_deep_freeze(final_record) if final_record is not None else None,
        manifest=_deep_freeze(manifest) if manifest is not None else None,
    )

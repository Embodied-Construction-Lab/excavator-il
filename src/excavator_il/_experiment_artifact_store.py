"""Run-owned immutable snapshots for external Experiment Run artifacts."""

from __future__ import annotations

import dataclasses
import errno
import fcntl
import hashlib
import os
import shutil
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ._experiment_run_support import (
    _canonical_json_bytes,
    _fsync_directory,
    _hash_regular_file,
    _safe_artifact_snapshot_relative_path,
)
from ._experiment_run_types import (
    ExperimentRunValidationError,
    PathFingerprint,
)


_FICLONE: Final = 0x40049409
_REFLINK_FALLBACK_ERRNOS: Final = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTTY,
        errno.EOPNOTSUPP,
        errno.EPERM,
        errno.EXDEV,
    }
)


@dataclasses.dataclass(frozen=True)
class ArtifactSnapshot:
    source_path: str
    snapshot_path: str
    snapshot_method: str
    fingerprint: PathFingerprint


def fingerprint_path(source_path: str | Path) -> PathFingerprint:
    """Hash a regular file or deterministic directory tree without copying it."""

    source = Path(source_path).expanduser()
    if source.is_symlink():
        raise ExperimentRunValidationError(
            f"artifact path must not be a symlink: {source}"
        )
    if not source.exists():
        raise ExperimentRunValidationError(f"artifact path does not exist: {source}")
    source = source.resolve(strict=True)
    if source.is_file():
        digest, size = _hash_regular_file(source)
        return PathFingerprint("file", digest, size, 1)
    if not source.is_dir():
        raise ExperimentRunValidationError(
            f"artifact path must be a regular file or directory: {source}"
        )

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    file_count = 0
    children = sorted(
        source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()
    )
    for child in children:
        relative = child.relative_to(source).as_posix()
        if child.is_symlink():
            raise ExperimentRunValidationError(
                f"artifact directory contains symlink: {relative}"
            )
        try:
            child_stat = child.lstat()
        except OSError as exc:
            raise ExperimentRunValidationError(
                f"artifact directory changed while hashing: {relative}"
            ) from exc
        if stat.S_ISDIR(child_stat.st_mode):
            entries.append({"path": relative, "type": "directory"})
            continue
        if not stat.S_ISREG(child_stat.st_mode):
            raise ExperimentRunValidationError(
                f"artifact directory contains special file: {relative}"
            )
        digest, size = _hash_regular_file(child)
        entries.append(
            {"path": relative, "type": "file", "size_bytes": size, "sha256": digest}
        )
        total_bytes += size
        file_count += 1
    canonical = _canonical_json_bytes({"tree_version": 1, "entries": entries})
    return PathFingerprint(
        "directory", hashlib.sha256(canonical).hexdigest(), total_bytes, file_count
    )


def snapshot_artifact(
    run_dir: Path,
    source_path: str | Path,
    *,
    artifact_id: str,
    sequence: int,
) -> ArtifactSnapshot:
    """Create and atomically publish one content-stable snapshot inside a Run."""

    snapshot_root = run_dir / "artifact_snapshots"
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise ExperimentRunValidationError(
            "artifact snapshot directory is unavailable or a symlink"
        )
    source = Path(source_path).expanduser()
    before = fingerprint_path(source)
    resolved_source = source.resolve(strict=True)
    if resolved_source == run_dir or run_dir in resolved_source.parents:
        raise ExperimentRunValidationError(
            "artifact source must be outside its Experiment Run directory"
        )

    destination = snapshot_root / f"{sequence:06d}-{artifact_id}"
    if destination.exists() or destination.is_symlink():
        raise ExperimentRunValidationError(
            f"artifact snapshot already exists: {destination.name}"
        )
    temporary = snapshot_root / f".copying-{uuid.uuid4().hex}"
    try:
        if before.object_type == "file":
            methods = {_copy_regular_file(resolved_source, temporary)}
        else:
            methods = _copy_directory(resolved_source, temporary)
        after = fingerprint_path(resolved_source)
        copied = fingerprint_path(temporary)
        if before != after or before != copied:
            raise ExperimentRunValidationError(
                f"artifact changed while snapshotting: {resolved_source}"
            )
        _seal_snapshot(temporary)
        os.replace(temporary, destination)
        _fsync_directory(snapshot_root)
    except BaseException:
        _remove_snapshot_path(temporary)
        raise

    if not methods or methods == {"copy"}:
        snapshot_method = "copy"
    elif methods == {"reflink"}:
        snapshot_method = "reflink"
    else:
        snapshot_method = "mixed"
    return ArtifactSnapshot(
        source_path=str(resolved_source),
        snapshot_path=destination.relative_to(run_dir).as_posix(),
        snapshot_method=snapshot_method,
        fingerprint=before,
    )


def verify_registered_artifacts(
    artifacts: Sequence[Mapping[str, Any]], run_dir: Path
) -> None:
    """Verify every Run-owned snapshot against its immutable record."""

    for artifact in artifacts:
        relative = _safe_artifact_snapshot_relative_path(
            artifact["snapshot_path"], "artifact"
        )
        snapshot_path = run_dir / relative
        try:
            actual = fingerprint_path(snapshot_path)
        except Exception as exc:
            raise ExperimentRunValidationError(
                f"artifact {artifact['artifact_id']} cannot be verified: {exc}"
            ) from exc
        if (
            actual.object_type != artifact["object_type"]
            or actual.sha256 != artifact["sha256"]
            or actual.size_bytes != artifact["size_bytes"]
            or actual.file_count != artifact["file_count"]
        ):
            raise ExperimentRunValidationError(
                f"artifact {artifact['artifact_id']} fingerprint mismatch"
            )


def discard_artifact_snapshot(run_dir: Path, snapshot_path: str) -> None:
    relative = _safe_artifact_snapshot_relative_path(snapshot_path, "artifact")
    _remove_snapshot_path(run_dir / relative)
    _fsync_directory(run_dir / "artifact_snapshots")


def seal_artifact_store(run_dir: Path) -> None:
    snapshot_root = run_dir / "artifact_snapshots"
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise ExperimentRunValidationError(
            "artifact snapshot directory is unavailable or a symlink"
        )
    snapshot_root.chmod(0o555)


def reopen_artifact_store(run_dir: Path) -> None:
    snapshot_root = run_dir / "artifact_snapshots"
    if snapshot_root.is_dir() and not snapshot_root.is_symlink():
        snapshot_root.chmod(0o700)


def recover_artifact_store(
    run_dir: Path,
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    """Remove interrupted, unreferenced snapshots from one active Run."""

    snapshot_root = run_dir / "artifact_snapshots"
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise ExperimentRunValidationError(
            "artifact snapshot directory is unavailable or a symlink"
        )
    snapshot_root.chmod(0o700)
    referenced = {
        _safe_artifact_snapshot_relative_path(
            artifact["snapshot_path"], "artifact"
        ).name
        for artifact in artifacts
    }
    removed = False
    for child in tuple(snapshot_root.iterdir()):
        if child.name in referenced:
            continue
        _remove_snapshot_path(child)
        removed = True
    if removed:
        _fsync_directory(snapshot_root)


def _copy_regular_file(source: Path, destination: Path) -> str:
    source_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    source_descriptor = os.open(source, source_flags)
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExperimentRunValidationError(
                f"artifact contains non-regular file: {source}"
            )
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            method = _reflink_or_copy(
                source_descriptor,
                destination_descriptor,
                source,
            )
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
        after = os.fstat(source_descriptor)
        if (
            before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ExperimentRunValidationError(
                f"artifact file changed while copying: {source}"
            )
        return method
    except BaseException:
        if destination.exists() and not destination.is_symlink():
            destination.unlink()
        raise
    finally:
        os.close(source_descriptor)


def _reflink_or_copy(
    source_descriptor: int,
    destination_descriptor: int,
    source: Path,
) -> str:
    try:
        fcntl.ioctl(destination_descriptor, _FICLONE, source_descriptor)
        return "reflink"
    except OSError as exc:
        if exc.errno not in _REFLINK_FALLBACK_ERRNOS:
            raise ExperimentRunValidationError(
                f"cannot snapshot artifact file: {source}"
            ) from exc

    os.ftruncate(destination_descriptor, 0)
    os.lseek(source_descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(source_descriptor, 1024 * 1024)
        if not chunk:
            break
        offset = 0
        while offset < len(chunk):
            written = os.write(destination_descriptor, chunk[offset:])
            if written <= 0:  # pragma: no cover - filesystem failure
                raise ExperimentRunValidationError(
                    f"short write while snapshotting artifact: {source}"
                )
            offset += written
    return "copy"


def _copy_directory(source: Path, destination: Path) -> set[str]:
    destination.mkdir(mode=0o700)
    methods: set[str] = set()
    children = sorted(
        source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()
    )
    for child in children:
        relative = child.relative_to(source)
        if child.is_symlink():
            raise ExperimentRunValidationError(
                f"artifact directory contains symlink: {relative.as_posix()}"
            )
        try:
            child_stat = child.lstat()
        except OSError as exc:
            raise ExperimentRunValidationError(
                f"artifact directory changed while copying: {relative.as_posix()}"
            ) from exc
        copied_path = destination / relative
        if stat.S_ISDIR(child_stat.st_mode):
            copied_path.mkdir(mode=0o700, parents=True, exist_ok=True)
            continue
        if not stat.S_ISREG(child_stat.st_mode):
            raise ExperimentRunValidationError(
                f"artifact directory contains special file: {relative.as_posix()}"
            )
        copied_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        methods.add(_copy_regular_file(child, copied_path))
    return methods


def _seal_snapshot(path: Path) -> None:
    if path.is_file():
        path.chmod(0o444)
        return
    children = sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for child in children:
        if child.is_dir() and not child.is_symlink():
            child.chmod(0o555)
        elif child.is_file() and not child.is_symlink():
            child.chmod(0o444)
    path.chmod(0o555)


def _remove_snapshot_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_dir() and not child.is_symlink():
            child.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)

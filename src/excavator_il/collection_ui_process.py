"""Single-instance lifecycle for the PC-local collection WebUI."""

from __future__ import annotations

import json
import os
import signal
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_SCHEMA_VERSION = "excavator_collection_ui_process.v1"
_RECORD_FIELDS = frozenset({"schema_version", "pid", "repo_root", "config_path"})
_TERMINAL_COLLECTION_STAGES = frozenset(
    {"idle", "cancelled", "completed", "failed"}
)
_TERMINAL_HYBRID_STAGES = frozenset({"idle", "cancelled", "completed", "failed"})
_STOP_TIMEOUT_S = 8.0


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@contextmanager
def collection_ui_process_lease(
    *,
    config_path: str | Path,
    host: str,
    port: int,
    pid_path: str | Path | None = None,
) -> Iterator[None]:
    """Replace one idle owned WebUI, then publish this process as its owner."""

    config = Path(config_path).expanduser().resolve()
    path = (
        repository_root() / "logs" / "collection_ui.pid.json"
        if pid_path is None
        else Path(pid_path).expanduser().resolve()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _reclaim_existing(path, config_path=config, host=host, port=port)
    record = _record_bytes(config)
    _create_exclusive_record(path, record)
    try:
        yield
    finally:
        _unlink_if_matches(path, record)


def _reclaim_existing(
    path: Path,
    *,
    config_path: Path,
    host: str,
    port: int,
) -> None:
    if not path.exists():
        return
    record, original = _read_record(path)
    pid = record["pid"]
    if not _process_alive(pid):
        _unlink_if_matches(path, original)
        return
    if not _is_owned_ui_process(pid, record):
        raise RuntimeError(
            f"existing pid={pid} does not belong to this WebUI; "
            f"inspect {path} manually"
        )
    collection_stage, hybrid_stage = _read_ui_statuses(host, port)
    if collection_stage not in _TERMINAL_COLLECTION_STAGES:
        raise RuntimeError(
            f"existing WebUI has active collection stage={collection_stage}; "
            "use 安全停止 before restarting"
        )
    if hybrid_stage not in _TERMINAL_HYBRID_STAGES:
        raise RuntimeError(
            f"existing WebUI has active hybrid stage={hybrid_stage}; "
            "use 安全停止 before restarting"
        )
    if Path(record["config_path"]) != config_path:
        raise RuntimeError(
            "existing idle WebUI uses a different config; stop it manually"
        )
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + _STOP_TIMEOUT_S
    while _process_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_alive(pid):
        raise RuntimeError(
            f"existing WebUI pid={pid} did not stop after SIGINT; "
            "refusing to use SIGKILL"
        )
    _unlink_if_matches(path, original)


def _read_record(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        original = path.read_bytes()
        value = json.loads(original)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid collection UI process record: {path}") from exc
    if not isinstance(value, dict) or frozenset(value) != _RECORD_FIELDS:
        raise RuntimeError(f"invalid collection UI process record: {path}")
    pid = value.get("pid")
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
    ):
        raise RuntimeError(f"invalid collection UI process record: {path}")
    for name in ("repo_root", "config_path"):
        raw = value.get(name)
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise RuntimeError(f"invalid collection UI process record: {path}")
    return value, original


def _record_bytes(config_path: Path) -> bytes:
    value = {
        "schema_version": _SCHEMA_VERSION,
        "pid": os.getpid(),
        "repo_root": str(repository_root()),
        "config_path": str(config_path),
    }
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _create_exclusive_record(path: Path, record: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(record)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as exc:
        raise RuntimeError("another collection UI is starting") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _unlink_if_matches(path: Path, expected: bytes) -> None:
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        return
    if current == expected:
        path.unlink(missing_ok=True)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_owned_ui_process(pid: int, record: dict[str, object]) -> bool:
    expected_root = repository_root()
    if Path(str(record["repo_root"])) != expected_root:
        return False
    try:
        cwd = Path(f"/proc/{pid}/cwd").resolve(strict=True)
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    if cwd != expected_root:
        return False
    argv = [part.decode(errors="strict") for part in raw.split(b"\0") if part]
    expected_script = expected_root / "scripts" / "run_collection_ui.py"
    script_matches = any(
        not argument.startswith("-")
        and (cwd / argument).resolve() == expected_script
        for argument in argv[1:]
    )
    if not script_matches:
        return False
    try:
        config_index = argv.index("--config") + 1
        active_config = (cwd / argv[config_index]).expanduser().resolve()
    except (ValueError, IndexError):
        active_config = (expected_root / "config" / "collection_ui.pc.json").resolve()
    return active_config == Path(str(record["config_path"]))


def _read_ui_statuses(host: str, port: int) -> tuple[str, str]:
    query_host = "127.0.0.1" if host in {"0.0.0.0", "localhost"} else host
    try:
        collection = _read_json(f"http://{query_host}:{port}/api/status")
        hybrid = _read_json(f"http://{query_host}:{port}/api/hybrid/status")
        return _stage(collection, "collection"), _stage(hybrid, "hybrid")
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise RuntimeError(
            "existing WebUI state is unavailable; refusing automatic termination"
        ) from exc


def _read_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        return json.load(response)


def _stage(value: object, name: str) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"{name} status must be an object")
    stage = value.get("stage")
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"{name} status stage is invalid")
    return stage

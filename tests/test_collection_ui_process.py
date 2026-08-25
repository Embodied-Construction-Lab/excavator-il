import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import excavator_il.collection_ui_process as process_module
from excavator_il.collection_ui_process import collection_ui_process_lease


def _record(pid: int, repo_root, config_path):
    return {
        "schema_version": "excavator_collection_ui_process.v1",
        "pid": pid,
        "repo_root": str(repo_root),
        "config_path": str(config_path),
    }


def test_process_lease_publishes_and_removes_own_record(tmp_path):
    pid_path = tmp_path / "collection_ui.pid.json"
    config_path = tmp_path / "ui.json"
    config_path.write_text("{}", encoding="utf-8")

    with collection_ui_process_lease(
        config_path=config_path,
        host="127.0.0.1",
        port=8088,
        pid_path=pid_path,
    ):
        value = json.loads(pid_path.read_text(encoding="utf-8"))
        assert value["pid"] == os.getpid()
        assert value["config_path"] == str(config_path.resolve())

    assert not pid_path.exists()


def test_process_lease_replaces_only_an_idle_owned_ui(tmp_path, monkeypatch):
    pid_path = tmp_path / "collection_ui.pid.json"
    config_path = tmp_path / "ui.json"
    config_path.write_text("{}", encoding="utf-8")
    repo_root = process_module.repository_root()
    pid_path.write_text(
        json.dumps(_record(4242, repo_root, config_path.resolve())),
        encoding="utf-8",
    )
    alive = {4242: True}
    signals = []
    monkeypatch.setattr(
        process_module, "_process_alive", lambda pid: alive.get(pid, False)
    )
    monkeypatch.setattr(process_module, "_is_owned_ui_process", lambda *_args: True)
    monkeypatch.setattr(
        process_module,
        "_read_ui_statuses",
        lambda *_args: ("cancelled", "idle"),
    )

    def send(pid, passed_signal):
        signals.append((pid, passed_signal))
        alive[pid] = False

    monkeypatch.setattr(process_module.os, "kill", send)

    with collection_ui_process_lease(
        config_path=config_path,
        host="127.0.0.1",
        port=8088,
        pid_path=pid_path,
    ):
        assert json.loads(pid_path.read_text(encoding="utf-8"))["pid"] == os.getpid()

    assert signals == [(4242, signal.SIGINT)]


def test_process_lease_refuses_to_replace_an_active_ui(tmp_path, monkeypatch):
    pid_path = tmp_path / "collection_ui.pid.json"
    config_path = tmp_path / "ui.json"
    config_path.write_text("{}", encoding="utf-8")
    pid_path.write_text(
        json.dumps(
            _record(4242, process_module.repository_root(), config_path.resolve())
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(process_module, "_process_alive", lambda _pid: True)
    monkeypatch.setattr(process_module, "_is_owned_ui_process", lambda *_args: True)
    monkeypatch.setattr(
        process_module,
        "_read_ui_statuses",
        lambda *_args: ("teleoperation", "idle"),
    )
    monkeypatch.setattr(
        process_module.os,
        "kill",
        lambda *_args: pytest.fail("active UI must not be signalled"),
    )

    with pytest.raises(RuntimeError, match="active collection stage=teleoperation"):
        with collection_ui_process_lease(
            config_path=config_path,
            host="127.0.0.1",
            port=8088,
            pid_path=pid_path,
        ):
            pass


def test_process_lease_refuses_an_unrelated_live_pid(tmp_path, monkeypatch):
    pid_path = tmp_path / "collection_ui.pid.json"
    config_path = tmp_path / "ui.json"
    config_path.write_text("{}", encoding="utf-8")
    pid_path.write_text(
        json.dumps(
            _record(4242, process_module.repository_root(), config_path.resolve())
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(process_module, "_process_alive", lambda _pid: True)
    monkeypatch.setattr(process_module, "_is_owned_ui_process", lambda *_args: False)

    with pytest.raises(RuntimeError, match="does not belong to this WebUI"):
        with collection_ui_process_lease(
            config_path=config_path,
            host="127.0.0.1",
            port=8088,
            pid_path=pid_path,
        ):
            pass


def test_process_lease_removes_a_stale_record(tmp_path, monkeypatch):
    pid_path = tmp_path / "collection_ui.pid.json"
    config_path = tmp_path / "ui.json"
    config_path.write_text("{}", encoding="utf-8")
    pid_path.write_text(
        json.dumps(
            _record(4242, process_module.repository_root(), config_path.resolve())
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(process_module, "_process_alive", lambda _pid: False)

    with collection_ui_process_lease(
        config_path=config_path,
        host="127.0.0.1",
        port=8088,
        pid_path=pid_path,
    ):
        assert json.loads(pid_path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_process_lease_refuses_an_idle_ui_with_different_config(
    tmp_path, monkeypatch
):
    pid_path = tmp_path / "collection_ui.pid.json"
    config_path = tmp_path / "ui.json"
    other_config = tmp_path / "other.json"
    config_path.write_text("{}", encoding="utf-8")
    other_config.write_text("{}", encoding="utf-8")
    pid_path.write_text(
        json.dumps(
            _record(4242, process_module.repository_root(), other_config.resolve())
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(process_module, "_process_alive", lambda _pid: True)
    monkeypatch.setattr(process_module, "_is_owned_ui_process", lambda *_args: True)
    monkeypatch.setattr(
        process_module, "_read_ui_statuses", lambda *_args: ("idle", "idle")
    )

    with pytest.raises(RuntimeError, match="different config"):
        with collection_ui_process_lease(
            config_path=config_path,
            host="127.0.0.1",
            port=8088,
            pid_path=pid_path,
        ):
            pass


def test_process_lease_refuses_sigkill_when_idle_ui_does_not_exit(
    tmp_path, monkeypatch
):
    pid_path = tmp_path / "collection_ui.pid.json"
    config_path = tmp_path / "ui.json"
    config_path.write_text("{}", encoding="utf-8")
    pid_path.write_text(
        json.dumps(
            _record(4242, process_module.repository_root(), config_path.resolve())
        ),
        encoding="utf-8",
    )
    signals = []
    monkeypatch.setattr(process_module, "_process_alive", lambda _pid: True)
    monkeypatch.setattr(process_module, "_is_owned_ui_process", lambda *_args: True)
    monkeypatch.setattr(
        process_module, "_read_ui_statuses", lambda *_args: ("idle", "idle")
    )
    monkeypatch.setattr(
        process_module.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )
    monotonic_values = iter((0.0, 9.0))
    monkeypatch.setattr(
        process_module.time, "monotonic", lambda: next(monotonic_values)
    )

    with pytest.raises(RuntimeError, match="refusing to use SIGKILL"):
        with collection_ui_process_lease(
            config_path=config_path,
            host="127.0.0.1",
            port=8088,
            pid_path=pid_path,
        ):
            pass

    assert signals == [(4242, signal.SIGINT)]


@pytest.mark.parametrize(
    "payload",
    (
        b"not-json",
        b"[]",
        json.dumps({"schema_version": "wrong"}).encode(),
        json.dumps(
            _record(True, process_module.repository_root(), Path("/tmp/config.json"))
        ).encode(),
        json.dumps(
            _record(2, "relative", Path("/tmp/config.json"))
        ).encode(),
    ),
)
def test_read_record_rejects_malformed_or_unsafe_content(tmp_path, payload):
    path = tmp_path / "record.json"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError, match="invalid collection UI process record"):
        process_module._read_record(path)


def test_status_reader_normalizes_wildcard_host_and_requires_valid_stages(monkeypatch):
    urls = []

    def read_json(url):
        urls.append(url)
        return {"stage": "idle"}

    monkeypatch.setattr(process_module, "_read_json", read_json)
    assert process_module._read_ui_statuses("0.0.0.0", 8088) == ("idle", "idle")
    assert urls == [
        "http://127.0.0.1:8088/api/status",
        "http://127.0.0.1:8088/api/hybrid/status",
    ]

    monkeypatch.setattr(process_module, "_read_json", lambda _url: {"stage": 1})
    with pytest.raises(RuntimeError, match="state is unavailable"):
        process_module._read_ui_statuses("127.0.0.1", 8088)


def test_process_alive_distinguishes_missing_and_permission_denied(monkeypatch):
    monkeypatch.setattr(
        process_module.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert not process_module._process_alive(4242)

    monkeypatch.setattr(
        process_module.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(PermissionError()),
    )
    assert process_module._process_alive(4242)


def test_unlink_does_not_remove_a_replaced_record(tmp_path):
    path = tmp_path / "record.json"
    path.write_bytes(b"new owner")

    process_module._unlink_if_matches(path, b"old owner")

    assert path.read_bytes() == b"new owner"


def test_owned_process_identity_checks_repo_script_and_config(tmp_path):
    config_path = tmp_path / "ui.json"
    config_path.write_text("{}", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "scripts/run_collection_ui.py",
            "--config",
            str(config_path),
        ],
        cwd=process_module.repository_root(),
    )
    try:
        record = _record(
            process.pid,
            process_module.repository_root(),
            config_path.resolve(),
        )
        assert process_module._is_owned_ui_process(process.pid, record)
        assert not process_module._is_owned_ui_process(
            process.pid,
            {**record, "repo_root": str(tmp_path)},
        )
    finally:
        process.terminate()
        process.wait(timeout=5)

"""Shared PC-side process and SSH lifecycle primitives for Orin runtimes."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol


class RunCommand(Protocol):
    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]: ...


class SshRuntimeHost:
    """Hide the validated SSH transport shared by guided and hybrid workflows."""

    def __init__(
        self,
        ssh_host: str,
        *,
        run_command: RunCommand = subprocess.run,
        connect_timeout_s: int = 5,
        command_timeout_s: int = 30,
    ) -> None:
        if not isinstance(ssh_host, str) or not ssh_host:
            raise ValueError("ssh_host must be non-empty")
        if connect_timeout_s <= 0 or command_timeout_s <= 0:
            raise ValueError("SSH timeouts must be positive")
        self._ssh_host = ssh_host
        self._run_command = run_command
        self._connect_timeout_s = connect_timeout_s
        self._command_timeout_s = command_timeout_s

    def argv(self, remote_command: str) -> list[str]:
        if not isinstance(remote_command, str) or not remote_command:
            raise ValueError("remote_command must be non-empty")
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self._connect_timeout_s}",
            self._ssh_host,
            remote_command,
        ]

    def run(
        self,
        remote_command: str,
        *,
        accepted_returncodes: tuple[int, ...] = (0,),
    ) -> str:
        result = self._run_command(
            self.argv(remote_command),
            capture_output=True,
            text=True,
            timeout=self._command_timeout_s,
        )
        if result.returncode not in accepted_returncodes:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"remote command failed: {detail}")
        return result.stdout

    def reclaim_serial_owner(
        self,
        *,
        serial_path: str | PurePosixPath,
        known_argv_suffixes: tuple[tuple[str, ...], ...],
        timeout_s: int,
        execute: Callable[[str], str] | None = None,
    ) -> str:
        """Release a known stale Runtime while refusing every unknown owner.

        The comparison is against the process' NUL-delimited argv rather than a
        broad ``pkill`` pattern.  This makes explicit workflow takeover useful
        after a UI crash without turning it into a bypass around serial-owner
        mutual exclusion.
        """

        serial = PurePosixPath(serial_path)
        if not serial.is_absolute() or not str(serial).startswith("/dev/"):
            raise ValueError("serial_path must be an absolute /dev path")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a positive integer")
        if (
            not isinstance(known_argv_suffixes, tuple)
            or not known_argv_suffixes
            or any(
                not isinstance(argv, tuple)
                or not argv
                or any(
                    not isinstance(value, str)
                    or not value
                    or "\x00" in value
                    for value in argv
                )
                for argv in known_argv_suffixes
            )
        ):
            raise ValueError(
                "known_argv_suffixes must contain non-empty NUL-free argv tuples"
            )

        program = r'''import json
import os
import pathlib
import signal
import subprocess
import sys
import time

serial = sys.argv[1]
timeout_s = float(sys.argv[2])
known = tuple(tuple(argv) for argv in json.loads(sys.argv[3]))

def owners():
    result = subprocess.run(
        ["fuser", serial],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return tuple(int(value) for value in result.stdout.split())

initial = owners()
if not initial:
    print("idle")
    raise SystemExit(0)

matched = []
for pid in initial:
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        continue
    argv = tuple(os.fsdecode(value) for value in raw.split(b"\0") if value)
    if not any(
        len(argv) >= len(expected) and argv[-len(expected):] == expected
        for expected in known
    ):
        command = " ".join(argv)
        raise SystemExit(
            f"serial owner is not reclaimable: pid={pid} command={command}"
        )
    matched.append(pid)

for pid in matched:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

deadline = time.monotonic() + timeout_s
while time.monotonic() < deadline:
    if not owners():
        print("reclaimed")
        raise SystemExit(0)
    time.sleep(0.1)

details = []
for pid in owners():
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
        argv = tuple(os.fsdecode(value) for value in raw.split(b"\0") if value)
    except FileNotFoundError:
        continue
    details.append(f"pid={pid} command={' '.join(argv)}")
raise SystemExit("serial remained owned after TERM: " + "; ".join(details))
'''
        payload = json.dumps(known_argv_suffixes, ensure_ascii=True)
        command = shlex.join(
            [
                "/usr/bin/python3",
                "-c",
                program,
                str(serial),
                str(timeout_s),
                payload,
            ]
        )
        run_remote = execute or self.run
        result = run_remote(command).strip()
        if result not in {"idle", "reclaimed"}:
            raise RuntimeError(
                f"serial-owner reconciliation returned an invalid result: {result!r}"
            )
        return result

    def stop_owned_process(
        self,
        *,
        pid: int,
        identity_ere: str,
        serial_path: str | PurePosixPath,
        timeout_s: int,
        require_serial_release: bool = True,
        cleanup_paths: tuple[str | PurePosixPath, ...] = (),
        execute: Callable[[str], str] | None = None,
    ) -> None:
        """Stop one verified Runtime and prove its physical interface is released."""

        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        if not isinstance(identity_ere, str) or not identity_ere:
            raise ValueError("identity_ere must be non-empty")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a positive integer")
        serial = PurePosixPath(serial_path)
        if not serial.is_absolute() or not str(serial).startswith("/dev/"):
            raise ValueError("serial_path must be an absolute /dev path")
        cleanup = tuple(PurePosixPath(path) for path in cleanup_paths)
        if any(not path.is_absolute() or path == PurePosixPath("/") for path in cleanup):
            raise ValueError("cleanup_paths must contain safe absolute paths")

        attempts = max(4, timeout_s * 4)
        serial_check = ""
        if require_serial_release:
            serial_check = f"""command -v fuser >/dev/null
if fuser -s {shlex.quote(str(serial))}; then
  echo "serial is still owned: {serial}" >&2
  exit 14
fi
"""
        cleanup_command = ""
        if cleanup:
            cleanup_command = "rm -f -- " + " ".join(
                shlex.quote(str(path)) for path in cleanup
            )
        script = f"""set -eu
pid={pid}
if kill -0 "$pid" 2>/dev/null; then
  tr '\\000' ' ' < "/proc/$pid/cmdline" | grep -Eq {shlex.quote(identity_ere)}
  kill -TERM "$pid"
fi
attempt=0
while kill -0 "$pid" 2>/dev/null; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge {attempts} ]; then
    echo "Runtime did not exit after SIGTERM" >&2
    exit 13
  fi
  sleep 0.25
done
{serial_check}{cleanup_command}
echo released
"""
        run_remote = execute or self.run
        output = run_remote(f"/bin/sh -c {shlex.quote(script)}")
        if output.strip() != "released":
            raise RuntimeError("Runtime stop did not confirm release")


class LineWaitTimeout(RuntimeError):
    """A bounded line wait expired while the child process remained active."""


class LineProcess:
    """Own one line-oriented child process and its immutable output history."""

    def __init__(
        self,
        argv: list[str],
        *,
        log_path: Path,
        prefix: str,
        output: Callable[[str], None],
        echo_output: bool = True,
        popen_command: Callable[..., object] = subprocess.Popen,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._condition = threading.Condition()
        self._lines: tuple[str, ...] = ()
        self._done = False
        self._prefix = prefix
        self._output = output
        self._echo_output = echo_output
        self._log_path = log_path
        self._process = popen_command(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        pid = getattr(self._process, "pid", None)
        self._process_group_id = (
            pid
            if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
            else None
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        assert self._process.stdout is not None
        with self._log_path.open("a", encoding="utf-8") as log:
            for raw_line in self._process.stdout:
                line = raw_line.rstrip("\r\n")
                log.write(line + "\n")
                log.flush()
                if self._echo_output:
                    self._output(f"[{self._prefix}] {line}")
                with self._condition:
                    self._lines = (*self._lines, line)
                    self._condition.notify_all()
        with self._condition:
            self._done = True
            self._condition.notify_all()

    def wait_for(
        self,
        predicate: Callable[[str], bool],
        timeout_s: float | None,
        *,
        after_index: int = -1,
    ) -> tuple[int, str]:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while True:
                for index, line in enumerate(self._lines):
                    if index > after_index and predicate(line):
                        return index, line
                if self._done:
                    raise RuntimeError(
                        f"{self._prefix} exited before the expected readiness signal"
                    )
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise LineWaitTimeout(
                        f"timed out waiting for {self._prefix} readiness"
                    )
                self._condition.wait(remaining_s)

    def _signal_owned_process_group(
        self,
        signum: int,
        *,
        fallback: Callable[[], None],
    ) -> None:
        process_group_id = self._process_group_id
        if process_group_id is None:
            fallback()
            return
        try:
            os.killpg(process_group_id, signum)
        except ProcessLookupError:
            if self._process.poll() is None:
                fallback()

    def stop(self, signum: int, *, timeout_s: float = 5.0) -> None:
        if self._process_group_id is not None or self._process.poll() is None:
            self._signal_owned_process_group(
                signum,
                fallback=lambda: self._process.send_signal(signum),
            )
        try:
            self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._signal_owned_process_group(
                signal.SIGTERM,
                fallback=self._process.terminate,
            )
            try:
                self._process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._signal_owned_process_group(
                    signal.SIGKILL,
                    fallback=self._process.kill,
                )
                self._process.wait(timeout=timeout_s)
        self._reader.join(timeout=1.0)

    def wait(self, timeout_s: float = 5.0) -> None:
        self._process.wait(timeout=timeout_s)
        self._reader.join(timeout=1.0)

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    @property
    def running(self) -> bool:
        return self._process.poll() is None

    @property
    def lines(self) -> tuple[str, ...]:
        with self._condition:
            return self._lines

"""PC-side deadman-guided hardware Episode collection."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol


GUIDED_EPISODE_CONFIG_SCHEMA_VERSION = "excavator_guided_episode_config.v2"
_SSH_HOST = re.compile(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+")
_EPISODE_NAME = re.compile(r"episode_\d{4,}")
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config/guided_episode.pc.json"


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class GuidedEpisodeConfig:
    teleop_config: Path
    orin_ssh_host: str
    orin_repo: PurePosixPath
    orin_executable: PurePosixPath
    orin_collection_config: PurePosixPath
    task: str
    operator_id: str
    dig_target_m: tuple[float, float, float]
    material_id: str
    collector_ready_timeout_s: int
    ack_timeout_s: int
    teleop_print_every: int
    log_dir: Path
    failure_reason: str = "diagnostic_task_failed"
    zero_soak_duration_s: int = 30

    @classmethod
    def load(cls, path: str | Path) -> "GuidedEpisodeConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            root = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load guided Episode config {config_path}: {exc}") from exc
        root = _object(root, "config")
        if root.get("schema_version") != GUIDED_EPISODE_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {GUIDED_EPISODE_CONFIG_SCHEMA_VERSION}"
            )
        orin = _object(root.get("orin"), "orin")
        episode = _object(root.get("episode"), "episode")
        runtime = _object(root.get("runtime"), "runtime")
        ssh_host = _text(orin.get("ssh_host"), "orin.ssh_host")
        if _SSH_HOST.fullmatch(ssh_host) is None:
            raise ValueError("orin.ssh_host must be user@host without shell syntax")
        target = episode.get("dig_target_m")
        if not isinstance(target, list) or len(target) != 3:
            raise ValueError("episode.dig_target_m must contain three numbers")
        target_values = tuple(float(value) for value in target)
        if any(not math.isfinite(value) for value in target_values):
            raise ValueError("episode.dig_target_m must be finite")
        teleop_print_every = _positive_int(
            runtime.get("teleop_print_every"), "runtime.teleop_print_every"
        )
        if teleop_print_every != 1:
            raise ValueError(
                "runtime.teleop_print_every must be 1 for 20 Hz deadman edge detection"
            )
        base = config_path.parent
        return cls(
            teleop_config=(base / _text(root.get("teleop_config"), "teleop_config")).resolve(),
            orin_ssh_host=ssh_host,
            orin_repo=PurePosixPath(_text(orin.get("repo"), "orin.repo")),
            orin_executable=PurePosixPath(
                _text(orin.get("executable"), "orin.executable")
            ),
            orin_collection_config=PurePosixPath(
                _text(orin.get("collection_config"), "orin.collection_config")
            ),
            task=_text(episode.get("task"), "episode.task"),
            operator_id=_text(episode.get("operator_id"), "episode.operator_id"),
            dig_target_m=target_values,
            material_id=_text(episode.get("material_id"), "episode.material_id"),
            collector_ready_timeout_s=_positive_int(
                runtime.get("collector_ready_timeout_s"),
                "runtime.collector_ready_timeout_s",
            ),
            ack_timeout_s=_positive_int(
                runtime.get("ack_timeout_s"), "runtime.ack_timeout_s"
            ),
            teleop_print_every=teleop_print_every,
            log_dir=(base / _text(runtime.get("log_dir"), "runtime.log_dir")).resolve(),
            failure_reason=_text(
                episode.get("failure_reason", "diagnostic_task_failed"),
                "episode.failure_reason",
            ),
            zero_soak_duration_s=_positive_int(
                runtime.get("zero_soak_duration_s", 30),
                "runtime.zero_soak_duration_s",
            ),
        )


class GuidedEpisodeOperations(Protocol):
    def preflight(self) -> None: ...

    def start_collector(self) -> None: ...

    def start_teleop(self) -> None: ...

    def wait_for_ack(self, timeout_s: int) -> None: ...

    def wait_for_deadman_pressed(self) -> None: ...

    def wait_for_deadman_released(self) -> None: ...

    def start_episode(self) -> str: ...

    def seal_episode(self) -> str: ...

    def finalize_episode(
        self, episode_path: str, result: str, reason: str = ""
    ) -> str: ...

    def abort_episode(self, reason: str) -> str: ...

    def discard_episode(self, episode_path: str) -> None: ...

    def stop_teleop(self) -> None: ...

    def stop_collector(self) -> None: ...

    def build_and_validate(self, episode_path: str) -> None: ...


class _LineWaitTimeout(RuntimeError):
    """A bounded line wait expired while the child process remained active."""


class _LineProcess:
    def __init__(
        self,
        argv: list[str],
        *,
        log_path: Path,
        prefix: str,
        output: Callable[[str], None],
        echo_output: bool = True,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._condition = threading.Condition()
        self._lines: tuple[str, ...] = ()
        self._done = False
        self._prefix = prefix
        self._output = output
        self._echo_output = echo_output
        self._log_path = log_path
        self._process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
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
                    if index <= after_index:
                        continue
                    if predicate(line):
                        return index, line
                if self._done:
                    raise RuntimeError(
                        f"{self._prefix} exited before the expected readiness signal"
                    )
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        raise _LineWaitTimeout(
                            f"timed out waiting for {self._prefix} readiness"
                        )
                    self._condition.wait(remaining_s)

    def stop(self, signum: int, *, timeout_s: float = 5.0) -> None:
        if self._process.poll() is None:
            self._process.send_signal(signum)
        try:
            self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=timeout_s)
        self._reader.join(timeout=1.0)

    def wait(self, timeout_s: float = 5.0) -> None:
        self._process.wait(timeout=timeout_s)
        self._reader.join(timeout=1.0)

    @property
    def running(self) -> bool:
        return self._process.poll() is None


class SystemGuidedEpisodeOperations:
    """Real PC/SSH boundary used by the guided Episode script."""

    _ACK = re.compile(
        r"accepted_acks=(?P<accepted>\d+) rejected_acks=(?P<rejected>\d+).*"
        r"deadman=(?P<deadman>True|False)"
    )

    def __init__(
        self,
        config: GuidedEpisodeConfig,
        *,
        output: Callable[[str], None] = print,
        timestamp: str | None = None,
    ) -> None:
        self._config = config
        self._output = output
        self._timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._collector: _LineProcess | None = None
        self._teleop: _LineProcess | None = None
        self._collector_pid: int | None = None
        self._teleop_cursor = -1
        self._started_episode_paths: tuple[str, ...] = ()
        self._discardable_episode_paths: tuple[str, ...] = ()

    @property
    def log_paths(self) -> tuple[Path, Path, Path]:
        stem = f"guided_episode_{self._timestamp}"
        return (
            self._config.log_dir / f"{stem}.collector.log",
            self._config.log_dir / f"{stem}.teleop.log",
            self._config.log_dir / f"{stem}.validation.log",
        )

    def _ssh_argv(self, remote_command: str) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            self._config.orin_ssh_host,
            remote_command,
        ]

    def _in_repo(self, argv: list[str]) -> str:
        return (
            f"cd {shlex.quote(str(self._config.orin_repo))} && "
            f"{shlex.join(argv)}"
        )

    def _run_ssh(
        self,
        remote_command: str,
        *,
        accepted_returncodes: tuple[int, ...] = (0,),
    ) -> str:
        result = subprocess.run(
            self._ssh_argv(remote_command),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode not in accepted_returncodes:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"remote command failed: {detail}")
        return result.stdout

    def _remote_cli(
        self,
        argv: list[str],
        *,
        accepted_returncodes: tuple[int, ...] = (0,),
    ) -> Mapping[str, Any]:
        executable = str(self._config.orin_executable)
        output = self._run_ssh(
            self._in_repo([executable, *argv]),
            accepted_returncodes=accepted_returncodes,
        )
        try:
            response = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid remote CLI JSON: {output!r}") from exc
        if not isinstance(response, Mapping):
            raise RuntimeError("remote CLI response must be an object")
        return response

    @staticmethod
    def _episode_path(response: Mapping[str, Any]) -> str:
        path = response.get("path")
        if not isinstance(path, str) or not PurePosixPath(path).is_absolute():
            raise RuntimeError("remote Episode response did not contain an absolute path")
        return path

    def preflight(self) -> None:
        if not self._config.teleop_config.is_file():
            raise RuntimeError(
                f"teleop config does not exist: {self._config.teleop_config}"
            )
        remote_check = self._in_repo(
            [
                "test",
                "-x",
                str(self._config.orin_executable),
            ]
        )
        self._run_ssh(remote_check)

    def start_collector(self) -> None:
        collector_log, _, _ = self.log_paths
        executable = str(self._config.orin_executable)
        command = self._in_repo(
            [
                executable,
                "collect",
                "--config",
                str(self._config.orin_collection_config),
            ]
        )
        command = command.replace("&& ", "&& echo GUIDED_COLLECTOR_PID=$$ && exec ", 1)
        self._collector = _LineProcess(
            self._ssh_argv(command),
            log_path=collector_log,
            prefix="collector",
            output=self._output,
        )
        try:
            _, pid_line = self._collector.wait_for(
                lambda line: line.startswith("GUIDED_COLLECTOR_PID="),
                self._config.collector_ready_timeout_s,
            )
            self._collector_pid = int(pid_line.split("=", maxsplit=1)[1])
            self._collector.wait_for(
                lambda line: "collector ready:" in line,
                self._config.collector_ready_timeout_s,
            )
        except BaseException:
            self.stop_collector()
            raise

    def start_teleop(self) -> None:
        _, teleop_log, _ = self.log_paths
        self._teleop_cursor = -1
        self._teleop = _LineProcess(
            [
                sys.executable,
                "-u",
                "-m",
                "excavator_il.cli",
                "teleop",
                "--config",
                str(self._config.teleop_config),
                "--print-every",
                str(self._config.teleop_print_every),
            ],
            log_path=teleop_log,
            prefix="teleop",
            output=self._output,
            echo_output=False,
        )

    @classmethod
    def _accepted_safe_ack(cls, line: str) -> bool:
        match = cls._ACK.search(line)
        if match is None:
            return False
        rejected = int(match.group("rejected"))
        deadman = match.group("deadman") == "True"
        if rejected or deadman:
            raise RuntimeError(
                "teleop ACK gate failed: rejected ACK or deadman pressed before standby"
            )
        return int(match.group("accepted")) > 0

    def wait_for_ack(self, timeout_s: int) -> None:
        if self._teleop is None:
            raise RuntimeError("teleop is not running")
        self._teleop_cursor, _ = self._teleop.wait_for(
            self._accepted_safe_ack,
            timeout_s,
            after_index=self._teleop_cursor,
        )

    @classmethod
    def _deadman_is(cls, line: str, expected: bool) -> bool:
        match = cls._ACK.search(line)
        return match is not None and (match.group("deadman") == "True") is expected

    def _wait_for_deadman(self, expected: bool) -> None:
        if self._teleop is None:
            raise RuntimeError("teleop is not running")
        self._teleop_cursor, _ = self._teleop.wait_for(
            lambda line: self._deadman_is(line, expected),
            None,
            after_index=self._teleop_cursor,
        )

    def wait_for_deadman_pressed(self) -> None:
        self._wait_for_deadman(True)

    def wait_for_deadman_released(self) -> None:
        self._wait_for_deadman(False)

    def monitor_deadman_released(self, duration_s: int) -> None:
        """Require continuous safe ACK samples for one bounded soak interval."""
        if self._teleop is None:
            raise RuntimeError("teleop is not running")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        deadline = time.monotonic() + duration_s
        sample_count = 0
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            try:
                self._teleop_cursor, line = self._teleop.wait_for(
                    lambda candidate: self._ACK.search(candidate) is not None,
                    min(1.0, remaining_s),
                    after_index=self._teleop_cursor,
                )
            except _LineWaitTimeout:
                if time.monotonic() >= deadline:
                    break
                raise
            match = self._ACK.search(line)
            assert match is not None
            if match.group("deadman") == "True":
                raise RuntimeError("zero-command soak failed: deadman was pressed")
            if int(match.group("rejected")):
                raise RuntimeError("zero-command soak failed: Collector rejected an ACK")
            sample_count += 1
        if sample_count < duration_s * 15:
            raise RuntimeError(
                f"zero-command soak received only {sample_count} teleop samples"
            )

    def start_episode(self) -> str:
        response = self._remote_cli(
            [
                "episode",
                "--config",
                str(self._config.orin_collection_config),
                "start",
                "--task",
                self._config.task,
                "--operator",
                self._config.operator_id,
                "--dig-target-m",
                *(str(value) for value in self._config.dig_target_m),
                "--material-id",
                self._config.material_id,
            ]
        )
        path = self._episode_path(response)
        self._started_episode_paths = (*self._started_episode_paths, path)
        return path

    def seal_episode(self) -> str:
        response = self._remote_cli(
            [
                "episode",
                "--config",
                str(self._config.orin_collection_config),
                "seal",
            ]
        )
        path = self._episode_path(response)
        if response.get("status") != "pending_review":
            raise RuntimeError("sealed Episode was not confirmed pending_review")
        if path not in self._started_episode_paths:
            raise RuntimeError("sealed Episode was not started by this run")
        self._discardable_episode_paths = (*self._discardable_episode_paths, path)
        return path

    def finalize_episode(
        self, episode_path: str, result: str, reason: str = ""
    ) -> str:
        response = self._remote_cli(
            [
                "episode",
                "--config",
                str(self._config.orin_collection_config),
                "finalize",
                episode_path,
                "--result",
                result,
                "--failure-reason",
                reason,
            ]
        )
        path = self._episode_path(response)
        if path != episode_path:
            raise RuntimeError("Collector finalized an unexpected Episode path")
        self._discardable_episode_paths = tuple(
            candidate
            for candidate in self._discardable_episode_paths
            if candidate != path
        )
        return path

    def abort_episode(self, reason: str) -> str:
        response = self._remote_cli(
            [
                "episode",
                "--config",
                str(self._config.orin_collection_config),
                "abort",
                "--reason",
                reason,
            ]
        )
        path = self._episode_path(response)
        return path

    def discard_episode(self, episode_path: str) -> None:
        if episode_path not in self._discardable_episode_paths:
            raise RuntimeError("refusing to discard an unapproved Episode path")
        path = PurePosixPath(episode_path)
        if (
            not path.is_absolute()
            or _EPISODE_NAME.fullmatch(path.name) is None
            or ".." in path.parts
            or len(path.parts) < 4
        ):
            raise RuntimeError(f"refusing to discard unsafe Episode path: {path}")
        quoted_path = shlex.quote(str(path))
        self._run_ssh(
            f"test -d {quoted_path} && test ! -L {quoted_path} && "
            f"rm -rf -- {quoted_path} && test ! -e {quoted_path}"
        )
        self._discardable_episode_paths = tuple(
            candidate
            for candidate in self._discardable_episode_paths
            if candidate != episode_path
        )

    def stop_teleop(self) -> None:
        if self._teleop is not None:
            self._teleop.stop(signal.SIGINT)
            self._teleop = None

    def stop_collector(self) -> None:
        if (
            self._collector_pid is not None
            and self._collector is not None
            and self._collector.running
        ):
            self._run_ssh(f"kill -TERM -- -{self._collector_pid}")
        self._collector_pid = None
        if self._collector is not None:
            self._collector.wait(timeout_s=15.0)
            self._collector = None

    def build_and_validate(self, episode_path: str) -> None:
        _, _, validation_log = self.log_paths
        outputs = []
        for argv in (
            ["build-steps", episode_path],
            ["validate", episode_path],
        ):
            response = self._remote_cli(argv)
            outputs.append(json.dumps(dict(response), ensure_ascii=False, indent=2))
        report_path = str(PurePosixPath(episode_path) / "quality_report.json")
        report_text = self._run_ssh(self._in_repo(["cat", report_path]))
        try:
            report = json.loads(report_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("quality_report.json is not valid JSON") from exc
        outputs.append(json.dumps(report, ensure_ascii=False, indent=2))
        validation_log.parent.mkdir(parents=True, exist_ok=True)
        with validation_log.open("a", encoding="utf-8") as log:
            log.write(f"=== {episode_path} ===\n")
            log.write("\n".join(outputs) + "\n")
        self._output(json.dumps(report, ensure_ascii=False, indent=2))

    def inspect_zero_soak(self, episode_path: str) -> Mapping[str, Any]:
        return self._remote_cli(
            ["inspect-zero-soak", episode_path],
            accepted_returncodes=(0, 3),
        )


def run_guided_episode(
    config: GuidedEpisodeConfig,
    operations: GuidedEpisodeOperations,
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> str:
    """Collect deadman-bounded attempts and validate them after motion I/O stops."""
    collector_started = False
    teleop_started = False
    episode_active = False
    deadman_started = False
    pending_path: str | None = None
    completed_path: str | None = None
    retained_paths: tuple[str, ...] = ()
    failure: BaseException | None = None
    cleanup_errors: list[str] = []
    try:
        operations.preflight()
        operations.start_collector()
        collector_started = True
        operations.start_episode()
        episode_active = True
        operations.start_teleop()
        teleop_started = True
        operations.wait_for_ack(config.ack_timeout_s)
        while True:
            output(
                "Recorder 已进入待命。保持双杆居中；按下 deadman 后可立即操纵 XY。"
            )
            operations.wait_for_deadman_pressed()
            deadman_started = True
            output(
                "记录已开始：按住 deadman 完成动作；双杆回中后松开 deadman 结束。"
            )
            operations.wait_for_deadman_released()
            completed_path = operations.seal_episode()
            episode_active = False
            pending_path = completed_path
            output("检测到 deadman 松开，动作命令已回零，Episode 已自动保存。")
            outcome = _read_outcome(input_fn, output)
            if outcome == "success":
                completed_path = operations.finalize_episode(
                    completed_path, "success"
                )
                pending_path = None
            elif outcome == "failure":
                completed_path = operations.finalize_episode(
                    completed_path, "failure", config.failure_reason
                )
                pending_path = None
            retained_paths = (*retained_paths, completed_path)
            if outcome != "retake":
                break
            operations.discard_episode(completed_path)
            pending_path = None
            retained_paths = tuple(
                path for path in retained_paths if path != completed_path
            )
            output(
                f"本次已删除：{completed_path}。双杆居中后可再次按 deadman 重录，"
                "Episode 编号保持不变。"
            )
            operations.start_episode()
            episode_active = True
            deadman_started = False
    except BaseException as exc:
        failure = exc
        if episode_active:
            try:
                completed_path = operations.abort_episode(
                    "guided_episode_interrupted"
                )
                if deadman_started:
                    retained_paths = (*retained_paths, completed_path)
            except Exception as abort_exc:
                output(f"ERROR: failed to abort active Episode: {abort_exc}")
            episode_active = False
        elif pending_path is not None:
            try:
                completed_path = operations.finalize_episode(
                    pending_path,
                    "aborted",
                    "guided_episode_interrupted",
                )
                retained_paths = (*retained_paths, completed_path)
            except Exception as finalize_exc:
                output(
                    "ERROR: failed to finalize sealed Episode after interruption: "
                    f"{finalize_exc}"
                )
            pending_path = None
    finally:
        if teleop_started:
            try:
                operations.stop_teleop()
            except Exception as exc:
                cleanup_errors.append(f"teleop cleanup failed: {exc}")
        if collector_started:
            try:
                operations.stop_collector()
            except Exception as exc:
                cleanup_errors.append(f"Collector cleanup failed: {exc}")
        if cleanup_errors:
            message = "; ".join(cleanup_errors)
            if failure is not None:
                output(f"ERROR: {message}")
            else:
                failure = RuntimeError(message)
    for episode_path in retained_paths:
        try:
            operations.build_and_validate(episode_path)
        except BaseException as build_exc:
            if failure is None:
                failure = build_exc
            else:
                output(
                    f"ERROR: failed to validate retained Episode "
                    f"{episode_path}: {build_exc}"
                )
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    assert completed_path is not None
    return completed_path


def _read_outcome(
    input_fn: Callable[[str], str], output: Callable[[str], None]
) -> str:
    choices = {
        "成功": "success",
        "s": "success",
        "失败": "failure",
        "f": "failure",
        "重录": "retake",
        "r": "retake",
    }
    while True:
        value = input_fn(
            "请输入结果（成功/s、失败/f、重录/r）后按 Enter："
        ).strip().lower()
        outcome = choices.get(value)
        if outcome is not None:
            return outcome
        output("无法识别结果，请输入：成功、失败或重录。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="collect and validate one guided diagnostic Episode"
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="guided PC workflow configuration",
    )
    args = parser.parse_args(argv)
    try:
        config = GuidedEpisodeConfig.load(args.config)
        operations = SystemGuidedEpisodeOperations(config)
        path = run_guided_episode(config, operations)
    except KeyboardInterrupt:
        print("guided Episode aborted by operator", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    collector_log, teleop_log, validation_log = operations.log_paths
    print(f"Episode complete and validated: {path}")
    print(f"collector log: {collector_log}")
    print(f"teleop log: {teleop_log}")
    print(f"validation log: {validation_log}")
    return 0

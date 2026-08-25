"""PC/Orin process and Episode lifecycle operations for guided collection."""

from __future__ import annotations

import json
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from ._guided_episode_config import GuidedEpisodeConfig
from ._guided_episode_rl import _GuidedEpisodeRlOperations
from ._guided_episode_targets import (
    capture_target_source_provenance as _capture_target_source_provenance,
)
from .collector.config import (
    validate_collection_protocol,
    validate_recording_purpose,
    validate_target_source_provenance,
)
from .remote_runtime import LineProcess, LineWaitTimeout, SshRuntimeHost


_EPISODE_NAME = re.compile(r"episode_\d{4,}")


class SystemGuidedEpisodeOperations(_GuidedEpisodeRlOperations):
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
        line_process_factory: Callable[..., Any] = LineProcess,
    ) -> None:
        self._config = config
        self._remote_host: SshRuntimeHost | None = None
        self._output = output
        self._timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._line_process_factory = line_process_factory
        self._collector: Any | None = None
        self._teleop: Any | None = None
        self._collector_pid: int | None = None
        self._rl_runtime: Any | None = None
        self._rl_runtime_pid: int | None = None
        self._operator_preview: Any | None = None
        self._operator_preview_pid: int | None = None
        self._rl_hardware_start_gate: PurePosixPath | None = None
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
        return self._ssh_host().argv(remote_command)

    def _in_repo(self, argv: list[str]) -> str:
        return (
            f"cd {shlex.quote(str(self._config.orin_repo))} && "
            f"{shlex.join(argv)}"
        )

    def _in_remote_rl_repo(self, argv: list[str]) -> str:
        return (
            f"cd {shlex.quote(str(self._config.rl_orin_repo))} && "
            f"{shlex.join(argv)}"
        )

    def _run_ssh(
        self,
        remote_command: str,
        *,
        accepted_returncodes: tuple[int, ...] = (0,),
    ) -> str:
        return self._ssh_host().run(
            remote_command, accepted_returncodes=accepted_returncodes
        )

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

    def _ssh_host(self) -> SshRuntimeHost:
        if self._remote_host is None:
            self._remote_host = SshRuntimeHost(
                self._config.orin_ssh_host, run_command=subprocess.run
            )
        return self._remote_host

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
        self._reclaim_known_serial_owner()

    def _known_serial_owner_argv(self) -> tuple[tuple[str, ...], ...]:
        collector = (
            str(self._config.orin_executable),
            "collect",
            "--config",
            str(self._config.orin_collection_config),
        )
        rl_runtime = (
            str(self._config.rl_orin_python),
            "-u",
            "orin_state_sender.py",
            "--serial-port",
            str(self._config.rl_serial_port),
            "--control-enabled",
            "--pc-host",
            self._config.rl_pc_host,
            "--edge-config",
            str(self._config.rl_edge_config),
            "--edge-motion-authorization",
            "ALLOW_EDGE_MACHINE_MOTION",
            "--print-every",
            "100",
        )
        # A manually launched but otherwise identical Runtime may use the
        # environment's ``python`` command instead of the configured absolute
        # interpreter path.  Match the complete behavior argv from ``-u``
        # onward so workflow takeover remains exact without depending on how
        # that interpreter was named.
        rl_runtime_interpreter_independent = rl_runtime[1:]
        return collector, rl_runtime, rl_runtime_interpreter_independent

    def _operator_preview_argv(self) -> tuple[str, ...]:
        return (
            str(self._config.orin_executable),
            "camera-preview",
            "--config",
            str(self._config.orin_collection_config),
        )

    def _reclaim_known_camera_owner(self) -> None:
        result = self._ssh_host().reclaim_serial_owner(
            serial_path="/dev/video0",
            known_argv_suffixes=(self._operator_preview_argv(),),
            timeout_s=self._config.rl_serial_release_timeout_s,
            execute=self._run_ssh,
        )
        if result == "reclaimed":
            self._output("检测到并释放了上一次遗留的 Orin 相机预览进程。")

    def _reclaim_known_serial_owner(self) -> None:
        result = self._ssh_host().reclaim_serial_owner(
            serial_path=self._config.rl_serial_port,
            known_argv_suffixes=self._known_serial_owner_argv(),
            timeout_s=self._config.rl_serial_release_timeout_s,
            execute=self._run_ssh,
        )
        if result == "reclaimed":
            self._output("检测到并释放了上一次遗留的 Orin 串口 Runtime。")


    def capture_target_source_provenance(
        self,
        point_id: str,
        expected_target_m: tuple[float, float, float],
    ) -> Mapping[str, str | bool]:
        return _capture_target_source_provenance(
            self._config, point_id, expected_target_m
        )

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
        self._collector = self._line_process_factory(
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
        self._teleop = self._line_process_factory(
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
            except LineWaitTimeout:
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

    def start_episode(
        self,
        dig_target_m: tuple[float, float, float] | None = None,
        *,
        task_variant: str | None = None,
        soil_reset_block_id: str | None = None,
        dig_point_id: str | None = None,
        recording_purpose: str = "demonstration",
        target_source_provenance: Mapping[str, Any] | None = None,
    ) -> str:
        target = self._config.dig_target_m if dig_target_m is None else dig_target_m
        protocol = validate_collection_protocol(
            task_variant=task_variant,
            soil_reset_block_id=soil_reset_block_id,
            dig_point_id=dig_point_id,
        )
        recording_purpose = validate_recording_purpose(recording_purpose)
        normalized_target_source = (
            None
            if target_source_provenance is None
            else validate_target_source_provenance(
                target_source_provenance
            )
        )
        if (
            recording_purpose == "demonstration"
            and protocol
            and normalized_target_source is None
        ):
            raise ValueError(
                "formal demonstration requires target_source_provenance"
            )
        command = [
            "episode",
            "--config",
            str(self._config.orin_collection_config),
            "start",
            "--task",
            self._config.task,
            "--operator",
            self._config.operator_id,
            "--dig-target-m",
            *(str(value) for value in target),
            "--material-id",
            self._config.material_id,
        ]
        if recording_purpose != "demonstration":
            command.extend(["--recording-purpose", recording_purpose])
        if protocol:
            command.extend(
                [
                    "--task-variant",
                    protocol["task_variant"],
                    "--soil-reset-block-id",
                    protocol["soil_reset_block_id"],
                    "--dig-point-id",
                    protocol["dig_point_id"],
                ]
            )
            if normalized_target_source is not None:
                command.extend(
                    [
                        "--target-source-provenance-json",
                        json.dumps(
                            dict(normalized_target_source),
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ]
                )
        response = self._remote_cli(command)
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
        collector = self._collector
        collector_pid = self._collector_pid
        if (
            collector_pid is not None
            and collector is not None
            and collector.running
        ):
            self._run_ssh(f"kill -TERM -- -{collector_pid}")
        try:
            if collector is not None:
                collector.wait(timeout_s=2.0)
        except subprocess.TimeoutExpired as exc:
            if collector_pid is None:
                assert collector is not None
                collector.stop(signal.SIGKILL, timeout_s=2.0)
                raise RuntimeError(
                    "Collector SSH transport did not exit and its remote PID is unknown"
                ) from exc
            remote_state = self._run_ssh(
                "for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 "
                "18 19 20; do "
                f"if ! kill -0 -- -{collector_pid} 2>/dev/null; then "
                "echo exited; exit 0; fi; sleep 0.25; done; echo running"
            ).strip()
            assert collector is not None
            collector.stop(signal.SIGKILL, timeout_s=2.0)
            if remote_state != "exited":
                raise RuntimeError(
                    f"remote Collector process group {collector_pid} is still running "
                    "after TERM timeout"
                ) from exc
        finally:
            self._collector = None
            self._collector_pid = None

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
        if self._config.orin_experiment_run_config is not None:
            evidence = self._remote_cli(
                [
                    "record-collection-run",
                    episode_path,
                    "--config",
                    str(self._config.orin_experiment_run_config),
                ]
            )
            outputs.append(json.dumps(dict(evidence), ensure_ascii=False, indent=2))
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

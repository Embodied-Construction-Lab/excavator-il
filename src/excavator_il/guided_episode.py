"""PC-side deadman-guided hardware Episode collection."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from ._guided_episode_config import (
    GUIDED_EPISODE_CONFIG_SCHEMA_VERSION,
    GuidedEpisodeConfig,
)
from ._guided_episode_system import SystemGuidedEpisodeOperations
from ._guided_episode_targets import load_rl_dig_targets, resolve_rl_dig_target
from .collector.config import validate_collection_protocol


_BRACKETED_PASTE_MARKER = re.compile(r"\x1b\[(?:200|201)~")
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config/guided_episode.pc.json"


class PositioningMode(str, Enum):
    DIRECT = "direct"
    MANUAL = "manual"
    RL = "rl"


class GuidedEpisodeStage(str, Enum):
    PREFLIGHT = "preflight"
    RL_POSITIONING = "rl_positioning"
    COLLECTOR_STARTING = "collector_starting"
    MANUAL_POSITIONING = "manual_positioning"
    TELEOPERATION = "teleoperation"
    RECORDER_STANDBY = "recorder_standby"
    RECORDING = "recording"
    REVIEW = "review"
    FINALIZING = "finalizing"
    VALIDATING = "validating"
    COMPLETED = "completed"


class GuidedEpisodeOperations(Protocol):
    def preflight(self) -> None: ...

    def start_rl_runtime(self) -> None: ...

    def run_rl_preposition(
        self, target_id: str | None = None
    ) -> tuple[float, float, float]: ...

    def stop_rl_runtime_and_wait_for_serial(self) -> None: ...

    def capture_target_source_provenance(
        self,
        point_id: str,
        expected_target_m: tuple[float, float, float],
    ) -> Mapping[str, str | bool]: ...

    def require_expected_campaign_slot(
        self,
        *,
        task_variant: str,
        soil_reset_block_id: str,
        dig_point_id: str,
    ) -> None: ...

    def start_collector(self) -> None: ...

    def start_teleop(self) -> None: ...

    def wait_for_ack(self, timeout_s: int) -> None: ...

    def wait_for_deadman_pressed(self) -> None: ...

    def wait_for_deadman_released(self) -> None: ...

    def start_episode(
        self,
        dig_target_m: tuple[float, float, float] | None = None,
        *,
        task_variant: str | None = None,
        soil_reset_block_id: str | None = None,
        dig_point_id: str | None = None,
        recording_purpose: str = "demonstration",
        target_source_provenance: Mapping[str, Any] | None = None,
    ) -> str: ...

    def seal_episode(self) -> str: ...

    def finalize_episode(
        self, episode_path: str, result: str, reason: str = ""
    ) -> str: ...

    def abort_episode(self, reason: str) -> str: ...

    def discard_episode(self, episode_path: str) -> None: ...

    def stop_teleop(self) -> None: ...

    def stop_collector(self) -> None: ...

    def build_and_validate(self, episode_path: str) -> None: ...


def run_standalone_teleop(
    config: GuidedEpisodeConfig,
    operations: GuidedEpisodeOperations,
    *,
    wait_fn: Callable[[], None],
    output: Callable[[str], None] = print,
    stage_callback: Callable[[GuidedEpisodeStage], None] | None = None,
) -> None:
    """Run deadman-gated manual control without creating an Episode."""
    collector_started = False
    teleop_started = False
    failure: BaseException | None = None
    cleanup_errors: list[str] = []
    emit_stage = stage_callback or (lambda _stage: None)
    try:
        emit_stage(GuidedEpisodeStage.PREFLIGHT)
        operations.preflight()
        emit_stage(GuidedEpisodeStage.COLLECTOR_STARTING)
        operations.start_collector()
        collector_started = True
        operations.start_teleop()
        teleop_started = True
        operations.wait_for_ack(config.ack_timeout_s)
        emit_stage(GuidedEpisodeStage.TELEOPERATION)
        output(
            "仅遥操作已就绪：按住 deadman 后用双杆控制；释放 deadman 立即回零。"
            "按 Ctrl+C 或点击安全停止退出，不会创建 Episode。"
        )
        wait_fn()
    except BaseException as exc:
        failure = exc
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
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    emit_stage(GuidedEpisodeStage.COMPLETED)


def run_guided_episode(
    config: GuidedEpisodeConfig,
    operations: GuidedEpisodeOperations,
    *,
    preposition: bool = False,
    positioning_mode: PositioningMode | str | None = None,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    stage_callback: Callable[[GuidedEpisodeStage], None] | None = None,
    rl_target_id: str | None = None,
    task_variant: str | None = None,
    soil_reset_block_id: str | None = None,
    dig_point_id: str | None = None,
) -> str:
    """Collect deadman-bounded attempts and validate them after motion I/O stops."""
    if positioning_mode is None:
        mode = PositioningMode.MANUAL if preposition else PositioningMode.DIRECT
    else:
        mode = PositioningMode(positioning_mode)
        if preposition and mode is not PositioningMode.MANUAL:
            raise ValueError("preposition=True conflicts with positioning_mode")
    collector_started = False
    rl_runtime_started = False
    teleop_started = False
    episode_active = False
    deadman_started = False
    pending_path: str | None = None
    completed_path: str | None = None
    retained_paths: tuple[str, ...] = ()
    failure: BaseException | None = None
    cleanup_errors: list[str] = []
    # Protocol-free CLI runs predate the formal campaign and intentionally
    # retain the configured fallback target.  A formal protocol always names
    # a point and must resolve it from the authoritative Airy demo config.
    episode_target_m = config.dig_target_m
    protocol = validate_collection_protocol(
        task_variant=task_variant,
        soil_reset_block_id=soil_reset_block_id,
        dig_point_id=dig_point_id,
    )
    selected_rl_target_id = rl_target_id
    target_source_provenance: Mapping[str, str | bool] | None = None
    if protocol:
        protocol_target_id = protocol["dig_point_id"]
        if (
            mode is PositioningMode.RL
            and rl_target_id is not None
            and rl_target_id != protocol_target_id
        ):
            raise ValueError(
                "RL target id must match the collection protocol dig_point_id"
            )
        if mode is PositioningMode.RL:
            selected_rl_target_id = protocol_target_id
        episode_target_m = resolve_rl_dig_target(
            config,
            protocol_target_id,
        )
        target_source_provenance = (
            operations.capture_target_source_provenance(
                protocol_target_id,
                episode_target_m,
            )
        )
    emit_stage = stage_callback or (lambda _stage: None)

    def start_episode() -> str:
        if not protocol:
            return operations.start_episode(episode_target_m)
        refreshed_target_source = operations.capture_target_source_provenance(
            protocol["dig_point_id"],
            episode_target_m,
        )
        if dict(refreshed_target_source) != dict(target_source_provenance or {}):
            raise ValueError(
                "AiryLidar target source changed before Episode creation"
            )
        operations.require_expected_campaign_slot(**protocol)
        return operations.start_episode(
            episode_target_m,
            **protocol,
            target_source_provenance=refreshed_target_source,
        )

    try:
        emit_stage(GuidedEpisodeStage.PREFLIGHT)
        operations.preflight()
        if protocol and mode in {PositioningMode.RL, PositioningMode.MANUAL}:
            operations.require_expected_campaign_slot(**protocol)
        if mode is PositioningMode.RL:
            emit_stage(GuidedEpisodeStage.RL_POSITIONING)
            output(
                "RL 定位阶段：将按 AiryLidar Mission 配置执行 Plan DIG → Follow。"
            )
            operations.start_rl_runtime()
            rl_runtime_started = True
            if selected_rl_target_id is None:
                episode_target_m = operations.run_rl_preposition()
            else:
                positioned_target_m = operations.run_rl_preposition(
                    selected_rl_target_id
                )
                if protocol and positioned_target_m != episode_target_m:
                    raise RuntimeError(
                        "RL positioning target does not match the selected "
                        "collection dig point"
                    )
                episode_target_m = positioned_target_m
            operations.stop_rl_runtime_and_wait_for_serial()
            rl_runtime_started = False
            output(
                "RL Follow 已成功归零，RL Runtime 已退出并释放串口；"
                "开始切换到人工示教 Collector。"
            )
        emit_stage(GuidedEpisodeStage.COLLECTOR_STARTING)
        operations.start_collector()
        collector_started = True
        if mode is PositioningMode.MANUAL:
            operations.start_teleop()
            teleop_started = True
            operations.wait_for_ack(config.ack_timeout_s)
            emit_stage(GuidedEpisodeStage.MANUAL_POSITIONING)
            output(
                "预定位阶段（不记录 Episode）：按住 deadman，用双杆把挖掘机移动到 "
                "RL Follow 的交接位姿附近。"
            )
            _wait_for_preposition_complete(input_fn, output)
            operations.wait_for_deadman_released()
            operations.stop_teleop()
            teleop_started = False
            output(
                "预定位结束：已确认 deadman 释放并停止预定位 teleop。"
                "请保持双杆 X/Y/Z 全部回中，开始正式 Recorder 门禁。"
            )
        start_episode()
        episode_active = True
        operations.start_teleop()
        teleop_started = True
        operations.wait_for_ack(config.ack_timeout_s)
        while True:
            emit_stage(GuidedEpisodeStage.RECORDER_STANDBY)
            output(
                "Recorder 已进入待命。保持双杆 X/Y/Z 全部回中；按下 deadman 后可立即操纵 XY。"
            )
            operations.wait_for_deadman_pressed()
            deadman_started = True
            emit_stage(GuidedEpisodeStage.RECORDING)
            output(
                "记录已开始：按住 deadman 完成动作；记录阶段只执行 XY，完成后将 X/Y/Z 全部回中并松开 deadman。"
            )
            operations.wait_for_deadman_released()
            completed_path = operations.seal_episode()
            episode_active = False
            pending_path = completed_path
            output("检测到 deadman 松开，动作命令已回零，Episode 已自动保存。")
            emit_stage(GuidedEpisodeStage.REVIEW)
            outcome = _read_outcome(input_fn, output)
            emit_stage(GuidedEpisodeStage.FINALIZING)
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
                f"本次已删除：{completed_path}。双杆 X/Y/Z 全部回中后可再次按 deadman 重录，"
                "Episode 编号保持不变。"
            )
            start_episode()
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
        if rl_runtime_started:
            try:
                operations.stop_rl_runtime_and_wait_for_serial()
            except Exception as exc:
                cleanup_errors.append(f"RL Runtime cleanup failed: {exc}")
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
    emit_stage(GuidedEpisodeStage.VALIDATING)
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
    emit_stage(GuidedEpisodeStage.COMPLETED)
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
        raw_value = input_fn("请输入结果（成功/s、失败/f、重录/r）后按 Enter：")
        value = _BRACKETED_PASTE_MARKER.sub("", raw_value).strip().lower()
        outcome = choices.get(value)
        if outcome is not None:
            return outcome
        output("无法识别结果，请输入：成功、失败或重录。")


def _read_positioning_choice(
    input_fn: Callable[[str], str], output: Callable[[str], None]
) -> PositioningMode:
    choices = {
        "": PositioningMode.DIRECT,
        "rl定位": PositioningMode.RL,
        "rl": PositioningMode.RL,
        "l": PositioningMode.RL,
        "人工预定位": PositioningMode.MANUAL,
        "预定位": PositioningMode.MANUAL,
        "y": PositioningMode.MANUAL,
        "yes": PositioningMode.MANUAL,
        "直接采集": PositioningMode.DIRECT,
        "n": PositioningMode.DIRECT,
        "no": PositioningMode.DIRECT,
    }
    while True:
        value = input_fn(
            "选择采集前定位方式（RL定位/l、人工预定位/y、直接采集/n，默认 n）："
        ).strip().lower()
        choice = choices.get(value)
        if choice is not None:
            return choice
        output("无法识别选择，请输入：RL定位/l、人工预定位/y 或直接采集/n。")


def _wait_for_preposition_complete(
    input_fn: Callable[[str], str], output: Callable[[str], None]
) -> None:
    while True:
        value = input_fn(
            "预定位完成后，将双杆 X/Y/Z 全部回中并松开 deadman，再输入 完成/c："
        ).strip().lower()
        if value in {"完成", "c", "complete"}:
            return
        output("预定位仍在进行；完成后请输入：完成/c。")


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
        positioning_mode = _read_positioning_choice(input, print)
        path = run_guided_episode(
            config, operations, positioning_mode=positioning_mode
        )
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

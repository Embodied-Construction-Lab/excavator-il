"""Command-line entry points for validating and preparing excavator demonstrations."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="excavator-il")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate one raw RGB episode")
    validate.add_argument("episode")
    validate.add_argument("--max-camera-age-ms", type=float, default=120.0)
    validate.add_argument("--max-action-age-ms", type=float, default=100.0)

    convert = commands.add_parser("convert", help="convert raw episodes to LeRobotDataset v3")
    convert.add_argument("episodes", nargs="+")
    convert.add_argument("--output-root", required=True)
    convert.add_argument("--repo-id", default="local/excavator_rgb_v1")
    convert.add_argument("--fps", type=int, default=10)
    convert.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="explicitly allow pipeline-only synthetic Episodes",
    )

    split = commands.add_parser(
        "prepare-training-split",
        help="create a stable parent-Episode train/validation manifest",
    )
    split.add_argument("--dataset-root", required=True)
    split.add_argument("--repo-id", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--seed", type=int, default=0)

    materialize = commands.add_parser(
        "materialize-training-split",
        help="write isolated train/validation datasets with subset statistics",
    )
    materialize.add_argument("--manifest", required=True)
    materialize.add_argument("--output-root", required=True)

    swing_zero = commands.add_parser(
        "derive-zero-swing-split",
        help="copy a materialized split and force expert action_swing labels to zero",
    )
    swing_zero.add_argument("--source-root", required=True)
    swing_zero.add_argument("--output-root", required=True)
    swing_zero.add_argument("--repo-suffix", default="swing_zero")

    evaluate = commands.add_parser(
        "evaluate-checkpoints",
        help="rank ACT checkpoints on held-out Episodes with action safety gates",
    )
    evaluate.add_argument("checkpoints", nargs="+")
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.add_argument("--deployment-manifest")
    evaluate.add_argument("--machine-profile")
    evaluate.add_argument("--max-deployment-prior-l1", type=float)

    smoke = commands.add_parser("smoke-train", help="run one CPU ACT train and inference step")
    smoke.add_argument("--image-height", type=int, default=64)
    smoke.add_argument("--image-width", type=int, default=64)
    smoke.add_argument("--chunk-size", type=int, default=10)

    infer = commands.add_parser(
        "smoke-infer",
        help="reload an ACT checkpoint and infer one LeRobotDataset sample",
    )
    infer.add_argument("checkpoint")
    infer.add_argument("--dataset-root", required=True)
    infer.add_argument("--repo-id", required=True)
    infer.add_argument("--sample-index", type=int, default=0)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--warmup-runs", type=int, default=0)
    infer.add_argument("--timed-runs", type=int, default=1)
    infer.add_argument("--max-inference-ms", type=float)

    synthesize = commands.add_parser(
        "synthesize-episodes",
        help="duplicate one Episode for isolated offline pipeline validation",
    )
    synthesize.add_argument("source_episode")
    synthesize.add_argument("--output-root", required=True)
    synthesize.add_argument("--count", type=int, required=True)

    teleop = commands.add_parser("teleop", help="send two PC joysticks to Orin at 20 Hz")
    teleop.add_argument("--config", default="config/teleop.pc.json")
    teleop.add_argument("--print-every", type=int, default=20)

    commands.add_parser("list-joysticks", help="list stable pygame joystick identities")

    diagnose_joysticks = commands.add_parser(
        "diagnose-joysticks",
        help="interactively verify local joystick axes and deadman without network I/O",
    )
    diagnose_joysticks.add_argument("--config", default="config/teleop.pc.json")

    collect = commands.add_parser("collect", help="run the Orin hardware collector")
    collect.add_argument("--config", default="config/collection.orin.json")

    diagnose_stm32 = commands.add_parser(
        "diagnose-stm32-link",
        help="read-only check of the Orin USART2 telemetry link",
    )
    diagnose_stm32.add_argument("--config", default="config/collection.orin.json")
    diagnose_stm32.add_argument("--duration-s", type=float, default=10.0)

    act_runtime = commands.add_parser(
        "act-runtime", help="run fail-closed online ACT inference on Orin"
    )
    act_runtime.add_argument("--config", default="config/act_runtime.orin.json")
    act_runtime.add_argument(
        "--motion-authorization",
        help="exact explicit authorization; omitted means no-write shadow mode",
    )
    act_runtime.add_argument(
        "--max-steps",
        type=int,
        help="stop safely after this many post-warmup 10 Hz inference steps",
    )

    inspect_runtime = commands.add_parser(
        "inspect-act-runtime-log",
        help="offline acceptance check for a completed ACT Runtime JSONL log",
    )
    inspect_runtime.add_argument("log")
    inspect_runtime.add_argument("--mode", choices=("shadow", "motion"), required=True)
    inspect_runtime.add_argument("--config", default="config/act_runtime.orin.json")

    build = commands.add_parser("build-steps", help="build causal 10 Hz ACT steps")
    build.add_argument("episode")
    build.add_argument("--max-action-age-ms", type=float, default=100.0)
    build.add_argument("--max-camera-age-ms", type=float, default=120.0)

    zero_soak = commands.add_parser(
        "inspect-zero-soak", help="validate one deadman-released diagnostic Episode"
    )
    zero_soak.add_argument("episode")

    episode = commands.add_parser("episode", help="control a running local Collector")
    episode.add_argument("--config", default="config/collection.orin.json")
    episode_commands = episode.add_subparsers(dest="episode_command", required=True)
    start = episode_commands.add_parser("start")
    start.add_argument("--task", required=True)
    start.add_argument("--operator", required=True, dest="operator_id")
    start.add_argument("--dig-target-m", nargs=3, type=float)
    start.add_argument("--material-id")
    stop = episode_commands.add_parser("stop")
    result = stop.add_mutually_exclusive_group(required=True)
    result.add_argument("--success", action="store_true")
    result.add_argument("--failure-reason")
    stop.add_argument("--intervention", action="store_true")
    abort = episode_commands.add_parser("abort")
    abort.add_argument("--reason", required=True)
    episode_commands.add_parser("seal")
    finalize = episode_commands.add_parser("finalize")
    finalize.add_argument("path")
    finalize.add_argument(
        "--result", choices=("success", "failure", "aborted"), required=True
    )
    finalize.add_argument("--failure-reason", default="")
    episode_commands.add_parser("status")
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            from .raw_episode import validate_episode

            report = validate_episode(
                args.episode,
                max_camera_age_ms=args.max_camera_age_ms,
                max_action_age_ms=args.max_action_age_ms,
            )
            _print_json(asdict(report))
        elif args.command == "convert":
            from .lerobot_conversion import convert_episodes

            summary = convert_episodes(
                args.episodes,
                args.output_root,
                args.repo_id,
                fps=args.fps,
                allow_synthetic=args.allow_synthetic,
            )
            _print_json(asdict(summary))
        elif args.command == "prepare-training-split":
            from .training_split import prepare_training_split

            split = prepare_training_split(
                dataset_root=args.dataset_root,
                repo_id=args.repo_id,
                output_path=args.output,
                train_ratio=args.train_ratio,
                seed=args.seed,
            )
            _print_json(asdict(split))
        elif args.command == "materialize-training-split":
            from .training_split import materialize_training_split

            split = materialize_training_split(
                manifest_path=args.manifest,
                output_root=args.output_root,
            )
            _print_json(asdict(split))
        elif args.command == "derive-zero-swing-split":
            from .action_dataset_transform import derive_zero_swing_split

            split = derive_zero_swing_split(
                source_root=args.source_root,
                output_root=args.output_root,
                repo_suffix=args.repo_suffix,
            )
            _print_json(asdict(split))
        elif args.command == "evaluate-checkpoints":
            from .checkpoint_evaluation import (
                evaluate_act_checkpoints,
                write_act_deployment_manifest,
            )

            result = evaluate_act_checkpoints(
                checkpoint_paths=args.checkpoints,
                split_root=args.split_root,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            manifest_inputs = (
                args.deployment_manifest,
                args.machine_profile,
                args.max_deployment_prior_l1,
            )
            if any(value is not None for value in manifest_inputs) and not all(
                value is not None for value in manifest_inputs
            ):
                raise ValueError(
                    "deployment manifest, machine profile, and max L1 must be provided together"
                )
            if args.deployment_manifest:
                write_act_deployment_manifest(
                    result=result,
                    split_root=args.split_root,
                    machine_profile_path=args.machine_profile,
                    output_path=args.deployment_manifest,
                    max_deployment_prior_l1=args.max_deployment_prior_l1,
                )
            _print_json(asdict(result))
            return 0 if result.selected_checkpoint is not None else 3
        elif args.command == "smoke-train":
            from .act_smoke import run_act_smoke_train_step

            result = run_act_smoke_train_step(
                image_shape=(3, args.image_height, args.image_width),
                state_dim=11,
                action_dim=4,
                chunk_size=args.chunk_size,
            )
            _print_json(asdict(result))
        elif args.command == "smoke-infer":
            from .act_smoke import run_act_checkpoint_inference

            result = run_act_checkpoint_inference(
                checkpoint_path=args.checkpoint,
                dataset_root=args.dataset_root,
                repo_id=args.repo_id,
                sample_index=args.sample_index,
                device=args.device,
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
                max_inference_ms=args.max_inference_ms,
            )
            _print_json(asdict(result))
        elif args.command == "synthesize-episodes":
            from .synthetic_episodes import synthesize_episodes

            result = synthesize_episodes(
                args.source_episode,
                args.output_root,
                count=args.count,
            )
            _print_json(asdict(result))
        elif args.command == "teleop":
            from .teleop import TeleopConfig, run_teleop

            try:
                run_teleop(
                    TeleopConfig.load(args.config), print_every=args.print_every
                )
            except KeyboardInterrupt:
                print("teleop interrupted by operator", file=sys.stderr)
                return 130
        elif args.command == "list-joysticks":
            from .teleop import list_pygame_devices

            _print_json(list_pygame_devices())
        elif args.command == "diagnose-joysticks":
            from .joystick_diagnostic import run_joystick_diagnostic
            from .teleop import TeleopConfig

            try:
                report = run_joystick_diagnostic(TeleopConfig.load(args.config))
            except KeyboardInterrupt:
                print("diagnostic interrupted by operator", file=sys.stderr)
                return 130
            return 0 if report.matches_config else 3
        elif args.command == "collect":
            from .collector.service import run_collector

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
            run_collector(args.config)
        elif args.command == "diagnose-stm32-link":
            from .stm32_link_diagnostic import run_stm32_link_diagnostic

            report = run_stm32_link_diagnostic(
                args.config, duration_s=args.duration_s
            )
            _print_json(asdict(report))
            return 0 if report.passed else 3
        elif args.command == "act-runtime":
            from .act_runtime_service import run_act_runtime

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                force=True,
            )
            run_act_runtime(
                args.config,
                motion_authorization=args.motion_authorization,
                max_steps=args.max_steps,
            )
        elif args.command == "inspect-act-runtime-log":
            from .act_runtime_config import load_act_runtime_config
            from .act_runtime_log import inspect_act_runtime_log

            config = load_act_runtime_config(args.config)
            report = inspect_act_runtime_log(
                args.log,
                mode=args.mode,
                max_state_to_decision_ms=config.max_inference_state_age_ms,
                max_camera_age_ms=config.max_camera_age_ms,
            )
            _print_json(asdict(report))
            return 0 if report.passed else 3
        elif args.command == "build-steps":
            from .episode_builder import build_steps

            report = build_steps(
                args.episode,
                max_action_age_ms=args.max_action_age_ms,
                max_camera_age_ms=args.max_camera_age_ms,
            )
            _print_json(asdict(report))
        elif args.command == "inspect-zero-soak":
            from .zero_soak import inspect_zero_command_episode

            report = inspect_zero_command_episode(args.episode)
            _print_json(asdict(report))
            return 0 if report.passed else 3
        elif args.command == "episode":
            from .collector.client import send_episode_command
            from .collector.config import load_collection_config

            config = load_collection_config(args.config)
            request: dict[str, object] = {"command": args.episode_command}
            if args.episode_command == "start":
                request.update(task=args.task, operator_id=args.operator_id)
                if args.dig_target_m is not None:
                    request["dig_target_m"] = args.dig_target_m
                if args.material_id is not None:
                    request["material_id"] = args.material_id
            elif args.episode_command == "stop":
                request.update(
                    success=bool(args.success),
                    failure_reason=args.failure_reason or "",
                    intervention=args.intervention,
                )
            elif args.episode_command == "abort":
                request["reason"] = args.reason
            elif args.episode_command == "finalize":
                request.update(
                    path=args.path,
                    result=args.result,
                    failure_reason=args.failure_reason,
                )
            _print_json(send_episode_command(config.episode_control_socket, request))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

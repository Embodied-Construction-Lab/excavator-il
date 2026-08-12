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

            run_teleop(TeleopConfig.load(args.config), print_every=args.print_every)
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

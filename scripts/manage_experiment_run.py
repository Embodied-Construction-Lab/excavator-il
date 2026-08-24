#!/usr/bin/env python3
"""Create, inspect, append to, verify, and finalize Experiment Run evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from excavator_il.experiment_run import ExperimentRun, load_experiment_run


CREATE_SPEC_SCHEMA = "experiment_run_create.v1"
CREATE_SPEC_FIELDS = {
    "schema_version",
    "run_kind",
    "task_context",
    "policy_ids",
    "host_topology",
    "repository_paths",
    "config_paths",
    "machine_profile_path",
    "evidence_requirements",
}


def _object_from_text(value: str, label: str) -> dict[str, Any]:
    source = Path(value[1:]) if value.startswith("@") else None
    try:
        raw = source.read_text(encoding="utf-8") if source else value
        decoded = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be a JSON object or @JSON_FILE") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def _load_create_spec(path: Path) -> dict[str, Any]:
    spec = _object_from_text(f"@{path}", "create spec")
    if set(spec) != CREATE_SPEC_FIELDS or spec["schema_version"] != CREATE_SPEC_SCHEMA:
        raise ValueError(
            f"create spec must use {CREATE_SPEC_SCHEMA} with fields {sorted(CREATE_SPEC_FIELDS)}"
        )
    config_paths = dict(spec["config_paths"])
    if "experiment_run_spec" in config_paths:
        raise ValueError("config_paths reserves the experiment_run_spec name")
    config_paths["experiment_run_spec"] = str(path.resolve(strict=True))
    return {**spec, "config_paths": config_paths}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _snapshot_to_json(snapshot: Any) -> dict[str, Any]:
    return {
        "run_id": snapshot.run_id,
        "run_dir": str(snapshot.run_dir),
        "state": snapshot.state,
        "start": _jsonable(snapshot.start),
        "events": _jsonable(snapshot.events),
        "artifacts": _jsonable(snapshot.artifacts),
        "final": _jsonable(snapshot.final),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="atomically create an active run")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--spec", type=Path, required=True)
    create.add_argument("--run-id")

    event = commands.add_parser("event", help="append one evidence event")
    event.add_argument("--root", type=Path, required=True)
    event.add_argument("--run-id", required=True)
    event.add_argument("--event-type", required=True)
    event.add_argument("--payload-json", default="{}")

    artifact = commands.add_parser("artifact", help="register an external artifact")
    artifact.add_argument("--root", type=Path, required=True)
    artifact.add_argument("--run-id", required=True)
    artifact.add_argument("--artifact-id", required=True)
    artifact.add_argument("--role", required=True)
    artifact.add_argument("--path", type=Path, required=True)
    artifact.add_argument("--metadata-json", default="{}")

    finalize = commands.add_parser("finalize", help="publish one final run")
    finalize.add_argument("--root", type=Path, required=True)
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--status", choices=("success", "failure"), required=True)
    finalize.add_argument("--metrics-json", default="{}")
    finalize.add_argument("--summary")

    show = commands.add_parser("show", help="strictly load and print a run")
    show.add_argument("--root", type=Path, required=True)
    show.add_argument("--run-id", required=True)

    verify = commands.add_parser("verify", help="verify all external artifact hashes")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create":
        spec = _load_create_spec(args.spec)
        run = ExperimentRun.create(
            args.root,
            run_id=args.run_id,
            run_kind=spec["run_kind"],
            task_context=spec["task_context"],
            policy_ids=spec["policy_ids"],
            host_topology=spec["host_topology"],
            repository_paths=spec["repository_paths"],
            config_paths=spec["config_paths"],
            machine_profile_path=spec["machine_profile_path"],
            evidence_requirements=spec["evidence_requirements"],
        )
        result: Any = {"run_id": run.run_id, "run_dir": str(run.run_dir), "state": run.state}
    elif args.command == "event":
        run = ExperimentRun.load(args.root, args.run_id)
        result = run.append_event(
            args.event_type, _object_from_text(args.payload_json, "payload-json")
        )
    elif args.command == "artifact":
        run = ExperimentRun.load(args.root, args.run_id)
        result = run.register_artifact(
            args.artifact_id,
            args.path,
            role=args.role,
            metadata=_object_from_text(args.metadata_json, "metadata-json"),
        )
    elif args.command == "finalize":
        run = ExperimentRun.load(args.root, args.run_id)
        result = _snapshot_to_json(
            run.finalize(
                args.status,
                metrics=_object_from_text(args.metrics_json, "metrics-json"),
                summary=args.summary,
            )
        )
    elif args.command == "verify":
        snapshot = load_experiment_run(args.root, args.run_id)
        snapshot.verify_artifacts()
        result = {"run_id": snapshot.run_id, "verified_artifact_count": len(snapshot.artifacts)}
    else:
        result = _snapshot_to_json(load_experiment_run(args.root, args.run_id))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

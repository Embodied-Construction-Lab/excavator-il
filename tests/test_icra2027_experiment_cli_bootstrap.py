from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_path",
    [
        "scripts/inspect_icra2027_experiments.py",
        "scripts/inspect_icra2027_experiment_readiness.py",
        "scripts/preflight_act_dig_transport_dump_reference_assets.py",
        "scripts/record_rl_sim_experiment_run.py",
        "scripts/record_rl_real_experiment_run.py",
        "scripts/evaluate_rl_sim_real_pair.py",
        "scripts/convert_rl_sim_trace.py",
    ],
)
def test_icra2027_experiment_scripts_bootstrap_repo_src_without_pythonpath(
    script_path: str,
) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(REPOSITORY / script_path), "--help"],
        cwd=REPOSITORY,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_collection_ui_script_bootstraps_repo_src_without_editable_install() -> None:
    script = REPOSITORY / "scripts/run_collection_ui.py"

    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        cwd=REPOSITORY,
        env={},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()

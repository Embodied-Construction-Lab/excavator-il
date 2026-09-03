from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/create_rl_trajectory_suite.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src")},
        capture_output=True,
        text=True,
        check=False,
    )


def test_creates_strict_10hz_suite_and_reports_frozen_hash(tmp_path: Path):
    output = tmp_path / "trajectory_suite.json"

    result = _run(
        "--suite-id",
        "icra2027-follow-10hz-v1",
        "--sample-count",
        "4",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "sample_ids": [0, 1, 2, 3],
        "sample_period_s": 0.1,
        "suite_id": "icra2027-follow-10hz-v1",
    }
    report = json.loads(result.stdout)
    assert report == {
        "duration_s": 0.4,
        "output": str(output.resolve()),
        "sample_count": 4,
        "sample_period_s": 0.1,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "suite_id": "icra2027-follow-10hz-v1",
    }


def test_refuses_to_overwrite_existing_suite(tmp_path: Path):
    output = tmp_path / "trajectory_suite.json"
    output.write_text("owner data\n", encoding="utf-8")

    result = _run(
        "--suite-id",
        "suite-001",
        "--sample-count",
        "2",
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert output.read_text(encoding="utf-8") == "owner data\n"
    assert "already exists" in result.stderr


def test_rejects_invalid_suite_identity_and_sample_budget(tmp_path: Path):
    invalid_id = _run(
        "--suite-id",
        "  ",
        "--sample-count",
        "2",
        "--output",
        str(tmp_path / "bad-id.json"),
    )
    invalid_count = _run(
        "--suite-id",
        "suite-001",
        "--sample-count",
        "0",
        "--output",
        str(tmp_path / "bad-count.json"),
    )

    assert invalid_id.returncode == 2
    assert "suite-id" in invalid_id.stderr
    assert invalid_count.returncode == 2
    assert "sample-count" in invalid_count.stderr

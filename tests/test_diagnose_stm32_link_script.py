from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_pc_stm32_diagnostic_has_remote_and_local_hard_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    script = Path(__file__).parents[1] / "scripts" / "diagnose_stm32_link.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--duration-s", "10"],
    )

    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(str(script), run_name="__main__")

    argv, kwargs = calls[0]
    remote_command = argv[-1]
    assert "timeout --signal=TERM --kill-after=2s 15s" in remote_command
    assert kwargs["timeout"] == 20.0


def test_pc_stm32_diagnostic_reports_remote_timeout_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=124),
    )
    script = Path(__file__).parents[1] / "scripts" / "diagnose_stm32_link.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--duration-s", "10"])

    with pytest.raises(SystemExit, match="3"):
        runpy.run_path(str(script), run_name="__main__")

    captured = capsys.readouterr()
    assert "remote timeout terminated it" in captured.err
    assert "Traceback" not in captured.err


def test_pc_stm32_diagnostic_reports_local_timeout_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    script = Path(__file__).parents[1] / "scripts" / "diagnose_stm32_link.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--duration-s", "10"])

    with pytest.raises(SystemExit, match="3"):
        runpy.run_path(str(script), run_name="__main__")

    captured = capsys.readouterr()
    assert "local hard deadline" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "61"])
def test_pc_stm32_diagnostic_rejects_unbounded_duration_before_ssh(
    monkeypatch: pytest.MonkeyPatch,
    duration: str,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("SSH must not start")
        ),
    )
    script = Path(__file__).parents[1] / "scripts" / "diagnose_stm32_link.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--duration-s", duration])

    with pytest.raises(SystemExit, match="2"):
        runpy.run_path(str(script), run_name="__main__")

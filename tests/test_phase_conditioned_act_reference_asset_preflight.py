import json
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/preflight_phase_conditioned_act_reference_assets.py"


def test_real_three_phase_reference_assets_pass_static_preflight():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["motion_commands_emitted"] == 0
    assert report["mission_id"] == "engineering_act_transport_three_phase_reference"
    assert report["behavior_id"] == "act_dig_transport_dump_three_phase"
    assert report["state_dim"] == 14
    assert report["camera_roles"] == ["front", "dump"]
    assert report["phase_boundaries"] == [93, 142]
    assert report["act_max_steps"] == 241
    assert report["checkpoint_model_sha256"] == (
        "8faf364022695312714ccbac3299635e8fc8ee30b1935b0f3122b1c1fdedd637"
    )

import json
from pathlib import Path

import pytest

from excavator_il.site_config import check_site_config


def test_commissioned_site_configs_are_consistent():
    config_dir = Path(__file__).parents[1] / "config"

    summary = check_site_config(config_dir)

    assert summary == {
        "orin_host": "192.168.50.2",
        "pc_host": "192.168.50.1",
        "joystick_port": 18090,
        "preview_port": 18092,
        "machine_state_port": 18081,
        "serial_port": "/dev/ttyTHS1",
    }


def test_site_config_check_reports_the_exact_drift(tmp_path):
    source = Path(__file__).parents[1] / "config"
    for name in (
        "guided_episode.pc.json",
        "teleop.pc.json",
        "collection.orin.json",
        "collection_ui.pc.json",
        "hybrid_mission.pc.json",
    ):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    teleop_path = tmp_path / "teleop.pc.json"
    teleop = json.loads(teleop_path.read_text(encoding="utf-8"))
    teleop["orin_host"] = "192.168.50.99"
    teleop_path.write_text(json.dumps(teleop), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"teleop\.orin_host .* must match Orin host 192\.168\.50\.2",
    ):
        check_site_config(tmp_path)

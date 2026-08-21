"""Read-only consistency check for the commissioned PC/Orin site topology."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_CONFIG_FILES = {
    "guided": "guided_episode.pc.json",
    "teleop": "teleop.pc.json",
    "collection": "collection.orin.json",
    "ui": "collection_ui.pc.json",
    "hybrid": "hybrid_mission.pc.json",
}


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load site config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"site config must be an object: {path}")
    return value


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} {actual!r} must match {expected!r}")


def _url_endpoint(label: str, value: object) -> tuple[str, int]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a URL string")
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
        raise ValueError(f"{label} must be an explicit http://host:port URL")
    return parsed.hostname, parsed.port


def check_site_config(config_dir: str | Path) -> dict[str, object]:
    """Fail closed when active configuration files describe different sites."""

    root = Path(config_dir).expanduser().resolve()
    configs = {
        key: _load_object(root / filename)
        for key, filename in _CONFIG_FILES.items()
    }
    guided = configs["guided"]
    teleop = configs["teleop"]
    collection = configs["collection"]
    ui = configs["ui"]
    hybrid = configs["hybrid"]

    ssh_host = guided["orin"]["ssh_host"]
    if not isinstance(ssh_host, str) or ssh_host.count("@") != 1:
        raise ValueError("guided.orin.ssh_host must be user@host")
    orin_host = ssh_host.split("@", maxsplit=1)[1]
    pc_host = guided["rl_preposition"]["pc_host"]
    joystick_port = collection["joystick_udp"]["port"]
    preview_port = collection["camera_preview_http"]["port"]
    machine_state_port = collection["machine_state_udp"]["port"]
    serial_port = collection["stm32_serial"]["port"]

    if teleop["orin_host"] != orin_host:
        raise ValueError(
            f"teleop.orin_host {teleop['orin_host']} must match Orin host {orin_host}"
        )
    _require_equal(
        "collection.joystick_udp.allowed_pc_host",
        collection["joystick_udp"]["allowed_pc_host"],
        pc_host,
    )
    _require_equal(
        "collection.machine_state_udp.host",
        collection["machine_state_udp"]["host"],
        pc_host,
    )
    _require_equal("teleop.orin_port", teleop["orin_port"], joystick_port)
    _require_equal(
        "guided.rl_preposition.serial_port",
        guided["rl_preposition"]["serial_port"],
        serial_port,
    )
    for label in ("camera_preview_url", "telemetry_url"):
        host, port = _url_endpoint(f"ui.{label}", ui[label])
        _require_equal(f"ui.{label} host", host, orin_host)
        _require_equal(f"ui.{label} port", port, preview_port)
    _require_equal(
        "ui.guided_config",
        Path(ui["guided_config"]).name,
        _CONFIG_FILES["guided"],
    )
    _require_equal(
        "ui.hybrid_mission_config",
        Path(ui["hybrid_mission_config"]).name,
        _CONFIG_FILES["hybrid"],
    )
    _require_equal(
        "hybrid.guided_config",
        Path(hybrid["guided_config"]).name,
        _CONFIG_FILES["guided"],
    )
    return {
        "orin_host": orin_host,
        "pc_host": pc_host,
        "joystick_port": joystick_port,
        "preview_port": preview_port,
        "machine_state_port": machine_state_port,
        "serial_port": serial_port,
    }

"""Strict local configuration for the guided collection Web UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


COLLECTION_UI_CONFIG_SCHEMA_VERSION = "excavator_collection_ui_config.v3"
DUAL_CAMERA_COLLECTION_UI_CONFIG_SCHEMA_VERSION = "excavator_collection_ui_config.v2"
LEGACY_COLLECTION_UI_CONFIG_SCHEMA_VERSION = "excavator_collection_ui_config.v1"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


@dataclass(frozen=True)
class CollectionUiConfig:
    guided_config: Path
    host: str
    port: int
    camera_preview_url: str
    visualization_url: str
    camera_dump_preview_url: str = ""
    telemetry_url: str = ""
    hybrid_mission_config: Path | None = None
    hybrid_evidence_config: Path | None = None
    resident_fixed_cycle_config: Path | None = None


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _http_url(value: object, field: str, *, allow_empty: bool = False) -> str:
    text = _text(value, field, allow_empty=allow_empty)
    if not text and allow_empty:
        return ""
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field} must be an absolute HTTP URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError(f"{field} must not contain credentials or a fragment")
    return text


def load_collection_ui_config(path: str | Path) -> CollectionUiConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load collection UI config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("collection UI config must be an object")
    if raw.get("schema_version") not in {
        COLLECTION_UI_CONFIG_SCHEMA_VERSION,
        DUAL_CAMERA_COLLECTION_UI_CONFIG_SCHEMA_VERSION,
        LEGACY_COLLECTION_UI_CONFIG_SCHEMA_VERSION,
    }:
        raise ValueError(
            "schema_version must be "
            f"{COLLECTION_UI_CONFIG_SCHEMA_VERSION} or "
            f"{DUAL_CAMERA_COLLECTION_UI_CONFIG_SCHEMA_VERSION} or "
            f"{LEGACY_COLLECTION_UI_CONFIG_SCHEMA_VERSION}"
        )
    server = raw.get("server")
    if not isinstance(server, dict):
        raise ValueError("server must be an object")
    host = _text(server.get("host"), "server.host")
    if host not in _LOOPBACK_HOSTS:
        raise ValueError("server.host must be a loopback address")
    port = server.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("server.port must be an integer in [1, 65535]")
    hybrid_mission = raw.get("hybrid_mission_config")
    hybrid_evidence = raw.get("hybrid_evidence_config")
    resident_fixed_cycle = raw.get("resident_fixed_cycle_config")
    if resident_fixed_cycle is not None and hybrid_mission is not None:
        raise ValueError(
            "resident_fixed_cycle_config is exclusive with V2 hybrid Mission config"
        )
    return CollectionUiConfig(
        guided_config=(
            config_path.parent
            / _text(raw.get("guided_config"), "guided_config")
        ).resolve(),
        host=host,
        port=port,
        camera_preview_url=_http_url(
            raw.get("camera_preview_url"), "camera_preview_url"
        ),
        camera_dump_preview_url=_http_url(
            raw.get("camera_dump_preview_url", ""),
            "camera_dump_preview_url",
            allow_empty=True,
        ),
        visualization_url=_http_url(
            raw.get("visualization_url", ""),
            "visualization_url",
            allow_empty=True,
        ),
        telemetry_url=_http_url(raw.get("telemetry_url"), "telemetry_url"),
        hybrid_mission_config=(
            None
            if hybrid_mission is None
            else (
                config_path.parent
                / _text(hybrid_mission, "hybrid_mission_config")
            ).resolve()
        ),
        hybrid_evidence_config=(
            None
            if hybrid_evidence is None
            else (
                config_path.parent
                / _text(hybrid_evidence, "hybrid_evidence_config")
            ).resolve()
        ),
        resident_fixed_cycle_config=(
            None
            if resident_fixed_cycle is None
            else (
                config_path.parent
                / _text(resident_fixed_cycle, "resident_fixed_cycle_config")
            ).resolve()
        ),
    )

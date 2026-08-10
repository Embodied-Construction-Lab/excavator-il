"""Unix-socket client for controlling a local Orin Collector Episode."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Mapping


def send_episode_command(
    socket_path: str | Path,
    request: Mapping[str, Any],
    *,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    payload = (json.dumps(dict(request), separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if len(payload) > 16_384:
        raise ValueError("episode control request is too large")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_s)
    try:
        client.connect(str(Path(socket_path).expanduser()))
        client.sendall(payload)
        chunks: list[bytes] = []
        total = 0
        while total <= 65_536:
            chunk = client.recv(min(4096, 65_537 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if b"\n" in chunk:
                break
    finally:
        client.close()
    if total > 65_536:
        raise RuntimeError("collector response is too large")
    raw = b"".join(chunks).split(b"\n", maxsplit=1)[0]
    try:
        response = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid collector response: {exc}") from exc
    if not isinstance(response, dict):
        raise RuntimeError("collector response must be an object")
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "collector command failed")))
    return response

"""Local FastAPI adapter for guided Demonstration Episode collection."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .airy_operator import AiryOperatorSnapshot
from .collection_ui_config import CollectionUiConfig
from .collection_ui_session import CollectionSessionSnapshot
from .hybrid_mission_session import (
    MAX_HYBRID_CYCLE_COUNT,
    HybridMissionSnapshot,
)


@dataclass(frozen=True)
class CollectionUiMetadata:
    operator_id: str
    task: str
    dig_target_m: tuple[float, float, float]
    orin_host: str
    rl_dig_targets: tuple[tuple[str, tuple[float, float, float]], ...]
    hybrid_act_max_steps: int = 0


class CollectionSupervisor(Protocol):
    def snapshot(self) -> CollectionSessionSnapshot: ...

    def start(self, positioning_mode: str, dig_target_id: str | None = None) -> None: ...

    def complete_manual_positioning(self) -> None: ...

    def submit_outcome(self, outcome: str) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class HybridSupervisor(Protocol):
    def snapshot(self) -> HybridMissionSnapshot: ...

    def start(
        self,
        dig_target_id: str,
        *,
        automatic: bool,
        motion_authorization: str | None,
        cycle_count: int = 1,
    ) -> None: ...

    def advance(self, *, motion_authorization: str | None) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class OperatorSupervisor(Protocol):
    def snapshot(self) -> AiryOperatorSnapshot: ...

    def start(self) -> AiryOperatorSnapshot: ...

    def stop(self) -> AiryOperatorSnapshot: ...

    def close(self) -> None: ...


class StartCollectionRequest(BaseModel):
    positioning_mode: Literal["rl", "manual", "direct", "teleop"]
    dig_target_id: str | None = None


class EpisodeOutcomeRequest(BaseModel):
    outcome: Literal["success", "failure", "retake"]


class StartHybridMissionRequest(BaseModel):
    dig_target_id: str
    automatic: bool = False
    cycle_count: int = Field(default=1, ge=1, le=MAX_HYBRID_CYCLE_COUNT)
    motion_authorization: str | None = None


class AdvanceHybridMissionRequest(BaseModel):
    motion_authorization: str | None = None


_STATIC_DIR = Path(__file__).with_name("collection_ui_static")
_MAX_CAMERA_SNAPSHOT_BYTES = 4 * 1024 * 1024


def _camera_snapshot_url(preview_url: str) -> str:
    parsed = urlsplit(preview_url)
    if not parsed.path.endswith(".mjpg"):
        raise RuntimeError("Collector camera preview URL must end with .mjpg")
    path = f"{parsed.path[:-len('.mjpg')]}.jpg"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _fetch_collector_camera(url: str) -> bytes:
    try:
        with urlopen(url, timeout=0.6) as response:
            content_type = response.headers.get_content_type()
            payload = response.read(_MAX_CAMERA_SNAPSHOT_BYTES + 1)
    except (HTTPError, URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"Collector camera unavailable: {exc}") from exc
    if content_type != "image/jpeg":
        raise RuntimeError(f"Collector camera returned {content_type}, expected image/jpeg")
    if len(payload) > _MAX_CAMERA_SNAPSHOT_BYTES:
        raise RuntimeError("Collector camera snapshot exceeds 4 MiB")
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise RuntimeError("Collector camera returned an invalid JPEG")
    return payload


def _fetch_collector_telemetry(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=0.4) as response:
            payload = json.load(response)
    except (HTTPError, URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"Collector telemetry unavailable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Collector telemetry response must be an object")
    return payload


def _require_ui_request(ui_header: str | None) -> None:
    if ui_header != "1":
        raise HTTPException(status_code=403, detail="local collection UI header required")


def create_collection_ui_app(
    *,
    config: CollectionUiConfig,
    metadata: CollectionUiMetadata,
    supervisor: CollectionSupervisor,
    hybrid_supervisor: HybridSupervisor | None = None,
    operator_supervisor: OperatorSupervisor | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        supervisor.close()
        if hybrid_supervisor is not None:
            hybrid_supervisor.close()
        if operator_supervisor is not None:
            operator_supervisor.close()

    app = FastAPI(title="Excavator Guided Collection UI", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/api/config")
    def ui_config() -> dict[str, Any]:
        return {
            "operator_id": metadata.operator_id,
            "task": metadata.task,
            "dig_target_m": list(metadata.dig_target_m),
            "orin_host": metadata.orin_host,
            "camera_preview_url": "/api/camera/frame.jpg",
            "visualization_url": config.visualization_url,
            "positioning_modes": ["rl", "manual", "direct", "teleop"],
            "hybrid_mission_enabled": hybrid_supervisor is not None,
            "operator_control_enabled": operator_supervisor is not None,
            "hybrid_act_max_steps": metadata.hybrid_act_max_steps,
            "rl_dig_targets": [
                {"target_id": target_id, "position_m": list(position)}
                for target_id, position in metadata.rl_dig_targets
            ],
        }

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return asdict(supervisor.snapshot())

    @app.get("/api/telemetry")
    def telemetry() -> dict[str, Any]:
        if not config.telemetry_url:
            raise HTTPException(status_code=503, detail="telemetry preview is disabled")
        try:
            return _fetch_collector_telemetry(config.telemetry_url)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/camera/frame.jpg", response_class=Response)
    def camera_frame() -> Response:
        try:
            payload = _fetch_collector_camera(
                _camera_snapshot_url(config.camera_preview_url)
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=payload,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/collection/start")
    def start_collection(
        request: StartCollectionRequest,
        ui_header: str | None = Header(default=None, alias="X-Excavator-UI"),
    ) -> dict[str, Any]:
        _require_ui_request(ui_header)
        if hybrid_supervisor is not None and _session_active(
            hybrid_supervisor.snapshot().stage,
            terminal={"idle", "completed", "failed", "cancelled"},
        ):
            raise HTTPException(
                status_code=409,
                detail="hybrid Mission owns the machine workflow",
            )
        known_targets = {target_id for target_id, _ in metadata.rl_dig_targets}
        if request.positioning_mode == "rl":
            if request.dig_target_id is None:
                raise HTTPException(status_code=422, detail="RL positioning requires dig_target_id")
            if request.dig_target_id not in known_targets:
                raise HTTPException(status_code=422, detail="unknown RL dig_target_id")
        elif request.dig_target_id is not None:
            raise HTTPException(
                status_code=422,
                detail="dig_target_id is only valid for RL positioning",
            )
        return _operator_action(
            lambda: supervisor.start(request.positioning_mode, request.dig_target_id),
            supervisor,
        )

    @app.post("/api/collection/manual-complete")
    def complete_manual_positioning(
        ui_header: str | None = Header(default=None, alias="X-Excavator-UI"),
    ) -> dict[str, Any]:
        _require_ui_request(ui_header)
        return _operator_action(supervisor.complete_manual_positioning, supervisor)

    @app.post("/api/collection/outcome")
    def submit_outcome(
        request: EpisodeOutcomeRequest,
        ui_header: str | None = Header(default=None, alias="X-Excavator-UI"),
    ) -> dict[str, Any]:
        _require_ui_request(ui_header)
        return _operator_action(
            lambda: supervisor.submit_outcome(request.outcome), supervisor
        )

    @app.post("/api/collection/stop")
    def stop_collection(
        ui_header: str | None = Header(default=None, alias="X-Excavator-UI"),
    ) -> dict[str, Any]:
        _require_ui_request(ui_header)
        return _operator_action(supervisor.stop, supervisor)

    @app.get("/api/hybrid/status")
    def hybrid_status() -> dict[str, Any]:
        if hybrid_supervisor is None:
            raise HTTPException(status_code=503, detail="hybrid Mission is disabled")
        return asdict(hybrid_supervisor.snapshot())

    @app.get("/api/operator/status")
    def operator_status() -> dict[str, object]:
        if operator_supervisor is None:
            raise HTTPException(status_code=404, detail="operator control is disabled")
        return asdict(operator_supervisor.snapshot())

    @app.post("/api/operator/start")
    def start_operator(
        x_excavator_ui: str | None = Header(default=None),
    ) -> dict[str, object]:
        _require_ui_request(x_excavator_ui)
        if operator_supervisor is None:
            raise HTTPException(status_code=404, detail="operator control is disabled")
        return _operator_action(operator_supervisor.start, operator_supervisor)

    @app.post("/api/operator/stop")
    def stop_operator(
        x_excavator_ui: str | None = Header(default=None),
    ) -> dict[str, object]:
        _require_ui_request(x_excavator_ui)
        if operator_supervisor is None:
            raise HTTPException(status_code=404, detail="operator control is disabled")
        return _operator_action(operator_supervisor.stop, operator_supervisor)

    @app.post("/api/hybrid/start")
    def start_hybrid_mission(
        request: StartHybridMissionRequest,
        ui_header: str | None = Header(default=None, alias="X-Excavator-UI"),
    ) -> dict[str, Any]:
        _require_ui_request(ui_header)
        if hybrid_supervisor is None:
            raise HTTPException(status_code=503, detail="hybrid Mission is disabled")
        known_targets = {target_id for target_id, _ in metadata.rl_dig_targets}
        if request.dig_target_id not in known_targets:
            raise HTTPException(status_code=422, detail="unknown RL dig_target_id")
        if _session_active(
            supervisor.snapshot().stage,
            terminal={"idle", "completed", "failed", "cancelled"},
        ):
            raise HTTPException(
                status_code=409,
                detail="guided collection owns the machine workflow",
            )
        if operator_supervisor is not None:
            operator_snapshot = operator_supervisor.snapshot()
            if operator_snapshot.stage != "ready":
                _operator_action(operator_supervisor.start, operator_supervisor)
                operator_snapshot = operator_supervisor.snapshot()
            if operator_snapshot.stage != "ready":
                raise HTTPException(
                    status_code=409,
                    detail="RL/RViz base service did not become ready",
                )
        return _operator_action(
            lambda: hybrid_supervisor.start(
                request.dig_target_id,
                automatic=request.automatic,
                motion_authorization=request.motion_authorization,
                cycle_count=request.cycle_count,
            ),
            hybrid_supervisor,
        )

    @app.post("/api/hybrid/advance")
    def advance_hybrid_mission(
        request: AdvanceHybridMissionRequest,
        ui_header: str | None = Header(default=None, alias="X-Excavator-UI"),
    ) -> dict[str, Any]:
        _require_ui_request(ui_header)
        if hybrid_supervisor is None:
            raise HTTPException(status_code=503, detail="hybrid Mission is disabled")
        return _operator_action(
            lambda: hybrid_supervisor.advance(
                motion_authorization=request.motion_authorization
            ),
            hybrid_supervisor,
        )

    @app.post("/api/hybrid/stop")
    def stop_hybrid_mission(
        ui_header: str | None = Header(default=None, alias="X-Excavator-UI"),
    ) -> dict[str, Any]:
        _require_ui_request(ui_header)
        if hybrid_supervisor is None:
            raise HTTPException(status_code=503, detail="hybrid Mission is disabled")
        return _operator_action(hybrid_supervisor.stop, hybrid_supervisor)

    return app


def _operator_action(
    action: Any, supervisor: Any
) -> dict[str, Any]:
    try:
        action()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(supervisor.snapshot())


def _session_active(stage: str, *, terminal: set[str]) -> bool:
    return stage not in terminal

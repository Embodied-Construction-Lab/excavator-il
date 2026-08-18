"""Local FastAPI adapter for guided Demonstration Episode collection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .collection_ui_config import CollectionUiConfig
from .collection_ui_session import CollectionSessionSnapshot


@dataclass(frozen=True)
class CollectionUiMetadata:
    operator_id: str
    task: str
    dig_target_m: tuple[float, float, float]
    orin_host: str


class CollectionSupervisor(Protocol):
    def snapshot(self) -> CollectionSessionSnapshot: ...

    def start(self, positioning_mode: str) -> None: ...

    def complete_manual_positioning(self) -> None: ...

    def submit_outcome(self, outcome: str) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class StartCollectionRequest(BaseModel):
    positioning_mode: Literal["rl", "manual", "direct"]


class EpisodeOutcomeRequest(BaseModel):
    outcome: Literal["success", "failure", "retake"]


_STATIC_DIR = Path(__file__).with_name("collection_ui_static")


def _require_ui_request(ui_header: str | None) -> None:
    if ui_header != "1":
        raise HTTPException(status_code=403, detail="local collection UI header required")


def create_collection_ui_app(
    *,
    config: CollectionUiConfig,
    metadata: CollectionUiMetadata,
    supervisor: CollectionSupervisor,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        supervisor.close()

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
            "camera_preview_url": config.camera_preview_url,
            "visualization_url": config.visualization_url,
            "positioning_modes": ["rl", "manual", "direct"],
        }

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return asdict(supervisor.snapshot())

    @app.post("/api/collection/start")
    def start_collection(
        request: StartCollectionRequest,
        ui_header: str | None = Header(default=None, alias="X-Excavator-UI"),
    ) -> dict[str, Any]:
        _require_ui_request(ui_header)
        return _operator_action(
            lambda: supervisor.start(request.positioning_mode), supervisor
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

    return app


def _operator_action(
    action: Any, supervisor: CollectionSupervisor
) -> dict[str, Any]:
    try:
        action()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(supervisor.snapshot())

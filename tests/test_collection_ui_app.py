from fastapi.testclient import TestClient

from excavator_il.collection_ui_app import (
    CollectionUiMetadata,
    create_collection_ui_app,
)
from excavator_il.collection_ui_config import CollectionUiConfig
from excavator_il.collection_ui_session import CollectionSessionSnapshot


class _Supervisor:
    def __init__(self):
        self.state = CollectionSessionSnapshot()
        self.calls = []

    def snapshot(self):
        return self.state

    def start(self, mode):
        self.calls.append(("start", mode))
        self.state = CollectionSessionSnapshot(
            stage="starting", positioning_mode=mode
        )

    def complete_manual_positioning(self):
        self.calls.append(("complete_manual_positioning",))

    def submit_outcome(self, outcome):
        self.calls.append(("submit_outcome", outcome))

    def stop(self):
        self.calls.append(("stop",))

    def close(self):
        self.calls.append(("close",))


def test_collection_ui_exposes_config_status_and_guided_collection_actions(tmp_path):
    supervisor = _Supervisor()
    app = create_collection_ui_app(
        config=CollectionUiConfig(
            guided_config=tmp_path / "guided.json",
            host="127.0.0.1",
            port=8088,
            camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
            visualization_url="",
        ),
        metadata=CollectionUiMetadata(
            operator_id="zhaoshuai",
            task="ExecuteDig",
            dig_target_m=(1.0, 0.0, 0.0),
            orin_host="192.168.50.2",
        ),
        supervisor=supervisor,
    )

    with TestClient(app) as client:
        page = client.get("/")
        stylesheet = client.get("/static/app.css")
        script = client.get("/static/app.js")
        config = client.get("/api/config").json()
        idle = client.get("/api/status").json()
        started = client.post(
            "/api/collection/start",
            json={"positioning_mode": "manual"},
            headers={"X-Excavator-UI": "1"},
        )
        completed = client.post(
            "/api/collection/manual-complete",
            headers={"X-Excavator-UI": "1"},
        )
        outcome = client.post(
            "/api/collection/outcome",
            json={"outcome": "success"},
            headers={"X-Excavator-UI": "1"},
        )

    assert page.status_code == 200
    assert 'data-app="excavator-collection-ui"' in page.text
    assert "选择定位方式" in page.text
    assert stylesheet.status_code == 200
    assert "collection-grid" in stylesheet.text
    assert script.status_code == 200
    assert "X-Excavator-UI" in script.text
    assert config["operator_id"] == "zhaoshuai"
    assert config["dig_target_m"] == [1.0, 0.0, 0.0]
    assert config["camera_preview_url"].endswith("/camera/front.mjpg")
    assert idle["stage"] == "idle"
    assert started.status_code == 200
    assert completed.status_code == 200
    assert outcome.status_code == 200
    assert supervisor.calls[:3] == [
        ("start", "manual"),
        ("complete_manual_positioning",),
        ("submit_outcome", "success"),
    ]
    assert supervisor.calls[-1] == ("close",)


def test_collection_ui_rejects_state_change_without_local_ui_header(tmp_path):
    supervisor = _Supervisor()
    app = create_collection_ui_app(
        config=CollectionUiConfig(
            guided_config=tmp_path / "guided.json",
            host="127.0.0.1",
            port=8088,
            camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
            visualization_url="",
        ),
        metadata=CollectionUiMetadata(
            operator_id="zhaoshuai",
            task="ExecuteDig",
            dig_target_m=(0.8, 0.0, -0.2),
            orin_host="192.168.50.2",
        ),
        supervisor=supervisor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/collection/start", json={"positioning_mode": "direct"}
        )

    assert response.status_code == 403
    assert supervisor.calls == [("close",)]

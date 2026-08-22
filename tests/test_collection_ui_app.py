import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from excavator_il.airy_operator import AiryOperatorSnapshot
from excavator_il.collection_ui_app import (
    CollectionUiMetadata,
    create_collection_ui_app,
)
from excavator_il.collection_ui_config import CollectionUiConfig
from excavator_il.collection_ui_session import CollectionSessionSnapshot
from excavator_il.hybrid_mission_session import HybridMissionSnapshot
from excavator_il import collection_ui_app


class _Supervisor:
    def __init__(self):
        self.state = CollectionSessionSnapshot()
        self.calls = []

    def snapshot(self):
        return self.state

    def start(self, mode, dig_target_id=None):
        self.calls.append(("start", mode, dig_target_id))
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


class _HybridSupervisor:
    def __init__(self):
        self.state = HybridMissionSnapshot()
        self.calls = []

    def snapshot(self):
        return self.state

    def start(self, target_id, *, automatic, motion_authorization, cycle_count=1):
        self.calls.append(
            ("start", target_id, automatic, motion_authorization, cycle_count)
        )
        self.state = HybridMissionSnapshot(
            stage="starting",
            dig_target_id=target_id,
            automatic=automatic,
            next_segment="rl_to_dig",
        )

    def advance(self, *, motion_authorization):
        self.calls.append(("advance", motion_authorization))

    def stop(self):
        self.calls.append(("stop",))

    def close(self):
        self.calls.append(("close",))


class _OperatorSupervisor:
    def __init__(self):
        self.state = AiryOperatorSnapshot()
        self.calls = []

    def snapshot(self):
        return self.state

    def start(self):
        self.calls.append(("start",))
        self.state = AiryOperatorSnapshot(stage="ready")

    def stop(self):
        self.calls.append(("stop",))
        self.state = AiryOperatorSnapshot(stage="stopped")

    def close(self):
        self.calls.append(("close",))


def test_collection_ui_can_start_and_stop_airy_operator(tmp_path):
    operator = _OperatorSupervisor()
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
            rl_dig_targets=(),
        ),
        supervisor=_Supervisor(),
        operator_supervisor=operator,
    )

    with TestClient(app) as client:
        config = client.get("/api/config").json()
        started = client.post(
            "/api/operator/start", headers={"X-Excavator-UI": "1"}
        )
        stopped = client.post(
            "/api/operator/stop", headers={"X-Excavator-UI": "1"}
        )

    assert config["operator_control_enabled"] is True
    assert started.json()["stage"] == "ready"
    assert stopped.json()["stage"] == "stopped"
    assert operator.calls == [("start",), ("stop",), ("close",)]


def test_hybrid_start_automatically_starts_airy_operator_when_stopped(tmp_path):
    collection = _Supervisor()
    hybrid = _HybridSupervisor()
    operator = _OperatorSupervisor()
    app = create_collection_ui_app(
        config=CollectionUiConfig(
            guided_config=tmp_path / "guided.json",
            hybrid_mission_config=tmp_path / "hybrid.json",
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
            rl_dig_targets=(("dig_01", (1.0, 0.2, 0.0)),),
            hybrid_act_max_steps=130,
        ),
        supervisor=collection,
        hybrid_supervisor=hybrid,
        operator_supervisor=operator,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/hybrid/start",
            json={
                "dig_target_id": "dig_01",
                "automatic": False,
                "cycle_count": 1,
                "motion_authorization": None,
            },
            headers={"X-Excavator-UI": "1"},
        )

    assert response.status_code == 200
    assert operator.calls[0] == ("start",)
    assert hybrid.calls[0] == ("start", "dig_01", False, None, 1)


def test_collection_ui_exposes_segmented_and_automatic_hybrid_mission_actions(
    tmp_path,
):
    collection = _Supervisor()
    hybrid = _HybridSupervisor()
    app = create_collection_ui_app(
        config=CollectionUiConfig(
            guided_config=tmp_path / "guided.json",
            hybrid_mission_config=tmp_path / "hybrid.json",
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
            rl_dig_targets=(("dig_01", (1.0, 0.2, 0.0)),),
            hybrid_act_max_steps=130,
        ),
        supervisor=collection,
        hybrid_supervisor=hybrid,
    )

    with TestClient(app) as client:
        config = client.get("/api/config").json()
        idle = client.get("/api/hybrid/status").json()
        started = client.post(
            "/api/hybrid/start",
            json={
                "dig_target_id": "dig_01",
                "automatic": False,
                "motion_authorization": None,
            },
            headers={"X-Excavator-UI": "1"},
        )
        advanced = client.post(
            "/api/hybrid/advance",
            json={"motion_authorization": "ALLOW_HYBRID_MACHINE_MOTION"},
            headers={"X-Excavator-UI": "1"},
        )
        stopped = client.post(
            "/api/hybrid/stop",
            headers={"X-Excavator-UI": "1"},
        )

    assert config["hybrid_mission_enabled"] is True
    assert config["hybrid_act_max_steps"] == 130
    assert idle["stage"] == "idle"
    assert started.status_code == 200
    assert advanced.status_code == 200
    assert stopped.status_code == 200
    assert hybrid.calls[:3] == [
        ("start", "dig_01", False, None, 1),
        ("advance", "ALLOW_HYBRID_MACHINE_MOTION"),
        ("stop",),
    ]
    assert hybrid.calls[-1] == ("close",)


def test_collection_ui_starts_nine_cycle_truck_loading_mission(tmp_path):
    collection = _Supervisor()
    hybrid = _HybridSupervisor()
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
            rl_dig_targets=(("dig_01", (1.0, 0.2, 0.0)),),
            hybrid_act_max_steps=130,
        ),
        supervisor=collection,
        hybrid_supervisor=hybrid,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/hybrid/start",
            json={
                "dig_target_id": "dig_01",
                "automatic": True,
                "cycle_count": 9,
                "motion_authorization": "ALLOW_HYBRID_MACHINE_MOTION",
            },
            headers={"X-Excavator-UI": "1"},
        )

    assert response.status_code == 200
    assert hybrid.calls[0] == (
        "start",
        "dig_01",
        True,
        "ALLOW_HYBRID_MACHINE_MOTION",
        9,
    )


def test_collection_and_hybrid_workflows_are_mutually_exclusive(tmp_path):
    collection = _Supervisor()
    hybrid = _HybridSupervisor()
    hybrid.state = HybridMissionSnapshot(
        stage="awaiting_act_dig",
        dig_target_id="dig_01",
        next_segment="act_dig",
    )
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
            rl_dig_targets=(("dig_01", (1.0, 0.2, 0.0)),),
        ),
        supervisor=collection,
        hybrid_supervisor=hybrid,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/collection/start",
            json={"positioning_mode": "direct"},
            headers={"X-Excavator-UI": "1"},
        )

    assert response.status_code == 409
    assert collection.calls == [("close",)]


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
            rl_dig_targets=(
                ("dig_01", (1.0, 0.2, 0.0)),
                ("dig_02", (1.0, 0.0, 0.0)),
            ),
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
    assert "选择工作模式" in page.text
    assert "遥操作" in page.text
    assert "三维可视化" not in page.text
    assert "RViz / Foxglove 扩展位" not in page.text
    assert "连续自动完成 1～9 铲装车循环" in page.text
    assert '<option value="9">9 铲</option>' in page.text
    assert '/static/app.js?v=20260821-nine-cycle' in page.text
    assert stylesheet.status_code == 200
    assert "collection-grid" in stylesheet.text
    assert script.status_code == 200
    assert "X-Excavator-UI" in script.text
    assert config["operator_id"] == "zhaoshuai"
    assert config["dig_target_m"] == [1.0, 0.0, 0.0]
    assert config["camera_preview_url"] == "/api/camera/frame.jpg"
    assert config["rl_dig_targets"][0] == {
        "target_id": "dig_01",
        "position_m": [1.0, 0.2, 0.0],
    }
    assert idle["stage"] == "idle"
    assert started.status_code == 200
    assert completed.status_code == 200
    assert outcome.status_code == 200
    assert supervisor.calls[:3] == [
        ("start", "manual", None),
        ("complete_manual_positioning",),
        ("submit_outcome", "success"),
    ]
    assert supervisor.calls[-1] == ("close",)


def test_collection_ui_starts_standalone_teleop_without_dig_target(tmp_path):
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
            rl_dig_targets=(),
        ),
        supervisor=supervisor,
    )

    with TestClient(app) as client:
        config = client.get("/api/config").json()
        response = client.post(
            "/api/collection/start",
            json={"positioning_mode": "teleop"},
            headers={"X-Excavator-UI": "1"},
        )

    assert "teleop" in config["positioning_modes"]
    assert response.status_code == 200
    assert supervisor.calls[0] == ("start", "teleop", None)


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
            rl_dig_targets=(),
        ),
        supervisor=supervisor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/collection/start", json={"positioning_mode": "direct"}
        )

    assert response.status_code == 403
    assert supervisor.calls == [("close",)]


def test_collection_ui_requires_a_known_target_for_rl_positioning(tmp_path):
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
            rl_dig_targets=(("dig_01", (1.0, 0.2, 0.0)),),
        ),
        supervisor=supervisor,
    )

    with TestClient(app) as client:
        missing = client.post(
            "/api/collection/start",
            json={"positioning_mode": "rl"},
            headers={"X-Excavator-UI": "1"},
        )
        unknown = client.post(
            "/api/collection/start",
            json={"positioning_mode": "rl", "dig_target_id": "dig_99"},
            headers={"X-Excavator-UI": "1"},
        )
        accepted = client.post(
            "/api/collection/start",
            json={"positioning_mode": "rl", "dig_target_id": "dig_01"},
            headers={"X-Excavator-UI": "1"},
        )

    assert missing.status_code == 422
    assert unknown.status_code == 422
    assert accepted.status_code == 200
    assert supervisor.calls[0] == ("start", "rl", "dig_01")


def test_collection_ui_proxies_collector_telemetry(monkeypatch, tmp_path):
    supervisor = _Supervisor()
    expected = {
        "age_ms": 12.5,
        "joint_angles_deg": {"boom": 1.0, "arm": 2.0, "bucket": 3.0, "swing": 4.0},
        "cylinders_mm": {"boom": 101.0, "stick": 202.0, "bucket": 303.0},
    }
    monkeypatch.setattr(
        collection_ui_app,
        "_fetch_collector_telemetry",
        lambda url: expected if url.endswith("latest.json") else None,
    )
    app = create_collection_ui_app(
        config=CollectionUiConfig(
            guided_config=tmp_path / "guided.json",
            host="127.0.0.1",
            port=8088,
            camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
            visualization_url="",
            telemetry_url="http://192.168.50.2:18092/telemetry/latest.json",
        ),
        metadata=CollectionUiMetadata(
            operator_id="zhaoshuai",
            task="ExecuteDig",
            dig_target_m=(1.0, 0.0, 0.0),
            orin_host="192.168.50.2",
            rl_dig_targets=(),
        ),
        supervisor=supervisor,
    )

    with TestClient(app) as client:
        response = client.get("/api/telemetry")

    assert response.status_code == 200
    assert response.json() == expected


def test_collection_ui_proxies_one_collector_camera_snapshot(monkeypatch, tmp_path):
    supervisor = _Supervisor()
    requested_urls = []

    def fetch(url):
        requested_urls.append(url)
        return b"\xff\xd8fixture-jpeg\xff\xd9"

    monkeypatch.setattr(collection_ui_app, "_fetch_collector_camera", fetch)
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
            rl_dig_targets=(),
        ),
        supervisor=supervisor,
    )

    with TestClient(app) as client:
        config = client.get("/api/config").json()
        response = client.get("/api/camera/frame.jpg")

    assert config["camera_preview_url"] == "/api/camera/frame.jpg"
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == b"\xff\xd8fixture-jpeg\xff\xd9"
    assert requested_urls == ["http://192.168.50.2:18092/camera/front.jpg"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_collection_ui_camera_recovers_when_collector_starts_later():
    test_script = (
        Path(__file__).parent
        / "js"
        / "collection_ui_camera_reconnect.test.cjs"
    )

    subprocess.run(["node", str(test_script)], check=True)

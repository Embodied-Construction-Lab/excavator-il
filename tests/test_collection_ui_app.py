import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from excavator_il.airy_operator import AiryOperatorSnapshot
from excavator_il.collection_ui_app import (
    CollectionUiMetadata,
    HybridDigGroupMetadata,
    StartCollectionRequest,
    create_collection_ui_app,
)
from excavator_il.collection_ui_config import CollectionUiConfig
from excavator_il.collection_ui_session import CollectionSessionSnapshot
from excavator_il.hybrid_mission_session import HybridMissionSnapshot
from excavator_il import collection_ui_app


ROOT = Path(__file__).resolve().parents[1]


def test_dual_camera_views_are_stacked_vertically_without_stretching_page():
    index = (
        ROOT / "src" / "excavator_il" / "collection_ui_static" / "index.html"
    ).read_text(encoding="utf-8")
    stylesheet = (
        ROOT / "src" / "excavator_il" / "collection_ui_static" / "app.css"
    ).read_text(encoding="utf-8")

    assert '/static/app.css?v=20260829-point-catalog' in index
    assert ".camera-grid { display: grid; grid-template-columns: 1fr;" in stylesheet
    assert "align-items: start;" in stylesheet.split(".collection-grid", 1)[1].split(
        "}", 1
    )[0]
    assert ".camera-frame { position: relative; aspect-ratio: 16 / 9;" in stylesheet
    assert (
        ".camera-frame img { display: none; width: 100%; height: 100%; "
        "object-fit: contain;"
    ) in stylesheet
    assert (
        ".camera-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));"
        not in stylesheet
    )


def test_collection_ui_omits_manual_episode_labels_and_campaign_progress():
    index = (
        ROOT / "src" / "excavator_il" / "collection_ui_static" / "index.html"
    ).read_text(encoding="utf-8")

    assert "本条 Episode 标记" not in index
    assert 'id="task-variant"' not in index
    assert 'id="soil-reset-block-id"' not in index
    assert 'id="dig-point-id"' not in index
    assert 'id="collection-zone-id"' not in index
    assert 'id="dig-repeat-index"' not in index
    assert 'id="operator-note"' not in index
    assert 'id="episode-context"' not in index
    assert "200 条采集进度" not in index
    assert "权威下一条" not in index
    assert "slot_" not in index
    assert "本次 UI 已完成" not in index
    assert 'id="completed-count"' not in index


def test_collection_ui_has_no_runtime_campaign_api_or_configuration(tmp_path):
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
    )

    with TestClient(app) as client:
        config = client.get("/api/config").json()
        campaign = client.get("/api/campaign/status")

    assert "campaign_tracking_enabled" not in config
    assert campaign.status_code == 404


class _Supervisor:
    def __init__(self):
        self.state = CollectionSessionSnapshot()
        self.calls = []

    def snapshot(self):
        return self.state

    def start(
        self,
        mode,
        dig_target_id=None,
        *,
        task_variant=None,
        soil_reset_block_id=None,
        dig_point_id=None,
        collection_zone_id=None,
        dig_repeat_index=None,
        operator_note=None,
    ):
        call = (
            "start",
            mode,
            dig_target_id,
            task_variant,
            soil_reset_block_id,
            dig_point_id,
        )
        if collection_zone_id is not None:
            call = (*call, collection_zone_id, dig_repeat_index, operator_note)
        self.calls.append(call)
        self.state = CollectionSessionSnapshot(
            stage="starting",
            positioning_mode=mode,
            task_variant=task_variant or "",
            soil_reset_block_id=soil_reset_block_id or "",
            dig_point_id=dig_point_id or "",
            collection_zone_id=collection_zone_id or "",
            dig_repeat_index=dig_repeat_index or 0,
            operator_note=operator_note or "",
        )

    def complete_manual_positioning(self):
        self.calls.append(("complete_manual_positioning",))

    def submit_outcome(self, outcome):
        self.calls.append(("submit_outcome", outcome))

    def stop(self):
        self.calls.append(("stop",))

    def clear_logs(self):
        self.calls.append(("clear_logs",))
        self.state = CollectionSessionSnapshot(stage=self.state.stage)

    def close(self):
        self.calls.append(("close",))


class _HybridSupervisor:
    def __init__(self):
        self.state = HybridMissionSnapshot()
        self.calls = []

    def snapshot(self):
        return self.state

    def start(
        self,
        target_id,
        *,
        automatic,
        motion_authorization,
        cycle_count=1,
        dig_group_id="all",
    ):
        self.calls.append(
            (
                "start",
                target_id,
                automatic,
                motion_authorization,
                cycle_count,
                dig_group_id,
            )
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

    def clear_logs(self):
        self.calls.append(("clear_logs",))
        self.state = HybridMissionSnapshot(stage=self.state.stage)

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


def _route_endpoint(app, path: str):
    return next(route.endpoint for route in app.routes if route.path == path)


def test_collection_ui_request_contract_rejects_removed_episode_labels(
    tmp_path,
):
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
    with pytest.raises(ValueError, match="task_variant"):
        StartCollectionRequest(
            positioning_mode="direct",
            task_variant="dig_only",
        )
    assert supervisor.calls == []


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


def test_collection_ui_can_clear_collection_and_hybrid_logs(tmp_path):
    collection = _Supervisor()
    collection.state = CollectionSessionSnapshot(logs=("collector line",))
    hybrid = _HybridSupervisor()
    hybrid.state = HybridMissionSnapshot(logs=("mission line",))
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
        ),
        supervisor=collection,
        hybrid_supervisor=hybrid,
    )

    with TestClient(app) as client:
        cleared_collection = client.post(
            "/api/collection/logs/clear",
            headers={"X-Excavator-UI": "1"},
        )
        cleared_hybrid = client.post(
            "/api/hybrid/logs/clear",
            headers={"X-Excavator-UI": "1"},
        )

    assert cleared_collection.status_code == 200
    assert cleared_collection.json()["logs"] == []
    assert cleared_hybrid.status_code == 200
    assert cleared_hybrid.json()["logs"] == []
    assert ("clear_logs",) in collection.calls
    assert ("clear_logs",) in hybrid.calls


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
    assert hybrid.calls[0] == ("start", "dig_01", False, None, 1, "all")


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
        ("start", "dig_01", False, None, 1, "all"),
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
        "all",
    )


def test_collection_ui_exposes_and_starts_selected_dig_group(tmp_path):
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
            rl_dig_targets=(
                ("near_01", (1.0, 0.4, 0.0)),
                ("near_02", (1.0, 0.15, 0.0)),
                ("far_01", (1.3, 0.4, 0.0)),
            ),
            hybrid_act_max_steps=130,
            hybrid_dig_groups=(
                HybridDigGroupMetadata(
                    group_id="all",
                    label="全部 3 点",
                    point_ids=("near_01", "near_02", "far_01"),
                ),
                HybridDigGroupMetadata(
                    group_id="near",
                    label="近端 2 点",
                    point_ids=("near_01", "near_02"),
                ),
                HybridDigGroupMetadata(
                    group_id="far",
                    label="远端 1 点",
                    point_ids=("far_01",),
                ),
            ),
            hybrid_default_dig_group_id="all",
        ),
        supervisor=collection,
        hybrid_supervisor=hybrid,
    )

    with TestClient(app) as client:
        config = client.get("/api/config").json()
        response = client.post(
            "/api/hybrid/start",
            json={
                "dig_target_id": "near_02",
                "dig_group_id": "near",
                "automatic": True,
                "cycle_count": 4,
                "motion_authorization": "ALLOW_HYBRID_MACHINE_MOTION",
            },
            headers={"X-Excavator-UI": "1"},
        )

    assert config["hybrid_default_dig_group_id"] == "all"
    assert config["hybrid_dig_groups"][1] == {
        "group_id": "near",
        "label": "近端 2 点",
        "point_ids": ["near_01", "near_02"],
    }
    assert response.status_code == 200
    assert hybrid.calls[0] == (
        "start",
        "near_02",
        True,
        "ALLOW_HYBRID_MACHINE_MOTION",
        4,
        "near",
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
    assert 'id="hybrid-dig-group"' in page.text
    assert 'id="hybrid-dig-target"' in page.text
    assert "起始挖掘点" in page.text
    assert '/static/app.js?v=20260902-unified-state' in page.text
    assert 'id="copy-log"' in page.text
    assert 'id="copy-hybrid-log"' in page.text
    assert 'id="clear-log"' in page.text
    assert 'id="clear-hybrid-log"' in page.text
    assert 'command("/api/collection/logs/clear")' in script.text
    assert 'commandHybrid("/api/hybrid/logs/clear")' in script.text
    assert "dig_group_id: selectedHybridGroupId()" in script.text
    assert '$("hybrid-dig-target").disabled = collectionActive || hybridActive' in script.text
    assert "state.hybridSnapshot?.can_stop === true" in script.text
    assert "本条 Episode 标记" not in page.text
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
        ("start", "manual", None, None, None, None),
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
    assert supervisor.calls[0] == (
        "start",
        "teleop",
        None,
        None,
        None,
        None,
    )


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
            json={
                "positioning_mode": "rl",
                "dig_target_id": "dig_01",
            },
            headers={"X-Excavator-UI": "1"},
        )

    assert missing.status_code == 422
    assert unknown.status_code == 422
    assert accepted.status_code == 200
    assert supervisor.calls[0] == (
        "start",
        "rl",
        "dig_01",
        None,
        None,
        None,
    )


def test_collection_ui_accepts_collection_without_episode_labels(tmp_path):
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
            json={"positioning_mode": "direct"},
            headers={"X-Excavator-UI": "1"},
        )
    assert missing.status_code == 200
    assert supervisor.calls[0] == ("start", "direct", None, None, None, None)


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


def test_collection_ui_uses_lifecycle_owned_machine_state_telemetry(tmp_path):
    supervisor = _Supervisor()

    class _TelemetrySource:
        def __init__(self):
            self.started = False
            self.closed = False

        def start(self):
            self.started = True

        def snapshot(self):
            assert self.started
            return {
                "source": "machine_state_v1/udp:18081",
                "seq": 18,
                "age_ms": 20.0,
                "joint_angles_deg": {},
                "cylinders_mm": {},
            }

        def close(self):
            self.closed = True

    source = _TelemetrySource()
    app = create_collection_ui_app(
        config=CollectionUiConfig(
            guided_config=tmp_path / "guided.json",
            host="127.0.0.1",
            port=8088,
            camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
            visualization_url="",
            telemetry_url="http://invalid.example/legacy.json",
        ),
        metadata=CollectionUiMetadata(
            operator_id="zhaoshuai",
            task="ExecuteDig",
            dig_target_m=(1.0, 0.0, 0.0),
            orin_host="192.168.50.2",
            rl_dig_targets=(),
        ),
        supervisor=supervisor,
        telemetry_source=source,
    )

    with TestClient(app) as client:
        response = client.get("/api/telemetry")
        assert source.started is True

    assert response.status_code == 200
    assert response.json()["source"] == "machine_state_v1/udp:18081"
    assert source.closed is True


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


def test_collection_ui_exposes_independent_front_and_dump_camera_snapshots(
    monkeypatch, tmp_path
):
    requested_urls = []

    def fetch(url):
        requested_urls.append(url)
        label = b"front" if "/front." in url else b"dump"
        return b"\xff\xd8" + label + b"\xff\xd9"

    monkeypatch.setattr(collection_ui_app, "_fetch_collector_camera", fetch)
    app = create_collection_ui_app(
        config=CollectionUiConfig(
            guided_config=tmp_path / "guided.json",
            host="127.0.0.1",
            port=8088,
            camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
            camera_dump_preview_url=(
                "http://192.168.50.2:18092/camera/dump.mjpg"
            ),
            visualization_url="",
        ),
        metadata=CollectionUiMetadata(
            operator_id="zhaoshuai",
            task="ExecuteDigAndDump",
            dig_target_m=(1.0, 0.0, 0.0),
            orin_host="192.168.50.2",
            rl_dig_targets=(),
        ),
        supervisor=_Supervisor(),
    )

    with TestClient(app) as client:
        config = client.get("/api/config").json()
        front = client.get("/api/camera/front.jpg")
        dump = client.get("/api/camera/dump.jpg")
        legacy_front = client.get("/api/camera/frame.jpg")

    assert config["camera_preview_urls"] == {
        "front": "/api/camera/front.jpg",
        "dump": "/api/camera/dump.jpg",
    }
    assert config["camera_preview_url"] == "/api/camera/frame.jpg"
    assert front.content == b"\xff\xd8front\xff\xd9"
    assert dump.content == b"\xff\xd8dump\xff\xd9"
    assert legacy_front.content == front.content
    assert requested_urls == [
        "http://192.168.50.2:18092/camera/front.jpg",
        "http://192.168.50.2:18092/camera/dump.jpg",
        "http://192.168.50.2:18092/camera/front.jpg",
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_collection_ui_camera_recovers_when_collector_starts_later():
    test_script = (
        Path(__file__).parent
        / "js"
        / "collection_ui_camera_reconnect.test.cjs"
    )

    subprocess.run(["node", str(test_script)], check=True)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_collection_ui_drives_two_camera_previews_independently():
    test_script = (
        Path(__file__).parent
        / "js"
        / "collection_ui_dual_camera.test.cjs"
    )

    subprocess.run(["node", str(test_script)], check=True)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_collection_ui_sends_and_renders_required_episode_protocol():
    test_script = (
        Path(__file__).parent
        / "js"
        / "collection_ui_protocol.test.cjs"
    )

    subprocess.run(["node", str(test_script)], check=True)

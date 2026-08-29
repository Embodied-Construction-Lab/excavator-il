import json
from pathlib import Path

import pytest

from excavator_il._resident_fixed_cycle_groups import (
    normalize_dig_groups,
    select_cycle_targets,
)
from excavator_il.dig_point_catalog import load_dig_point_catalog


def test_catalog_exposes_configured_points_and_ordered_groups(tmp_path: Path) -> None:
    document = {
        "schema_version": "excavation_dig_point_catalog.v1",
        "frame_id": "machine_root_ros",
        "dig_points": {
            "dig_near_01": [1.0, 0.4, 0.0],
            "dig_near_02": [1.0, 0.15, 0.0],
            "dig_far_01": [1.3, 0.4, 0.0],
            "dig_far_02": [1.3, 0.15, 0.0],
        },
        "default_dig_group": "all",
        "dig_groups": {
            "all": [
                "dig_near_01",
                "dig_near_02",
                "dig_far_01",
                "dig_far_02",
            ],
            "near": ["dig_near_01", "dig_near_02"],
            "far": ["dig_far_01", "dig_far_02"],
        },
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    catalog = load_dig_point_catalog(path)

    assert catalog.default_group_id == "all"
    assert catalog.groups["near"] == ("dig_near_01", "dig_near_02")
    assert catalog.points["dig_far_02"] == (1.3, 0.15, 0.0)


def _catalog_document():
    return {
        "schema_version": "excavation_dig_point_catalog.v1",
        "frame_id": "machine_root_ros",
        "dig_points": {"p1": [1.0, 0.4, 0.0], "p2": [1.3, 0.4, 0.0]},
        "default_dig_group": "all",
        "dig_groups": {"all": ["p1", "p2"], "near": ["p1"]},
    }


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda value: value.update(schema_version="bad"), "schema"),
        (lambda value: value.update(frame_id="map"), "frame"),
        (lambda value: value.update(dig_points={}), "dig_points"),
        (lambda value: value.update(dig_points={"bad id": [1, 2, 3]}), "id"),
        (lambda value: value.update(dig_points={"p1": [1, 2]}), "three"),
        (lambda value: value.update(dig_points={"p1": [1, "x", 3]}), "numeric"),
        (lambda value: value.update(dig_points={"p1": [1, float("nan"), 3]}), "finite"),
        (lambda value: value.update(dig_groups={}), "dig_groups"),
        (lambda value: value.update(dig_groups={"all": []}), "non-empty"),
        (lambda value: value.update(dig_groups={"all": ["p1", "p1"]}), "unique"),
        (lambda value: value.update(dig_groups={"all": ["p1", "p3"]}), "unknown"),
        (lambda value: value.update(dig_groups={"all": ["p2", "p1"]}), "order"),
        (lambda value: value.update(default_dig_group="missing"), "not defined"),
    ],
)
def test_catalog_rejects_invalid_contract(tmp_path, mutate, expected):
    document = _catalog_document()
    mutate(document)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        load_dig_point_catalog(path)


def test_catalog_rejects_unreadable_json_and_extra_fields(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load"):
        load_dig_point_catalog(path)

    document = _catalog_document()
    document["extra"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_dig_point_catalog(path)


def test_fixed_cycle_group_selection_is_strict_and_wraps():
    targets = ("near_01", "near_02", "far_01")
    groups, default_group = normalize_dig_groups(
        targets,
        {
            "all": targets,
            "near": ("near_01", "near_02"),
            "far": ("far_01",),
        },
        "near",
    )

    assert default_group == "near"
    assert select_cycle_targets(groups, "near", "near_02", 4) == (
        "near_02",
        "near_01",
        "near_02",
        "near_01",
    )
    with pytest.raises(ValueError, match="unknown"):
        select_cycle_targets(groups, "missing", "near_01", 1)
    with pytest.raises(ValueError, match="outside"):
        select_cycle_targets(groups, "near", "far_01", 1)


@pytest.mark.parametrize(
    "targets, groups, default_group, expected",
    [
        ((), None, "all", "non-empty"),
        (("p1", "p1"), None, "all", "unique"),
        (("p1",), {"all": []}, "all", "non-empty"),
        (("p1",), {"all": ("p2",)}, "all", "members"),
        (("p1", "p2"), {"all": ("p2", "p1")}, "all", "exactly"),
        (("p1",), {"all": ("p1",)}, "missing", "not defined"),
    ],
)
def test_fixed_cycle_group_contract_rejects_invalid_inputs(
    targets, groups, default_group, expected
):
    with pytest.raises(ValueError, match=expected):
        normalize_dig_groups(targets, groups, default_group)

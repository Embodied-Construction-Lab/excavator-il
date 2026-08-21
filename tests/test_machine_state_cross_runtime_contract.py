import importlib.util
import math
import sys
from pathlib import Path

import pytest

from excavator_il.collector.machine_state import build_machine_state_packet
from excavator_il.stm32_protocol import (
    STM32_TELEMETRY_FIELDS,
    STM32_TELEMETRY_SCHEMA_VERSION,
    Stm32TelemetryParser,
)


ORIN_RUNTIME = (
    Path(__file__).resolve().parents[2]
    / "excavator-orin-runtime"
    / "orin_state_sender.py"
)


def _load_orin_runtime():
    if not ORIN_RUNTIME.is_file():
        pytest.skip("cross-repository Orin Runtime checkout is unavailable")
    spec = importlib.util.spec_from_file_location(
        "contract_orin_state_sender", ORIN_RUNTIME
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _golden_v2_row() -> str:
    values = {field: "0" for field in STM32_TELEMETRY_FIELDS}
    values.update(
        {
            "schema_version": STM32_TELEMETRY_SCHEMA_VERSION,
            "control_seq": "41",
            "control_stamp_ms": "123456",
            "sensor_seq": "22",
            "sensor_is_new": "1",
            "boom_pos_mm": "160.5",
            "stick_pos_mm": "201.25",
            "bucket_pos_mm": "137.75",
            "boom_vel_mmps": "12.5",
            "stick_vel_mmps": "-3.25",
            "bucket_vel_mmps": "7.0",
            "boom_angle_deg": "65.2",
            "arm_angle_deg": "82.3",
            "bucket_angle_deg": "230.0",
            "swing_angle_deg": "-13.5",
            "swing_vel_degps": "2.75",
            "control_enabled": "1",
            "rs485_ok": "1",
            "dwj_ok": "1",
            "imu_ok": "1",
        }
    )
    return ",".join(values[field] for field in STM32_TELEMETRY_FIELDS)


def _assert_same_contract(left, right, path="packet"):
    assert type(left) is type(right), path
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_same_contract(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert len(left) == len(right), path
        for index, value in enumerate(left):
            _assert_same_contract(value, right[index], f"{path}[{index}]")
    elif isinstance(left, float):
        assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12), path
    else:
        assert left == right, path


def test_collector_and_rl_runtime_emit_the_same_machine_state_contract():
    orin = _load_orin_runtime()
    row = _golden_v2_row()
    frame = Stm32TelemetryParser().parse_line(
        row.encode("ascii"), receive_monotonic_ns=10
    )
    assert frame is not None
    orin_state = orin.parse_stm32_csv_line(row)
    assert orin_state is not None

    fixed_wall_ns = 1_786_800_000_123_000_000
    collector_packet = build_machine_state_packet(
        frame,
        sequence=7,
        machine_id="scale_excavator_v1",
        receive_wall_ns=fixed_wall_ns,
    )
    orin.now_ms = lambda: fixed_wall_ns // 1_000_000
    orin_packet = orin.build_machine_state_packet(
        orin_state,
        seq=7,
        machine_id="scale_excavator_v1",
        control_enabled=True,
        estop=False,
        include_raw=False,
        last_receive_monotonic_s=orin.time.monotonic(),
    )

    _assert_same_contract(collector_packet, orin_packet)

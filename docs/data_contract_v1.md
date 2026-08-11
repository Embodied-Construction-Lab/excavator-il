# RGB 挖掘示教数据契约 v1

本文件是 `excavator-il` 活动采集链的字段、频率、时钟和训练接口契约。历史调查见
`EvaluationReport/2026-08-04_stm32_il_collection_quick_contract.md`，但实现发生冲突时以本文件、
版本化 schema 和对应测试为准。

正式数据不得直接由当前 `deploy_scale_model/doublestick_send.py` 的 `formatted_data` 生成：
该联调发送器目前以 10 Hz 发送四舍五入到两位小数的文本轴值，且没有样本序号和单调时间。
正式采集必须保留 Orin 接收的未舍入四个 XY 原始值，并另存映射后的四维专家 Action。线上协议
继续保留 Z1/Z2 字段以维持 schema 兼容，但人工示教链必须将两者固定为零。

## 在线接口

### PC → Orin：`excavator_joystick.v1`

UDP 数值包固定包含：

- `session_id`、`sample_seq`、PC 单调/墙钟时间；
- 六轴 `X1/Y1/Z1/X2/Y2/Z2`：四个 XY 为未舍入的 `[-1,1]` 数值，Z1/Z2 必须精确为零；
- 两个手柄的 slot、GUID、名称和按钮数组；
- `deadman_pressed`、`mapping_id`、`calibration_id`。

PC 以 20 Hz 发送。Orin 只接受配置的 PC 地址、设备 ID、映射和标定版本，并使用
`(PC地址, session_id, sample_seq)` 检查重复和乱序。PC 的单调时间仅用于来源审计，不能与
Orin/STM32 时钟直接相减。

PC 本地使用带 USB 序列号的 `/dev/input/by-id/*-event-joystick` 稳定路径把物理手柄绑定到
slot；该路径不进入 UDP 包。两个同型号手柄可以具有相同 GUID，但必须配置不同路径，且启动时
同时校验路径、GUID 和 SDL 实例 ID，禁止依赖 SDL 枚举顺序猜测左右。此本地绑定要求 SDL 2.24
或更新版本；`device_path` 必填的本地配置 schema 是 `excavator_teleop_config.v3`，每只手柄只
配置 X/Y 两个原始轴号。PC 构造线上六轴包时插入零 Z1/Z2，不改变线上
`excavator_joystick.v1` 协议。Collector 在 STM32 串口边界再次强制 Z1/Z2 为零；即使异常或旧
发送端提供非零 Z，也不得转发到固件的左右行走输出。

Orin 收包后以自身 `CLOCK_MONOTONIC` 同时生成专家标签：

```text
[boom, stick, bucket, swing] = [Y2, Y1, X2, X1]
```

默认死区为 0.15：绝对值不超过死区时变为零，死区外保留原始幅值。PC 和 Orin 均不取反、
不重新归一化、不缩放。deadman 未按下时标签无效，并向 STM32 发送零命令。

### Orin → STM32：`stm32_manual_command.v1`

换行结尾 JSON 包含 schema、未改变方向的四个 XY、固定为零的 Z1/Z2、`command_seq` 和
`command_source_stamp_ms`。STM32 才负责死区后的阀角、PWM、泵和硬件方向适配。未知 schema、
缺字段、重复/乱序帧不会更新控制目标；超过 300 ms 未收到有效命令时输出中位/零命令。
Collector 每次启动都必须先读取一帧有效 STM32 遥测，以当前 `command_rx_seq` 恢复下一条
`command_seq`，再发送启动零命令；2 秒内无法取得有效遥测时不得进入 ready 状态。这样既保留
STM32 的重复/乱序拒绝，也避免 Collector 或 Orin 重启后从零计数导致合法命令被长时间拒绝。

### STM32 → Orin：`stm32_control_telemetry.v2`

USART2 固定 460800 baud，启动时发送一次 CSV 表头，随后 20 Hz 发送数据行。Collector 即使
启动较晚、错过表头，也会按固定 v2 字段数和 schema 严格解析；损坏行保留在
`stm32_raw.jsonl`，但不进入 `control.csv`。

关键字段包括：

- 控制/传感/命令序号及各自 STM32 时间；
- `sensor_is_new`（每隔一个 20 Hz 控制周期产生一次，即 10 Hz）；
- `command_action_boom/stick/bucket/swing`：硬件方向适配前、死区后的规范人工动作；
- 三缸位置/速度、三关节角、swing 角/速度；
- 参考速度、PID、阀角、PWM、泵；
- 控制模式、Homing、命令有效/超时、使能、急停、限位、传感器与故障状态。

## 原始 Episode

一次完整人工挖掘对应一个目录：

```text
episode_xxxx/
├── episode.json
├── stm32_raw.jsonl
├── joystick_raw.jsonl
├── expert_action.jsonl
├── command_tx.jsonl
├── control.csv
├── steps.csv
├── camera_front/
├── camera_front_timestamps.csv
└── quality_report.json
```

原始 JSONL 保留解析失败、重复/乱序和写失败事实。`command_tx.jsonl` 记录真实串口写结果，不能
把“已生成命令”误当成“已写入 STM32”。相机编码完成后使用 Orin 单调时间打戳，并先原子落盘，
再追加索引。正常完成、失败和中止分别写为 `complete`、`failed`、`aborted`。

## 最终训练接口

每个 10 Hz 样本包含以下内容。

### RGB Observation

| LeRobot 字段 | 形状 | 含义 |
|---|---:|---|
| `observation.images.front` | `[3, H, W]` | 前视 RGB 图像；转换前以原始 PNG 保存 |

相机清单 `camera_front_timestamps.csv` 固定三列：

| 字段 | 含义 |
|---|---|
| `camera_frame_index` | 从 0 连续递增的相机帧序号 |
| `camera_stamp_monotonic_ns` | Orin `CLOCK_MONOTONIC` 时间戳 |
| `image_path` | 相对 episode 根目录的图像路径 |

相机和状态必须落在 Orin 同一单调时钟域。转换器对每个状态选择“不晚于该状态”的最新 RGB，
不使用未来图像；默认要求图像年龄不超过 120 ms。120 ms 是第一版工程阈值，需在相机接入
后根据实际延迟分布复核。

### State Observation

`observation.state` 固定为以下 11 维，顺序不可变化：

```text
[
  boom_pos_m, stick_pos_m, bucket_pos_m,
  boom_vel_mps, stick_vel_mps, bucket_vel_mps,
  boom_angle_rad, arm_angle_rad, bucket_angle_rad,
  swing_angle_rad, swing_vel_radps
]
```

### Expert Action

`action` 固定为以下 4 维，范围 `[-1, 1]`，顺序不可变化：

```text
[action_boom, action_stick, action_bucket, action_swing]
```

它表示人工手柄经过轴映射和死区处理后形成的归一化目标速度。PC 与 Orin 不改变其正负方向；
STM32 负责真机方向和底层单位适配。PID 输出、阀角和实际测得速度不能冒充专家动作。

## `steps.csv`

固定 24 列：

```text
episode_id,frame_index,state_seq,state_stamp_ms,
state_receive_monotonic_ns,action_stamp_monotonic_ns,
boom_pos_m,stick_pos_m,bucket_pos_m,
boom_vel_mps,stick_vel_mps,bucket_vel_mps,
boom_angle_rad,arm_angle_rad,bucket_angle_rad,
swing_angle_rad,swing_vel_radps,
action_boom,action_stick,action_bucket,action_swing,
pump_percent,sensor_valid,control_mode
```

第一版训练 episode 中 `sensor_valid` 必须为 `1`，`control_mode` 必须为
`manual_joystick`。无效帧仍应留在 Orin 原始记录中，但不得写进供 ACT 转换的 `steps.csv`。

## `episode.json`

最低要求：

```json
{
  "schema_version": "excavator_demo_raw.v1",
  "episode_id": "episode_0001",
  "task": "ExecuteDig",
  "operator_id": "operator_01",
  "dig_target_m": [0.8, 0.1, -0.2],
  "material_id": "dry_soil_01",
  "success": true,
  "intervention": false,
  "firmware_commit": "...",
  "urdf_hash": "...",
  "machine_profile_hash": "...",
  "valve_calibration_id": "...",
  "pump_setting": "fixed_30_percent",
  "camera_front": {
    "device_id": "camera-serial-or-device-path",
    "width": 640,
    "height": 480,
    "nominal_fps": 30,
    "pixel_format": "RGB8",
    "timestamp_clock": "CLOCK_MONOTONIC"
  }
}
```

### 当前阀控配置基线

`valve_calibration_id=data_celect_manual_open_loop_v1` 对应
`F407/data_celect` 固件 `dda4a403aacb4ef770cd02898efcc42e03ae71b8` 中的以下开环配置：

- 动作死区 `0.15`，映射保持 `[boom, stick, bucket, swing]=[Y2,Y1,X2,X1]`，四轴方向系数均为 `+1`；
- bucket：中位 `90 deg`，正向从 `105` 到 `110 deg`，负向从 `76` 到 `71 deg`；
- boom：中位 `90 deg`，正向从 `110` 到 `115 deg`，负向从 `78` 到 `73 deg`；
- stick：中位 `90 deg`，正向从 `105` 到 `110 deg`，负向从 `75` 到 `69 deg`；
- swing 最大绝对值 `20%`；任一直线轴越过死区时泵为 `-30%`，否则为 `0%`；
- PCA9685 目标 `50 Hz`、振荡器参数 `26,540,000 Hz`；角度通道脉宽范围
  `500..2500 us`，速度通道中位 `1500 us`、范围 `500..2500 us`；
- 本固件为人工开环控制，没有物理速度 PID。

该 ID 只标识可由源码复现的阀/PWM 配置，不表示已经完成“归一化动作到实际速度、压力或流量”的
现场物理响应标定。ACT 标签仍是死区后的归一化人工动作。上述任一参数变化时必须分配新的
`valve_calibration_id`，不能继续使用 `v1`。

相机内参和外参使用独立的、带版本号的标定文件保存，并在正式采集前给
`episode.json` 增加对应 hash；当前尚未拿到相机型号与标定结果，因此 v1 验证器暂不强制该项。

## 时间频率

- STM32 控制与原始 `control.csv`：20 Hz；
- 有效本体状态与 `steps.csv`：10 Hz；
- RGB 原始采集：建议 20–30 Hz，至少稳定覆盖每个 10 Hz 状态；
- ACT 训练与第一版推理：10 Hz；
- STM32 在两个控制周期内保持同一 ACT 目标。

不要复制同一个 10 Hz 状态来伪造 20 Hz 训练样本，也不要为了对齐状态使用未来 RGB。

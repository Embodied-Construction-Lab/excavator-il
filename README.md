# excavator-il

缩比液压挖掘机的真机示教采集、数据校验、LeRobotDataset 转换与 ACT 训练项目。

当前活动链路为：

```text
PC 双手柄 20 Hz
  -> excavator_joystick.v1 UDP
Orin Collector
  -> STM32 命令 + 20 Hz stm32_control_telemetry.v2
  -> 30 Hz 前视 UVC RGB
  -> 原始 Episode
  -> 因果对齐的 10 Hz steps.csv
  -> LeRobotDataset v3 / ACT
```

正式采集期间，Collector 独占 `/dev/ttyTHS1`。不要同时运行
`excavator-orin-runtime/orin_state_sender.py`，两者会争用同一 STM32 串口。

## 1. 安装

Orin 仅安装采集依赖（Python 3.10）：

```bash
cd ~/workspace/excavator-il
conda env create -f environment.orin.yml
conda activate excavator-il-collector
```

PC 手柄发送端：

```bash
python -m pip install -e '.[teleop]'
```

训练机/数据转换环境：

```bash
conda env create -f environment.yml
conda activate excavator-il
```

`training` extra 将 LeRobot 固定到项目验证过的 fork commit
`12b88fce029cc3a8a94b061cd9e790018873c769`，不要改用旧的 `lerobot`、`act_env` 或
`collect_data_env` 环境执行本项目。

## 2. 首次配置

PC 查看手柄 GUID 和 SDL 设备路径：

```bash
excavator-il list-joysticks
```

Linux 上同时确认带 USB 序列号的稳定路径：

```bash
find /dev/input/by-id -maxdepth 1 -type l -name '*-event-joystick' \
  -printf '%f -> %l\n' | sort
```

将两个 GUID 同步写入以下配置，并在 PC 配置的每个 `devices` 项中写入对应的绝对
`device_path`：

- `config/teleop.pc.json`
- `config/collection.orin.json`

同型号手柄可以具有相同 GUID，但两个 `device_path` 必须按 USB 序列号区分左右并解析到不同的
物理设备。路径缺失、GUID 不匹配或两个路径指向同一设备时，Teleop 会在创建发送 socket 前退出。
稳定路径识别要求 Pygame 实际加载 SDL 2.24 或更新版本；版本过低时命令会直接报错退出。

同时确认 PC/Orin IP、`/dev/ttyTHS1`、`/dev/video0`、相机尺寸，以及
`episode_defaults.provenance` 中的固件和标定版本。配置文件是现场副本，换网络或设备后应显式修改，
代码中没有硬编码现场地址。

## 3. 双端采集

先在 Orin 停止所有会打开 STM32 串口的旧进程，然后启动 Collector：

```bash
conda activate excavator-il-collector
cd ~/workspace/excavator-il
excavator-il collect --config config/collection.orin.json
```

在 PC 启动 20 Hz 双手柄发送端：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
excavator-il teleop --config config/teleop.pc.json --print-every 20
```

在 Orin 的另一个终端控制 Episode：

```bash
excavator-il episode --config config/collection.orin.json start \
  --task ExecuteDig --operator operator_01

# 完整动作结束
excavator-il episode --config config/collection.orin.json stop --success

# 示教失败但需要保留原始数据
excavator-il episode --config config/collection.orin.json stop \
  --failure-reason bucket_empty

# 紧急人工中止
excavator-il episode --config config/collection.orin.json abort \
  --reason emergency_stop
```

松开 deadman、手柄超过 150 ms 无新包、Collector 启动或退出时都会向 STM32 发送明确零命令。
串口异常、相机异常或落盘异常会保留当前目录并把 Episode 标成中止。

## 4. 生成与校验训练样本

每个 `sensor_is_new=1` 的 STM32 状态只选择不晚于该状态的最新动作和 RGB。默认动作年龄
不超过 100 ms、图像年龄不超过 120 ms：

```bash
excavator-il build-steps ~/excavator-data/raw/episode_0001
excavator-il validate ~/excavator-data/raw/episode_0001
```

`quality_report.json` 给出各流频率/周期、序号缺口、乱序、串口解析失败、命令写失败、
传感器无效数、动作/图像年龄分布，以及可训练样本数和拒绝原因。未通过校验的 Episode 不应转换。

## 5. 转换与 ACT 冒烟

```bash
excavator-il convert \
  ~/excavator-data/raw/episode_0001 \
  --output-root data/lerobot/excavator_rgb_v1 \
  --repo-id local/excavator_rgb_v1 --fps 10

excavator-il smoke-train
```

ACT 接口固定为一台前视 RGB、11 维状态、4 维动作
`[boom, stick, bucket, swing]`，训练频率 10 Hz。详细字段和时钟语义见
[docs/data_contract_v1.md](docs/data_contract_v1.md)。下一阶段的采集数量、质量门槛、数据集划分和
ACT 训练命令见 [docs/collection_training_runbook.md](docs/collection_training_runbook.md)。

## 6. STM32 固件配套

配套源码位于 `../F407/data_celect`。Collector 要求：

- USART2：460800 baud；
- 串口命令：`stm32_manual_command.v1`；
- 遥测：`stm32_control_telemetry.v2`；
- 控制/遥测 20 Hz，新状态 10 Hz；
- 未知 schema、重复/乱序或超过 300 ms 的命令归零。

Host 回归：

```bash
cd ../F407/data_celect
Tests/run_host_tests.sh
```

CubeIDE 工程仍需在具备 ARM 工具链的构建机上重新生成/构建并烧录；PC 当前生成的 Debug Makefile
含历史 Windows 绝对路径，不能作为固件构建成功的证据。

## 7. 回归测试

```bash
conda activate excavator-il
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_cov \
  --cov=excavator_il --cov-report=term-missing
```

在安装 `.[training,test]` 的环境中会同时执行 LeRobot 转换和 ACT 训练/推理冒烟；未安装训练可选组时，
这两项会明确跳过，Collector/协议/时间对齐测试仍会执行。

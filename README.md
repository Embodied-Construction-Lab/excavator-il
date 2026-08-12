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
cd /home/jetson16/workspace_excavator/excavator-il
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
本地 `excavator_teleop_config.v3` 每只手柄只配置 X/Y 两个轴；线上 Z1/Z2 由 PC 和 Collector
两层强制为零，人工示教不得通过 Z 轴触发左右行走。

同时确认 PC/Orin IP、`/dev/ttyTHS1`、`/dev/video0`、相机尺寸，以及
`episode_defaults.provenance` 中的固件和标定版本。配置文件是现场副本，换网络或设备后应显式修改，
代码中没有硬编码现场地址。

## 3. 双端采集

发动机关闭、Orin/STM32/传感器/相机上电且 deadman 保持释放时，先运行一次自动零命令 soak：

```bash
conda activate excavator-il
python scripts/run_zero_command_soak.py
```

脚本根据 `config/guided_episode.pc.json` 的 `runtime.zero_soak_duration_s`（默认 30 秒）自动启动
Collector、诊断 Episode 和 teleop，全程监视 deadman，结束后以
`aborted: zero_command_soak_complete` 保留原始证据，并检查 STM32/新状态/手柄/相机频率分别约为
20/10/20/30 Hz。任一非零串口命令、非零 STM32 动作回显、有效专家动作、手柄超时、解析/写入
失败、传感器无效或序号异常都会让脚本非零退出。该 Episode 永不进入训练集。

首次真机短 Episode 建议从 PC 使用引导脚本。它自行启动并清理 Orin Collector 和 PC teleop，
先创建处于记录待命状态的 Episode，再启动 teleop 并通过 ACK 门禁；正常完成后自动运行
`build-steps`、`validate` 并保存
`quality_report.json` 输出：

```bash
conda activate excavator-il
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
python scripts/collect_guided_episode.py
```

teleop 在创建 UDP socket 前先要求四个 XY 居中且 deadman 释放，连续稳定 10 个 20 Hz 样本；
这样不会把 Pygame/SDL 初始化瞬态发送到 Orin。阈值、稳定样本数和超时集中在
`config/teleop.pc.json` 的 `startup_gate`。

现场参数集中在 `config/guided_episode.pc.json`。当前机器开机后液压即具备动作条件，没有独立的
软件“锁定/解锁”阶段；运行脚本前必须清空作业区、保证急停可立即操作，异常时立即释放 deadman、
手柄回中并按 `Ctrl+C`。Recorder、teleop 和 ACK 门禁都就绪后，脚本才提示等待 deadman；按下后
看到“记录已开始”再操纵双杆 XY。回中并松开 deadman 时，脚本先立即关闭所有原始流并将
Episode 标为 `pending_review`，随后才提示输入 `成功/s`、`失败/f` 或 `重录/r`，因此人工输入
耗时不会继续写进该 Episode。脚本不设置固定采集时长。
`重录`仅删除本轮刚封存的 Episode，下一次 deadman 复用同一个 Episode 编号。
异常或人工 `Ctrl+C` 中止仍保留原始证据。该条诊断数据不进入 Pilot 训练集。

原始 Episode 保存在 Orin 的 `config/collection.orin.json` 中 `data_root` 指定的位置；当前为
`/home/jetson16/workspace_excavator/data/excavator-data`。PC 的引导、teleop 和校验输出
仅保存在 `config/guided_episode.pc.json` 的 `runtime.log_dir`；当前解析为仓库内 `logs/`。

常规多条 Episode 采集仍可使用以下分端命令。

先在 Orin 停止所有会打开 STM32 串口的旧进程，然后启动 Collector：

```bash
conda activate excavator-il-collector
cd /home/jetson16/workspace_excavator/excavator-il
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
excavator-il build-steps \
  /home/jetson16/workspace_excavator/data/excavator-data/episode_0001
excavator-il validate \
  /home/jetson16/workspace_excavator/data/excavator-data/episode_0001
```

`quality_report.json` 给出各流频率/周期、序号缺口、乱序、串口解析失败、命令写失败、手柄
超时安全回零次数 `joystick_timeout_count`、
传感器无效数、动作/图像年龄分布，以及可训练样本数、片段数和拒绝原因。150 ms 安全回零门不
放宽；能定位并确认连续恢复的孤立事件会生成 `training_segments.json`，只隔离故障窗口，无法
恢复的事件仍校验失败。未通过校验的 Episode 不应转换。

## 5. 转换与 ACT 冒烟

```bash
excavator-il convert \
  /home/jetson16/workspace_excavator/data/excavator-data/episode_0001 \
  --output-root data/lerobot/excavator_rgb_v1 \
  --repo-id local/excavator_rgb_v1 --fps 10

excavator-il smoke-train
```

ACT 接口固定为一台前视 RGB、11 维状态、4 维动作
`[boom, stick, bucket, swing]`，训练频率 10 Hz。详细字段和时钟语义见
[docs/data_contract_v1.md](docs/data_contract_v1.md)。下一阶段的采集数量、质量门槛、数据集划分和
ACT 训练命令见 [docs/collection_training_runbook.md](docs/collection_training_runbook.md)。
每个连续 Training Segment 写成独立 LeRobot Episode，并使用 LeRobot 原生 `action_is_pad` 防止
ACT 动作块跨越故障边界；转换帧保留 parent Episode、segment 和原始 frame index。

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

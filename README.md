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
本地 `excavator_teleop_config.v4` 每只手柄配置 X/Y/Z 三个轴。Z1/Z2 仅在`仅遥操作`和正式 Episode
开始前的人工预定位阶段控制左右履带；Collector 一旦进入本轮 Episode，直到进程退出都在 STM32
串口边界强制 Z1/Z2 为零，ACT 标签仍只有 `[boom,stick,bucket,swing]` 四维。

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

首次真机短 Episode 建议从 PC 使用引导脚本。它自行启动并清理 Orin Collector 和 PC teleop。
启动时可选择 `RL定位/l`、`人工预定位/y` 或 `直接采集/n`；正常完成后自动运行
`build-steps`、`validate` 并保存
`quality_report.json` 输出：

```bash
conda activate excavator-il
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
python scripts/collect_guided_episode.py
```

teleop 在创建 UDP socket 前先要求六个 X/Y/Z 轴回中且 deadman 释放，连续稳定 10 个 20 Hz 样本；
这样不会把 Pygame/SDL 初始化瞬态发送到 Orin。阈值、稳定样本数和超时集中在
`config/teleop.pc.json` 的 `startup_gate`。

现场参数集中在 `config/guided_episode.pc.json`。当前机器开机后液压即具备动作条件，没有独立的
软件“锁定/解锁”阶段；运行脚本前必须清空作业区、保证急停可立即操作，异常时立即释放 deadman、
手柄回中并按 `Ctrl+C`。选择 `人工预定位/y` 时，Collector 先启动一次不创建 Episode 的 teleop；可
按住 deadman 手动移动到 RL Follow 的交接位姿附近。完成后 X/Y/Z 全部回中、松开 deadman 并输入
`完成/c`，脚本会停止预定位 teleop，随后才创建正式 Episode，并重新启动 teleop 以再次执行
中位稳定和 ACK 门禁。预定位数据不落盘、不占 Episode 编号，也不进入训练集。选择
`直接采集/n` 或直接按 Enter 时保持原流程。

选择 `RL定位/l` 前，PC 必须已经运行 AiryLidar `live_commissioning` Operator，但 Orin 上不要
手工启动 `orin_state_sender.py`。引导脚本会按 `rl_preposition` 配置启动唯一的 Orin RL Runtime，
从 `mission_config` 读取 Dig 目标并执行一次 execution-strict `Plan DIG → Follow`。只有 Follow
返回 `SUCCEEDED` 且 `quiescence_confirmed=true` 后，脚本才用 `SIGTERM` 触发 RL Runtime 的终态
归零清理，确认 `/dev/ttyTHS1` 已释放，再启动 Collector。任何一步失败都不会创建 Episode。
RL 模式下 Episode 的 `dig_target_m` 取本次 AiryLidar Mission 目标；人工/直接模式仍使用
`episode.dig_target_m`。两者都只是溯源元数据，不是 ACT 输入。

正式 Recorder、teleop 和 ACK 门禁都就绪后，脚本才提示等待 deadman；按下后
看到“记录已开始”再操纵双杆 XY，Z1/Z2 保持回中且不会转发到履带。全部回中并松开 deadman 时，脚本先立即关闭所有原始流并将
Episode 标为 `pending_review`，随后才提示输入 `成功/s`、`失败/f` 或 `重录/r`，因此人工输入
耗时不会继续写进该 Episode。脚本不设置固定采集时长。
`重录`仅删除本轮刚封存的 Episode，下一次 deadman 复用同一个 Episode 编号。
异常或人工 `Ctrl+C` 中止仍保留原始证据。该条诊断数据不进入 Pilot 训练集。

### 本地采集 UI

PC 端提供一个只监听 `127.0.0.1` 的本地 Web UI，复用与终端引导脚本相同的
采集/遥操作工作流。UI 不直接写 STM32 串口，也不自己打开 Orin 相机；点击开始后由
受控子进程依次管理 RL Runtime、Collector 和 PC teleop，点击“安全停止”只向本轮精确子进程发送
中止信号并复用原有归零清理。

首次安装或更新 PC 环境：

```bash
conda activate excavator-il
python -m pip install -e '.[teleop,training,test,ui]'
```

日常启动：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
conda activate excavator-il
python scripts/check_site_config.py
python scripts/run_collection_ui.py
```

`check_site_config.py` 只读核对当前 PC/Orin IP、Collector/teleop/预览端口与串口引用；它不会
修改网络或生成配置。换网、换电脑或改端口后应先让该检查通过。

浏览器默认打开 `http://127.0.0.1:8088/`。页面可选择：

- `RL 到目标点`：从 AiryLidar `excavation_demo.json` 中选择 `dig_01/02/03`，再执行该点的
  Plan → Follow；要求 `live_commissioning` Operator 已在 PC 运行；
- `手工预定位`：按住 deadman，用 X/Y 调整工作装置、Z1/Z2 调整左右履带，全部回中并释放后点击“完成手工预定位”；
- `直接采集`：跳过定位，直接等待 Recorder 与 deadman；
- `仅遥操作`：只启动 Collector 与 PC teleop，不创建 Episode、不占用编号、不写训练数据；按住
  deadman 可用 X/Y 控制工作装置、Z1/Z2 控制左右履带，释放即六轴回零，点击“安全停止”退出；
- Episode 结束后点击“成功”“失败”或“重录”；完成后可以直接选择下一点继续，页面显示本次
  UI 运行期间的完成计数，适合连续采集 Pilot/正式批次。
- `启动 RL + RViz`：从页面启动既有 AiryLidar `live_commissioning` Operator；不会复制规划、
  策略或可视化实现。录制演示视频时保留弹出的原生 RViz 窗口即可。

配置集中在 `config/collection_ui.pc.json`，其中 `guided_config` 继续指向权威的
`guided_episode.pc.json`。前视画面由 Collector 已经拥有的相机线程维护一个最新 JPEG，并通过
Orin `18092/tcp` 只读输出；因此 Collector 尚未启动或已经退出时页面显示等待状态是正常的。相机
预览连接失败后会每秒重试，所以页面可以先于 Collector 打开，Collector 就绪后不需要手动刷新。
预览不落入训练数据路径，不改变相机 30 Hz 采集或 Recorder 的 Episode 生命周期。
同一 Collector 还把已经解析的 STM32 最新帧投影为只读遥测，页面以 2 Hz 显示动臂/斗杆/铲斗/
回转关节角以及三个油缸活塞杆伸缩量。UI 不会为此再次打开串口；Collector 未运行时显示等待。

原生 RViz 是 Qt 桌面程序，第一版不把窗口强行嵌入浏览器。页面保留 `visualization_url` 扩展位；
未来需要浏览器三维视图时，再将 ROS 2 数据通过 Foxglove Bridge 提供给浏览器，并保持采集状态机
与可视化解耦。

### RL + ACT 混合 Mission（分段验收）

同一个本地 UI 还提供一个与采集状态机互斥的混合 Mission Module。它不是在浏览器里直接拼接
shell，而是按固定状态机执行：

```text
RL Plan/Follow DIG + 并行 ACT 模型/CUDA 预热（ACT 不打开硬件）
→ 终态零 + 退出 RL Runtime + 确认串口释放
→ 内部交接门放行 → ACT 打开串口/相机并挖掘
→ 终态零 + 退出 ACT Runtime + 确认串口释放
→ RL Plan/Follow DUMP → Orin ExecuteDump
→ RL Runtime 保持零动作热待命 → Plan/Follow DIG 返回同一挖掘点 → 退出
```

第一版 ACT 完成条件是 `config/hybrid_mission.pc.json` 中的 `act.max_steps=130`：live warmup 后累计
130 个有效 10 Hz 推理 step，约 13 秒，然后由 ACT Runtime 自己终态回零并退出。130 来自当前正式
采集 Episode 有效步数的中位数附近，它是有界实验时长，不是假装成视觉“挖掘成功检测器”。任何
遥测、相机、推理、串口或资源交接错误都会提前失败并归零。后续真实数据足够时再评估 learned
success detector。

可先在页面点击“启动 RL + RViz”并等待“已就绪”，用于纯 RL 演示或提前录屏；若直接启动分段/自动 Mission，后端会在 Operator 未就绪时先自动启动并等待就绪。若已经在外部终端运行 Operator，则继续
沿用该窗口，不要重复启动。Orin 不要手工启动 `orin_state_sender.py`、Collector 或 ACT Runtime。
Web UI 提供：

- `开始分段验证`：每完成一段后停在明确的等待阶段，由操作者点击下一段；点击执行 ACT 段本身
  作为本地显式运动授权，页面自动发送固定授权值，不再要求手工输入口令；
- `自动装车 1～5 铲`：选择铲数后点击一次即连续执行；以页面选中的 DIG 点为第一铲，后续按
  Mission 配置顺序循环 `dig_01 → dig_02 → dig_03 → dig_01`。RL 返回阶段直接去下一铲点位，
  到达后从交接位姿开始 ACT，不额外增加一次策略冷启动；
- `安全停止`：中断当前精确 owner，执行终态零并检查 `/dev/ttyTHS1` 释放。

为减少分段演示中的静止冷启动，分段 Mission 从开始到完成由同一个后台 worker 持有：第一段 RL
运行时并行加载 ACT checkpoint 并完成 synthetic CUDA warmup，但 ACT 在内部交接门放行前不打开
`/dev/ttyTHS1` 或 `/dev/video0`；只有 RL 已终态回零并确认串口释放后才放行。倾倒完成后同一个 RL
Runtime 保持零动作待命，返回时直接复用；多铲模式还会在 RL 返回期间预热下一铲 ACT。等待阶段
点击“安全停止”会清理预热 ACT 或热待命 RL 并释放设备。该优化只隐藏进程冷启动，不缩短轨迹
跟踪、ACT 130 step、固定倾倒或安全回零。

Web UI 通过 SSH 非交互启动 ACT Docker。Orin 的 `jetson16` 必须能直接运行 Docker；实验机可一次性
加入 `docker` 组并重新登录，之后先验证 `docker info`：

```bash
sudo usermod -aG docker jetson16
# 完全退出当前 SSH 会话并重新登录
docker info >/dev/null && echo docker-ready
```

`docker` 组具备主机高权限，只应在受控实验 Orin 上使用。若没有该权限，自动 ACT 段会在运动前
失败，不会把密码写入配置。更新 ACT Runtime Python 代码后还必须重新构建 Orin 镜像；仅 `git pull`
不会改变已存在镜像中的代码。

混合 Mission 期间 Collector 未运行，因此当前页面的 Collector 相机/遥测卡片会显示等待；ACT
容器仍独占读取相机用于推理，但 `--network=none` 下不对外提供预览。这不影响闭环控制，后续如确有
实验需求再复用 ACT 内部最新帧接口，不新增第二个相机 owner。

不启动 Web UI 时，也可从 PC 单独运行同一遥操作流程：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
conda activate excavator-il
python scripts/run_guided_teleop.py
```

脚本通过 SSH 启动唯一的 Orin Collector，完成安全 ACK 后即可按住 deadman 操纵双杆；`Ctrl+C`
会先停止 teleop、再停止 Collector 并释放串口。该入口与采集数据生命周期解耦，但仍复用 Collector
作为串口、相机和安全回零网关，因此不得同时运行 RL Runtime、ACT Runtime 或另一个 Collector。
每次从 Web UI 或对应引导脚本启动新的串口工作流时，PC 会先检查 Orin 的实际串口 owner。若 owner
的完整 argv 精确匹配本配置启动的遗留 Collector 或 `orin_state_sender.py`，则先发送 `SIGTERM` 并
确认 `/dev/ttyTHS1` 已释放后再接管；未知程序仍拒绝启动并显示 PID/命令，不使用全局 `pkill`。
Collector 还会把自己已解析的 STM32 状态按 `machine_state_v1` 发送到
`config/collection.orin.json.machine_state_udp`。如果 AiryLidar live Operator/RViz 已启动，其现有
PC 状态桥会继续发布 `/joint_states` 并实时显示挖掘机；该功能不启动第二个串口读取进程，也不参与
动作控制。

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

只验证离线数据链时，可把一条已校验 Episode 生成为带唯一 ID 的合成副本。合成副本在
`episode.json` 中固定标记 `training_eligible=false`，转换时必须显式使用
`--allow-synthetic`；它们只能验证转换、训练、checkpoint 保存和推理，不能证明模型学会挖掘，
也不得混入正式 Pilot：

```bash
excavator-il synthesize-episodes data/raw/pipeline_source/episode_0004 \
  --output-root data/raw/synthetic_episode_0004_x10 --count 10

excavator-il convert data/raw/synthetic_episode_0004_x10/synthetic_episode_* \
  --output-root data/lerobot/synthetic_episode_0004_x10 \
  --repo-id local/synthetic_episode_0004_x10 --fps 10 --allow-synthetic
```

合成 Episode 的图像使用硬链接节省空间，任何流程都不得原地修改这些图像；删除合成目录不会
删除源图像。

ACT 接口固定为一台前视 RGB、11 维状态、4 维动作
`[boom, stick, bucket, swing]`，训练频率 10 Hz。详细字段和时钟语义见
[docs/data_contract_v1.md](docs/data_contract_v1.md)。下一阶段的采集数量、质量门槛、数据集划分和
ACT 训练命令见 [docs/collection_training_runbook.md](docs/collection_training_runbook.md)。
逐条人工训练、监控、checkpoint 校验和断点续训命令见
[docs/manual_act_training.md](docs/manual_act_training.md)。
Orin NVIDIA PyTorch 镜像构建、GPU 自检、checkpoint 传输、只读离线推理和在线 Runtime 见
[docs/orin_act_inference.md](docs/orin_act_inference.md)。
PC/Orin 独立有线控制网的首次配置、路由验收、SSH 和回滚见
[docs/pc_orin_direct_control_network.md](docs/pc_orin_direct_control_network.md)。
在线 `act-runtime` 固定复用 LeRobot `ACTPolicy.select_action()`：motion 仅接受 v2 deployment
manifest、单前视因果 RGB、11 维状态和 `[boom, stick, bucket, swing]` 四维动作。启动时必须先通过
synthetic CUDA warmup 与真实相机/STM32 live warmup；状态丢帧、安全中断或无因果图像时清空
action queue 并保持零命令。motion 由 Orin 终端的一次性显式授权启动，运行期不依赖 PC teleop、
deadman、UDP 或 HMAC；PC 手柄链路仅用于人工示教采集和手动诊断。完整 shadow/motion 门禁与
验收命令以该 Orin 手册为准。
现场先运行 `python scripts/diagnose_stm32_link.py` 做 10 秒只读 USART2 验收；诊断分别检查遥测
接收频率、控制循环平均频率和最大接收间隔，不把两个独立 20 Hz 周期的合法相位交换误报为丢包。
通过后在 Orin 运行 `bash scripts/run_act_shadow.sh`，该脚本固定不传 motion authorization。
Shadow 或发动机关闭的 motion Runtime 正常退出后，分别运行
`bash scripts/inspect_latest_act_runtime_log.sh shadow` 或
`bash scripts/inspect_latest_act_runtime_log.sh motion`，对最新 JSONL 证据自动检查 10 Hz 时序、
因果图像、动作范围、串口写入、轴映射、命令序号以及启动/终止零命令。
每个连续 Training Segment 写成独立 LeRobot Episode，并使用 LeRobot 原生 `action_is_pad` 防止
ACT 动作块跨越故障边界；转换帧保留 parent Episode、segment 和原始 frame index。

## 6. STM32 固件配套

配套源码位于 `../F407/data_celect`，它也是 RL/ACT 共用的统一 STM32 固件。Collector 只使用其
manual 模式，要求：

- USART2：460800 baud；
- 串口命令：`stm32_manual_command.v1`；
- 遥测：`stm32_control_telemetry.v2`；
- 控制/遥测 20 Hz，新状态 10 Hz；
- 未知 schema、重复/乱序或超过 300 ms 的命令归零。

统一固件另接受 RL 的 `stm32_velocity_command.v1`，但两种模式互斥且必须先用目标 schema 的零命令
切换。RL Runtime 发送 terminal zero 并释放 `/dev/ttyTHS1` 后，Collector 可直接同步 sequence 并
占用 manual 模式；无需重启 STM32 或重新烧录。

Host 回归：

```bash
cd ../F407/data_celect
Tests/run_host_tests.sh
```

CubeIDE 工程仍需在具备 ARM 工具链的构建机上清理、构建并烧录；PC 当前生成的 Debug Makefile
含历史 Windows 绝对路径，不能作为固件构建成功的证据。

## 7. 回归测试

```bash
conda activate excavator-il
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_cov \
  --cov=excavator_il --cov-report=term-missing
```

在安装 `.[training,test]` 的环境中会同时执行 LeRobot 转换和 ACT 训练/推理冒烟；未安装训练可选组时，
这两项会明确跳过，Collector/协议/时间对齐测试仍会执行。

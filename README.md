# excavator-il

缩比液压挖掘机的真机示教采集、数据校验、LeRobotDataset 转换与 ACT 训练项目。

当前活动链路为：

```text
PC 双手柄 20 Hz
  -> excavator_joystick.v1 UDP
Orin Collector
  -> STM32 命令 + 20 Hz stm32_control_telemetry.v2
  -> 30 Hz front + dump 双路 UVC RGB
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

同时确认 PC/Orin IP、`/dev/ttyTHS1`、两路相机尺寸和
`config/collection.orin.json` 中的稳定 `/dev/v4l/by-path/...-video-index0` 路径，以及
`episode_defaults.provenance` 中的固件和标定版本。配置文件是现场副本，换网络或设备后应显式修改，
代码中没有硬编码现场地址。两台同型号相机的 `by-id` 会冲突，禁止改回易漂移的 `/dev/videoN`；
当前前视与倾倒相机分别接在 Orin USB `0:2.1` 和 `0:2.4`，采集前还必须确认倾倒相机没有被机身遮挡。

## 3. 双端采集

发动机关闭、Orin/STM32/传感器/相机上电且 deadman 保持释放时，先运行一次自动零命令 soak：

```bash
conda activate excavator-il
python scripts/run_zero_command_soak.py
```

脚本根据 `config/guided_episode.pc.json` 的 `runtime.zero_soak_duration_s`（默认 30 秒）自动启动
Collector、诊断 Episode 和 teleop，全程监视 deadman，结束后以
`aborted: zero_command_soak_complete` 保留原始证据，并检查 STM32/新状态/手柄/相机频率分别约为
20/10/20/30+30 Hz。报告分别列出 `camera_front` 和 `camera_dump`；任一路频率异常、任一非零串口
命令、非零 STM32 动作回显、有效专家动作、手柄超时、解析/写入
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
正式 campaign 中，无论选择 RL 定位、人工预定位还是直接采集，Episode 的 `dig_target_m` 都按
本槽位 `dig_point_id` 从 `rl_preposition.demo_config` 解析；RL 定位结果还必须与该权威坐标一致。
脚本在预检和每次 Episode 创建前都会重新核对 PC 上该文件的 clean Git HEAD、仓库相对路径、SHA-256
与目标坐标，并把结果写入 `episode.json.target_source_provenance`；任何漂移都会在录制前拒绝。
只有未携带 collection protocol 的旧式诊断入口才兼容使用 `episode.dig_target_m`。该字段只作
溯源元数据，不是 ACT 输入。

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
- Episode 结束后点击“成功”“失败”或“重录”。页面从 Orin 原始数据根目录只读计算 200 条 campaign
  的权威下一槽位，自动填写 `task_variant / soil_reset_block_id / dig_point_id` 并显示总进度；每条完成
  后重新读取，浏览器 `localStorage` 不参与计数。SSH 或 campaign 校验不可用时正式采集会被禁止，
  但不创建 Episode 的`仅遥操作`仍可使用。
- `启动 RL + RViz`：从页面启动既有 AiryLidar `live_commissioning` Operator；不会复制规划、
  策略或可视化实现。录制演示视频时保留弹出的原生 RViz 窗口即可。

配置集中在 `config/collection_ui.pc.json`，其中 `guided_config` 继续指向权威的
`guided_episode.pc.json`。前视与倾倒画面由同一个 Collector 独占的两个相机线程分别维护最新 JPEG，
并通过 Orin `18092/tcp` 只读输出；UI 不会再次打开 UVC 设备。Collector 尚未启动或已经退出时页面
显示等待状态是正常的。两路相机
预览连接失败后会每秒重试，所以页面可以先于 Collector 打开，Collector 就绪后不需要手动刷新。
预览不落入训练数据路径，不改变两路 30 Hz 采集或 Recorder 的 Episode 生命周期。任一路相机采集
线程异常都会让活动 Episode 以 `collector_runtime_error` 中止并执行安全回零，不能留下看似成功的
单路数据。
同一 Collector 还把已经解析的 STM32 最新帧投影为只读遥测，页面以 2 Hz 显示动臂/斗杆/铲斗/
回转关节角以及三个油缸活塞杆伸缩量。UI 不会为此再次打开串口；Collector 未运行时显示等待。

原生 RViz 是 Qt 桌面程序，当前 Web UI 不嵌入 RViz/Foxglove，也不显示三维可视化占位。
需要录制三维状态时，使用页面“启动 RL + RViz”打开的原生 RViz 窗口。

### RL + ACT 混合 Mission（分段验收）

同一个本地 UI 还提供一个与采集状态机互斥的混合 Mission Module。它不是在浏览器里直接拼接
shell，而是按固定状态机执行：

```text
Mission 开始时一次性启动 Orin resident owner（唯一串口 owner）
→ 一次性启动 ACT Worker，加载 checkpoint、CUDA warmup 并持续占用相机
→ RL Plan/Follow DIG（RL ONNX 已在 resident owner 内存中）
→ STM32 确认 RL terminal zero 与 ACT target zero
→ 同一 owner 内切换 generation，ACT 直接执行挖掘
→ ACT 末段并行准备 DUMP 轨迹
→ STM32 确认 ACT terminal zero 与 RL target zero
→ 激活已准备轨迹；不满足新鲜度/起点契约时安全回退到普通 Plan
→ ExecuteDump → RL Follow 下一挖掘点 → 下一铲
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
- `自动装车 1～9 铲`：选择铲数后点击一次即连续执行；以页面选中的 DIG 点为第一铲，后续按
  Mission 配置顺序循环 `dig_01 → dig_02 → dig_03 → dig_01`。RL 返回阶段直接去下一铲点位，
  到达后从交接位姿开始 ACT，不额外增加一次策略冷启动；
- `安全停止`：中断当前精确 owner，执行终态零并检查 `/dev/ttyTHS1` 释放。

默认 `resident` backend 不再在每段退出/重启硬件 Runtime。Orin owner 在整个 Mission 内持续拥有
`/dev/ttyTHS1`，RL ONNX 常驻其中；独立 ACT Worker 在整个 Mission 内保持 checkpoint、CUDA、
action queue 与 `/dev/video0` 就绪，但永不映射串口。交接只更换带
`control_generation` 的 Motion Authority，并严格等待旧来源 terminal zero、目标模式 zero claim
和首条非零命令的 STM32 telemetry ACK。ACT 最后 20 steps 时，PC 并行准备 DUMP 轨迹；若准备结果
过期、起点不兼容或未及时就绪，则保持归零并回退到普通 live Plan，不复用可疑轨迹。该结构消除模型、
容器、串口和相机冷启动，但不缩短真实轨迹跟踪、ACT 130 steps、固定倾倒或必要的 zero/ACK。

`config/act_runtime.orin.json` 的 `dig_policy_backend` 是挖掘算法选择边界；当前活动值必须显式为
`lerobot_act`。旧配置缺少该字段时仍兼容为 `lerobot_act`，未知 backend 会在加载配置时直接拒绝。
后续 Diffusion Policy 应实现同一具名状态、具名 RGB 角色与归一化
`[boom, stick, bucket, swing]` 输出契约，不修改 resident owner 或 Mission 运动权限。

PC 每 0.4 s 续约一次 1.5 s Mission lease。PC/Web UI 异常退出、网线断开或续约停止后，Orin owner
会不可逆 terminal disarm，等待最终零命令 ACK 后释放串口；ACT Worker 随本机 data link 关闭退出。
点击“安全停止”时先禁止新的续约，再 terminal disarm，不能让阻塞的远程清理延长运动授权。

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

在 Orin 的最新 `excavator-il` 工作树执行增量重建并验证常驻入口：

```bash
cd /home/jetson16/workspace_excavator/excavator-il
git status --short

sudo docker build --progress=plain --network=none \
  --build-arg BASE_RUNTIME_IMAGE=excavator-act-inference:jp72-pytorch261 \
  --build-arg EXCAVATOR_IL_REVISION="$(git rev-parse HEAD)$(test -z "$(git status --porcelain)" || printf '%s' -dirty)" \
  -f docker/act-inference.incremental.Dockerfile \
  -t excavator-act-inference:jp72-pytorch261 .

docker run --rm --network=none \
  excavator-act-inference:jp72-pytorch261 \
  python3 -c 'import excavator_il.resident_act_runtime; print("resident-act-import-ok")'
```

增量 Dockerfile 也会在 build 阶段执行同一 import；任一步失败都不要启动 resident Mission。

混合 Mission 仍保持单一硬件 owner，观测输出也不随策略切换消失：常驻 ACT Worker 在整个 Mission
内只独占前视相机；启动脚本把主机 `camera_front.device` 显式映射为容器 `/dev/video0`，不把倾倒
相机映射给当前 front-only ACT v1。同一次前视采样同时供策略 RGB 与 Web JPEG；resident owner 持续把 STM32
遥测发布为既有 `machine_state_v1`，因此 RL/ACT 两个阶段 RViz 关节状态都能更新。这些接口只提供
只读预览和状态，不接收任何运动命令，也不改变串口 owner 互斥。

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

原始 Episode 保存在 Orin 的 `config/collection.orin.json` 中 `data_root` 指定的位置；当前正式
ICRA 2027 campaign 为
`/home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw`。历史目录不删除，也不与
本 campaign 混用。PC 的引导、teleop 和校验输出
仅保存在 `config/guided_episode.pc.json` 的 `runtime.log_dir`；当前解析为仓库内 `logs/`。

以下分端命令只用于链路诊断或恢复，不用于正式 ICRA 2027 campaign。正式示教统一从 WebUI/引导
流程启动，由它从 AiryLidar 权威目标配置解析 `dig_point_id` 对应坐标并写入 Episode；不要手工把
点位 ID 与另一组坐标拼在一起。

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
  --task ExecuteDig --operator zhaoshuai \
  --recording-purpose diagnostic

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
双相机正式示教必须同时提供上述三项采集协议；诊断 Episode 使用
`--recording-purpose diagnostic`，不会占用 campaign 槽位也不会进入训练集。串口异常、任一路相机
异常或落盘异常会保留当前目录并把 Episode 标成中止。

## 4. 生成与校验训练样本

每个 `sensor_is_new=1` 的 STM32 状态只选择不晚于该状态的最新动作和 RGB。默认动作年龄
不超过 100 ms、图像年龄不超过 120 ms：

```bash
excavator-il build-steps \
  /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw/episode_0001
excavator-il validate \
  /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw/episode_0001
```

`quality_report.json` 给出各流频率/周期、序号缺口、乱序、串口解析失败、命令写失败、手柄
超时安全回零次数 `joystick_timeout_count`、
传感器无效数、动作/图像年龄分布，以及可训练样本数、片段数和拒绝原因。150 ms 安全回零门不
放宽；能定位并确认连续恢复的孤立事件会生成 `training_segments.json`，只隔离故障窗口，无法
恢复的事件仍校验失败。未通过校验的 Episode 不应转换。

正式 ICRA 2027 campaign 的进度与证据状态应从 Orin 原始目录只读检查，而不是依赖浏览器计数：

```bash
python scripts/inspect_collection_campaign.py \
  --collection-config config/collection.orin.json --next

excavator-il record-collection-run \
  /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw/episode_0001 \
  --config config/collection_evidence.orin.json

python scripts/manage_experiment_run.py verify \
  --root /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/experiment-runs \
  --run-id collection_episode_0001
```

未采满时 `inspect_collection_campaign.py --next` 以退出码 2 返回属于预期；只在
`complete_and_valid=true` 时表示整批 200 条 campaign 无 duplicate、unplanned 或 malformed
Episode。`record-collection-run` 与 `manage_experiment_run.py verify` 是幂等的，只追加或核验证据，
不会改写原始 Episode。正式证据还会读取并快照
`config/icra2027_collection_campaign_provenance.json`，逐条核对 F407 固件 commit、机器配置 hash、
AiryLidar 目标配置的 commit/path/hash 与 `dig_point_id/dig_target_m`；仓库脏树或任一来源漂移都会
拒绝发布成功 Run。`quality_report` 作为必需的完整性证据由统一 Evaluation Harness 校验，其相机质量
指标仍只从权威 raw Episode analyzer 生成，不重复制造指标。

`experiment_run_artifact.v2` 不再把可变外部路径当作最终证据。注册 artifact 时会在 Run 内
`artifact_snapshots/` 创建内容快照，`snapshot_path` 才是 finalize 与 Evaluation Harness 校验、解析的
权威字节；`source_path` 只用于追溯来源和幂等重录核对。实现优先尝试独立 CoW reflink，并在记录中
如实写入 `snapshot_method=reflink`；文件系统不支持时明确回退为逐字节 `copy`，此时 raw Episode 与
Experiment Run 快照可能让证据占用接近翻倍，不能把 fallback copy 误报成 reflink。正式采集前先用
campaign inspector 确认计划槽位，并结合试采 Episode 的实际大小检查目标分区余量：

```bash
python scripts/inspect_collection_campaign.py \
  --collection-config config/collection.orin.json --next
du -sh /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw
df -h /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1
```

## 5. 转换与 ACT 冒烟

```bash
excavator-il convert \
  /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw/episode_0001 \
  --output-root data/lerobot/excavator_rgb_v1 \
  --repo-id local/excavator_rgb_v1 --fps 10

excavator-il smoke-train
```

`convert` 默认使用 `--camera-roles auto`：当所有输入均为 v1 单相机时保留
`front`，当所有输入均为 v2 双相机时同时保留 `front` 与 `dump`；混合两种
相机契约会明确失败。当前已部署 ACT v1 的输入契约仍只有前视相机，因此为它建立训练视图时必须
显式传入 `--camera-roles front`，并与保留双路证据的正式数据集使用不同 repo ID；单相机消融也
使用这一显式参数。不得把 front-only 输出冒充正式双 RGB 数据集。失败、
中止或未标记成功的示教即使流数据可解析，也不会进入训练转换。

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

训练完成后的 checkpoint、resident Mission、以及 held-out 真机实验不再手工拼表；统一通过
evaluation harness 聚合：

```bash
python scripts/evaluate_experiment_runs.py \
  EvaluationReport/experiment_runs/runs/<run_id_1> \
  EvaluationReport/experiment_runs/runs/<run_id_2> \
  --aggregate-mode homogeneous \
  --output-dir EvaluationReport/evaluations/<evaluation_id>
```

`training_internal` 与 `held_out_experiment` 两种 scope 不能混合聚合；论文结论只使用后者。

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

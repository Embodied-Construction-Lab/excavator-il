# ACT 示教采集与训练运行手册

本文档描述 `excavator-il` 当前实现对应的下一阶段工作。权威字段、单位和时间语义见
[`data_contract_v1.md`](data_contract_v1.md)。

## 1. 阶段目标

### 阶段 A：设备与链路验收

1. 在 CubeIDE 中重新构建并烧录统一固件 `F407/data_celect`。
2. 确认双手柄、前视 UVC 相机和 STM32 串口设备名。
3. 完成 3 条由 deadman 按下/松开界定的短 Episode：成功、失败和人工中止各一条。
4. 每条均执行 `build-steps` 和 `validate`。

通过条件：

- 手柄与 STM32 控制/遥测接近 20 Hz，新状态接近 10 Hz，相机接近 30 Hz；
- 专家动作顺序严格为 `[boom, stick, bucket, swing]`；
- 不存在串口解析失败、命令写失败和文件写入队列丢帧；
- 训练样本不使用未来动作或未来图像；
- 正常停止、中止和进程退出均产生明确零命令。

### 阶段 B：小规模可学性验证

采集 10 条成功 Episode，覆盖安全范围内不同起始姿态和挖掘深度。先转换为一个小数据集，执行
ACT 冒烟训练，确认损失能够下降、模型能够保存并完成离线推理。此阶段只判断数据链是否可学，
不评价真机挖掘成功率。

### 阶段 C：ICRA 2027 双 RGB Demonstration 数据集

- 目标采集 200 条质量合格且人工确认成功的 Episode；
- 其中 `dig_only` 与 `dig_transport_dump` 各 100 条；
- 使用 20 个 `soil_reset_block_id`，每块 10 条并轮换三个 `dig_point_id`；
- 失败或人工中止 Episode 原地保留为证据，但不占 campaign 槽位且不能进入行为克隆转换；
- train/validation 按 soil block 原子切分，禁止按帧或单条 Episode 随机拆分；
- 正式 held-out live 评估使用采集结束后另行准备的新 soil blocks，不从这 200 条中挑选结果。

### 阶段 D：训练与离线评估

1. 转换一份保留 `front + dump` 的完整证据数据集，以及当前 ACT v1 使用的显式 `front` 视图。
2. 按 `soil_reset_block_id` 生成稳定 train/validation manifest 并 materialize。
3. 使用 train 数据训练策略，validation 数据选择 checkpoint。
4. 最终任务成功率只在独立 held-out live Experiment Runs 上报告。
5. 保存数据清单、Git commit、配置、随机种子、模型字节指纹和 checkpoint。

### 阶段 E：真机递进验证

先离线回放模型输出，不连接执行器；再进行短时、单次、可人工中止的真机测试。最终记录：

- 挖掘完成率；
- 单次动作时间；
- 人工干预率；
- 动作变化率与阀指令震荡；
- 挖掘后铲斗装载结果。

## 2. 首次配置

PC：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
conda activate excavator-il
excavator-il list-joysticks
```

`environment.yml` 已固定训练使用的 LeRobot fork commit，并安装 PC 手柄依赖；不要混用历史
`lerobot`、`act_env` 或 `collect_data_env` 环境。

将手柄 GUID 写入 `config/teleop.pc.json` 和 Orin 上的
`config/collection.orin.json`。PC 配置还必须为每个手柄写入带 USB 序列号的绝对
`/dev/input/by-id/*-event-joystick` 路径；同型号设备 GUID 可以相同，但路径必须不同并固定左右槽。
PC 上 Pygame 实际加载的 SDL 必须为 2.24 或更新版本。
`teleop.pc.json` 的 `startup_gate` 默认要求六个 X/Y/Z 在 `±0.15` 内且 deadman 释放，连续稳定
10 个 20 Hz 样本后才创建 UDP socket；5 秒仍不满足时失败关闭。该门只决定何时允许首包，
不修改、不取反或缩放专家动作。
同时核对 PC/Orin IP、串口、相机设备和 provenance 字段。

首次接入手柄、更换 USB Hub/PC 或修改轴映射后，保持挖掘机熄火并停止 Collector，在 PC 运行
本地交互诊断：

```bash
conda activate excavator-il
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
excavator-il diagnose-joysticks --config config/teleop.pc.json
```

该命令只读取 Pygame 手柄事件，不创建 socket、不访问 Orin/STM32。按终端提示依次保持中位、
移动 `X1/Y1/Z1/X2/Y2/Z2` 到两个端点并回中，再在 6 秒窗口内只按放计划作为 deadman 的按钮
3 次。Z1/Z2 不参与权威动作映射，只在 Recorder 外用于左右履带；正式记录期间 Collector 会在
STM32 串口边界清零。四个 XY 不取反、不缩放，并保持
`[boom,stick,bucket,swing]=[Y2,Y1,X2,X1]`。最终必须满足：

- 六个 X/Y/Z 轴均为 `PASS`，且每段只检测到预期手柄上的一个大幅活动轴；
- `detected_axis_indices` 与 `configured_axis_indices` 完全一致；
- `detected_deadman` 与 `configured_deadman` 完全一致；
- `matches_config=True`，进程退出码为 0。

任何一项失败时命令退出码为 3，禁止启动 Collector 或进入 Episode。诊断报告中的端点是未取反、
未缩放的 SDL 原始值；不要根据历史 `deploy_scale_model/doublestick_send.py` 的轴号直接修改配置。

Orin：

```bash
cd /home/jetson16/workspace_excavator/excavator-il
conda env create -f environment.orin.yml
conda activate excavator-il-collector
python -m pip install -e '.[collector]'
```

正式采集期间不能同时运行 `excavator-orin-runtime/orin_state_sender.py`，Collector 必须独占
STM32 串口。

### 网络、代理与防火墙迁移清单

当前现场已经增加 PC–Orin 千兆直连控制网。首次配置、路由隔离、双向延迟、SSH 主机密钥、应用
地址迁移和回滚步骤见 [`pc_orin_direct_control_network.md`](pc_orin_direct_control_network.md)。
下文 Wi-Fi 地址保留为外网、代理和迁移前备用 SSH 基线，不再作为正式采集控制链路的目标状态。

每次更换 Wi-Fi、PC 或 Orin 后都要重新执行本节。不要假定旧网络的 IP、接口名、UFW 规则或
代理 TUN 局域网段会自动迁移。当前 dr202 现场配置为：

| 项目 | 当前值 |
| --- | --- |
| PC Wi-Fi 接口 | `wlp128s20f3` |
| PC IP | `192.168.31.219` |
| Orin Wi-Fi 接口 | `wlP1p1s0` |
| Orin IP | `192.168.31.10` |
| Collector UDP | Orin `18090/udp`；应用层只允许当前 PC |
| PC 局域网代理 | `192.168.31.219:7897` |

先在两端发现实际地址和出接口，不要照抄旧接口名：

```bash
# PC
ip -4 -br address
ip route get <ORIN_IP>

# Orin
ip -4 -br address
ip route get <PC_IP>
```

随后更新受 Git 管理的现场配置：

- PC `config/teleop.pc.json`：`orin_host=<ORIN_IP>`；
- Orin `config/collection.orin.json`：`allowed_pc_host=<PC_IP>`；
- UDP 端口两端必须同为 `18090`。

先检查 Orin 是否真的启用了主机防火墙，不要因为某台机器使用 UFW 就假定所有机器相同：

```bash
command -v ufw || true
systemctl is-active ufw nftables firewalld
```

当前 dr202 的 Orin 没有启用 UFW、nftables 或 firewalld，Collector 依靠
`allowed_pc_host` 做来源限制，不需要额外安装防火墙。如果新 Orin 已启用 UFW，则必须显式允许
当前 PC 进入 Collector UDP 端口。把尖括号替换为本次发现的值：

```bash
sudo ufw allow in on <ORIN_WIFI_INTERFACE> \
  proto udp \
  from <PC_IP> \
  to <ORIN_IP> \
  port 18090 \
  comment 'ACT teleop PC'

sudo ufw status numbered
```

若未来在当前 dr202 Orin 上启用 UFW，对应实例命令是：

```bash
sudo ufw allow in on wlP1p1s0 \
  proto udp \
  from 192.168.31.219 \
  to 192.168.31.10 \
  port 18090 \
  comment 'ACT teleop PC on dr202'
```

防火墙列出规则不等于数据链已经可用。启动 Collector 和 teleop 后，teleop 的 `ack` 必须持续
递增且 `accepted_acks` 必须大于零；持续出现 `ack=-1` 时禁止开始 Episode，应检查规则是否加在
正确机器、接口名和目标 IP 是否正确，以及 Wi-Fi 是否启用了客户端隔离。测试时必须保证
Collector 在 teleop 整个测试窗口内持续运行；Collector 已退出时出现的 `ack=-1` 不能用于判断网络。

若 Orin 通过 PC 的局域网代理访问外网，还要在 Orin `/etc/environment` 中把代理地址和
`no_proxy` 更新为当前 IP，大小写变量保持一致：

```text
http_proxy="http://<PC_IP>:7897"
https_proxy="http://<PC_IP>:7897"
HTTP_PROXY="http://<PC_IP>:7897"
HTTPS_PROXY="http://<PC_IP>:7897"
no_proxy="localhost,127.0.0.1,::1,<PC_IP>,<ORIN_IP>"
NO_PROXY="localhost,127.0.0.1,::1,<PC_IP>,<ORIN_IP>"
```

同时确认 PC 代理进程监听 LAN 地址而不是仅监听 `127.0.0.1`，PC UFW 只允许当前 Orin 访问
`7897/tcp`。Mihomo/Clash 使用 TUN 时，还要更新或重建其本地局域网段；旧网络网段残留可能把
新网络入站连接重定向到内部端口。验证顺序为：

```bash
# PC：确认代理监听
ss -ltnp | grep ':7897'

# Orin：先验证 TCP 可达，再验证 HTTPS 代理
nc -vz -w 3 <PC_IP> 7897
curl -v --connect-timeout 5 --max-time 15 \
  -x http://<PC_IP>:7897 https://www.google.com/generate_204
```

VS Code Remote-SSH、终端或长期进程不会自动继承修改后的 `/etc/environment`。修改后重新登录
SSH，并重启对应的 VS Code Server/进程，再检查其实际环境。不要在配置或日志中记录代理密码、
令牌或其他密钥。

## 3. 双端采集命令

### 3.1 第一条诊断 Episode（推荐）

当前机器开机后液压即具备动作条件，不存在额外的“液压安全锁定/解锁”软件阶段。第一条短
Episode 从 PC 运行引导脚本，作业区必须无人且急停可立即操作：

在占用 STM32 串口之前，先在 Orin 单独验收两路 RGB 相机。打开现场照明，移除镜头遮挡，并确认
前视相机能覆盖入铲区域、倾倒相机能覆盖卡车/倾倒区域；然后执行：

```bash
cd /home/jetson16/workspace_excavator/excavator-il
conda activate excavator-il-collector
python scripts/diagnose_dual_camera.py \
  --duration-s 5 \
  --output-dir /tmp/icra2027-camera-preflight
```

该诊断只并发打开 `config/collection.orin.json` 中的 `camera_front` 和 `camera_dump`，不会导入串口
库、打开 `/dev/ttyTHS1`、创建网络连接或发送动作。两路先各自丢弃 3 帧完成 UVC/曝光预热，再进入
固定 5 秒测量窗口，避免把首次打开设备的等待时间误判为持续掉帧。只有两路解析到不同物理设备、分辨率均为
`640×480×3`、实测频率均在 `25–35 Hz`、画面没有达到 99.5% 近黑/近白，并且两路均产生有效
JPEG 时才返回 0。还必须人工查看 `/tmp/icra2027-camera-preflight/front.jpg` 和 `dump.jpg`，确认
内容、方向和视野正确；算法通过不能替代这一步。任一路返回失败或画面仍黑暗/被遮挡时，不得开始
零命令 soak、正式 Pilot 或 200 条 campaign，应先修正照明、相机朝向、遮挡或设备映射再复测。

在发动机关闭、所有采集硬件上电且 deadman 保持释放时，先运行 30 秒零命令 soak：

```bash
conda activate excavator-il
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
python scripts/run_zero_command_soak.py
```

持续时间由 `config/guided_episode.pc.json` 的 `runtime.zero_soak_duration_s` 设置。脚本创建并保留
一条明确标记为 `aborted: zero_command_soak_complete` 的诊断 Episode，自动检查全部 STM32 串口
命令和遥测动作回显为零、无有效专家动作、20/10/20/30 Hz 流频率、无 joystick timeout、无解析/
写入/传感器/序号错误。该条禁止执行 `build-steps` 或加入训练集。脚本返回 0 后，才能进入以下
deadman 动作诊断：

```bash
conda activate excavator-il
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
python scripts/collect_guided_episode.py
```

脚本读取 `config/guided_episode.pc.json`。启动后先选择
`RL定位/l`、`人工预定位/y` 或 `直接采集/n`：

该脚本是非正式终端诊断入口：实际创建的 Episode 固定写为
`recording_purpose=diagnostic`，且不携带 `collection_protocol`。即使 Orin 的 `data_root` 指向正式
campaign raw root，权威 inspector 也只会把它计入 `ignored_diagnostics`，不会形成 `malformed`、
占用槽位或进入训练集。不得使用该脚本填充正式 200 条 campaign；正式示教从 3.2 的 WebUI 发起。

- `人工预定位/y`：Collector 先运行一个不创建 Episode 的 teleop。按住 deadman 用双杆调整到 RL
  Follow 的交接位姿附近；X/Y 调整工作装置，Z1/Z2 调整左右履带。完成后六轴全部回中、松开
  deadman，再输入 `完成/c`。脚本确认释放后
  停止该 teleop，预定位过程不落盘、不占 Episode 编号；
- `RL定位/l`：脚本启动配置中指定的唯一 Orin RL Runtime，调用 AiryLidar live
  `Plan DIG → Follow`，Follow 成功并确认终态归零后安全退出 RL Runtime、等待串口释放，再启动
  Collector；
- `直接采集/n` 或直接按 Enter：跳过预定位。

`RL定位/l` 前必须在另一个 PC 终端启动 `live_commissioning` Operator；不要在 Orin 手工启动
`orin_state_sender.py`，否则串口独占预检会失败：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/AiryLidar
source /opt/ros/jazzy/setup.zsh
source ros2_ws/install/setup.zsh
ros2 launch airy_excavator_bringup operator.launch.py \
  profile:=live_commissioning \
  motion_authorization:=ALLOW_LIVE_MACHINE_MOTION \
  orin_host:=192.168.50.2 \
  orin_port:=18083
```

Orin 的 `excavator-orin-runtime/deploy/edge_runtime.remote.json` 必须把
`remote_behavior.allowed_client_host` 配为当前控制链路 PC 地址（有线链路为 `192.168.50.1`）。修改
AiryLidar Mission 目标后必须重启 Operator，使 Planner 与引导客户端加载同一 Mission SHA。

WebUI 的 RL、人工和直接三种模式都从 `rl_preposition.demo_config` 按本条手工填写的
`dig_point_id` 解析实际 Dig 目标并写入 Episode 的 `dig_target_m`；RL 定位返回值还必须与该权威
坐标一致。共享引导流程在 preflight 前和每次 Episode 创建（包括重录）前，都会重新读取 PC 上实际文件，
核对该目标配置文件相对 HEAD 没有修改、HEAD 未变、文件路径与 SHA-256 未变，并把这组事实写入
`episode.json.target_source_provenance`。只有没有 collection protocol 的旧式诊断入口才兼容使用
`episode.dig_target_m`。AiryLidar 仓库中与目标配置无关的笔记、RViz 或代码修改不再阻止单条采集；
目标配置文件本身的未提交修改仍会 fail closed。该字段仅作溯源元数据，不参与 ACT 输入。

RL 定位不切换或重烧 STM32 固件。脚本执行的串口所有权边界是：RL Follow 成功并确认 terminal
velocity zero → 向本轮脚本启动的精确 RL PID 发送 `SIGTERM` → 等待其 `finally` 再次归零并关闭串口
→ 用 `fuser` 确认 `/dev/ttyTHS1` 无占用 → 启动 Collector。Collector 从 v2 telemetry 同步
`command_rx_seq`，再以 manual zero 占用人工模式。不得让 RL Runtime 和 Collector 同时运行。

之后脚本完成：

1. 检查 PC 配置与 Orin SSH；
2. 启动本次专属 Collector，并记录其精确 PID；
3. 创建明确标记为 `recording_purpose=diagnostic` 的 Episode，使 Recorder 进入待命，再启动新的
   teleop；teleop 先通过 0.5 秒本地手柄中位
   稳定门，然后确认 ACK 已接受、无拒绝且 deadman 初始释放；稳定门通过前不创建 UDP socket；
4. Recorder、teleop 和 ACK 门禁全部就绪后才提示等待 deadman；按下后终端显示“记录已开始”，
   此时再操纵双杆 XY；Z1/Z2 在 Recorder 激活期间由 Collector 强制为零；
5. 六轴全部回中并松开 deadman 后，先关闭原始流并把 Episode 原子标记为 `pending_review`，再提示
   输入 `成功/s`、`失败/f` 或 `重录/r`。因此人工分类耗时不会产生尾部 `action_stale`；脚本
   不显示或限制采集时长。`重录`仅删除本轮刚封存的 Episode，随后等待下一次 deadman，并复用
   同一个 Episode 编号；异常或人工 `Ctrl+C` 中止不会删除已产生的动作证据；
6. 最终分类后先停止 teleop 和 Collector，再对本轮保留的每条 Episode 依次执行
   `build-steps`、`validate` 并打印质量报告。

该诊断 Episode 用于验证非零动作、相机、遥测、落盘和因果对齐闭环，不进入 Pilot 训练集。
Collector 或 teleop 的本地日志保存在 `logs/`，该目录不进入 Git。

原始 Episode 只保存在 Orin；根目录由 Orin 的 `config/collection.orin.json` 中 `data_root`
设置。ICRA 2027 双 RGB campaign 固定为
`/home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw`。
历史目录不删除，也不得与该 campaign 混合。PC 的
`config/guided_episode.pc.json` 中
`runtime.log_dir` 只控制引导、teleop 和校验日志位置，不是数据集存储位置。迁移机器时必须显式
检查这两个配置，不能仅复制 shell 历史中的绝对路径。

若研究计划仍希望离线审计既定的 200 条分布，可从 Orin 原始目录只读检查：

```bash
cd /home/jetson16/workspace_excavator/excavator-il
conda activate excavator-il-collector
python scripts/inspect_collection_campaign.py \
  --collection-config config/collection.orin.json --next
```

采满 200 条之前该命令以退出码 2 返回是预期行为；应读取 JSON 中的 `summary` 和
`next_expected_slot`。这是离线计划/审计结果，不会回填 WebUI、不会阻止手工选择标签，也不会自动
启动下一条。只有 `complete_and_valid=true` 时才表示该离线 campaign 已完成且不存在 duplicate、
unplanned 或 malformed Episode。失败/中止尝试保留，但不会占用计划槽位；
`recording_purpose=diagnostic` 的零命令 soak 和 `collect_guided_episode.py` 动作诊断都会计入
`ignored_diagnostics`，不会进入训练集或占用槽位。

### 3.2 PC 本地采集 UI

UI 复用终端引导脚本的运动与 Episode 生命周期实现，不是另一套采集实现。它调用同一个
`GuidedCollectionSupervisor`，再按模式进入 `run_guided_episode()` 或 `run_standalone_teleop()`，
所以 RL/Collector/teleop 串口互斥、deadman
门禁、Episode 封存、重录、校验和异常归零语义与 3.1 一致；但 recording purpose 不同：3.1 的
终端入口固定为 `diagnostic`，WebUI 正式请求携带完整 `collection_protocol` 并记录为
`demonstration`。每次点击只创建一条 Episode；操作者逐条填写任务类型、土壤批次和挖掘点，页面
不执行 campaign 槽位门禁、不显示批次进度，也不自动继续采集。服务强制只监听 loopback，浏览器
之外的页面没有 UI 请求头时不能触发状态改变。

PC 安装并启动：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
conda activate excavator-il
python -m pip install -e '.[teleop,training,test,ui]'
python scripts/check_site_config.py
python scripts/run_collection_ui.py
```

现场配置检查是只读的；应输出当前 `orin_host`、`pc_host`、Joystick/预览/Machine State 端口和
`/dev/ttyTHS1`。任何不一致都应先修正权威 JSON，再启动 UI，不要在命令行临时覆盖成另一套拓扑。

若不希望自动打开浏览器：

```bash
python scripts/run_collection_ui.py --no-browser
```

访问 `http://127.0.0.1:8088/`，选择 `RL 到目标点`、`手工预定位`、`直接采集` 或`仅遥操作`后点击开始。RL
模式还必须选择 `dig_01/02/03` 中的一个点；点位来自
`AiryLidar/mission/config/excavation_demo.json`，所选坐标会写入本条 Episode 的 `dig_target_m`。
RL 模式仍要求先按 3.1 启动 AiryLidar Operator。手工模式完成预定位后必须六轴全部回中、释放 deadman，
再点击页面中的完成按钮。正式记录仍由 deadman 按下开始、释放结束；页面只替代终端中的结果输入。
每条校验完成后可直接开始下一条，页面计数用于跟踪本次 UI 会话内完成数量；正式批次仍以 Orin
数据目录中的完整 Episode 和 quality report 为最终依据。

`仅遥操作`不进入 Recorder 状态，不创建 Episode，也不增加页面 Episode 计数。它只复用 Collector
的串口/相机所有权与 PC teleop 的 deadman 门禁。按住 deadman 操纵，释放立即回零；点击“安全停止”
后依次停止 teleop 和 Collector。相同功能的终端入口为：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
conda activate excavator-il
python scripts/run_guided_teleop.py
```

终端中按 `Ctrl+C` 安全退出。新采集、仅遥操作或混合 Mission 开始申请串口前，会自动回收完整
argv 精确匹配当前配置的遗留 Collector/`orin_state_sender.py`，并等待 `/dev/ttyTHS1` 释放。未知
owner 或独立 ACT Runtime 不会被盲目 `pkill`，仍应先安全停止，避免相机与 STM32 命令源竞争。

若同时启动 AiryLidar live Operator/RViz，Collector 会复用同一条已解析的 STM32 telemetry，按
`config/collection.orin.json.machine_state_udp` 向 PC `18081/UDP` 输出 `machine_state_v1`；PC
状态桥据此继续发布 `/joint_states`。不要为 RViz 另启 `orin_state_sender.py`，它会与 Collector
争抢 `/dev/ttyTHS1`。该只读输出在采集和仅遥操作两种 Collector 工作流中均生效。

相机预览由 Orin Collector 进程内部提供：同一 UVC 相机帧同时交给 Recorder 和容量为 1 的只读
JPEG 缓冲区（Recorder 未激活时只更新预览），UI 不会第二次打开 `/dev/video0`。
`config/collection.orin.json` 的 `camera_preview_http` 当前监听 `0.0.0.0:18092`，HTTP 层只接受
`joystick.allowed_pc_host`。若 Orin 启用 UFW，增加当前有线 PC 的最小 TCP 规则：

```bash
sudo ufw allow in on enP8p1s0 \
  proto tcp from 192.168.50.1 to 192.168.50.2 port 18092 \
  comment 'excavator collector camera preview'
```

Collector 启动前、RL 定位期间和 Collector 退出后，前视区域显示“等待 Collector”属于正常状态。
Collector 运行时，同一只读 HTTP 服务还提供最新 STM32 遥测。UI 显示 4 个关节角（deg）和 3 个
活塞杆伸缩量（mm），并显示接收新鲜度；它不参与训练标签，也不额外占用 `/dev/ttyTHS1`。

### 3.3 RL + ACT 混合 Mission 分段验证

闭环入口仍是 `python scripts/run_collection_ui.py`。开始前：

1. 如需提前录屏，可在 Web UI 点击“启动 RL + RViz”并等待 AiryLidar `live_commissioning` Operator 就绪；直接启动分段/自动 Mission 时，后端也会在 Operator 未就绪时先自动启动。也可沿用
   已经手工启动的 Operator，但两种方式只能选择一种；
2. Orin 不手工启动 `orin_state_sender.py`、Collector 或 ACT Runtime；
3. `jetson16` 执行 `docker info` 不应要求密码；
4. 首次逐段验收保持发动机关闭，确认每个 owner 切换后的串口释放和终态零。

页面选择 DIG 点后点击“开始分段验证”，按顺序检查：

- Mission 开始时 Orin resident owner 一次性打开 `/dev/ttyTHS1` 并在内存中保持 RL ONNX；独立
  ACT Worker 一次性加载 checkpoint、完成 CUDA warmup 并占用 `/dev/video0`，但绝不映射串口；
- `RL 到挖点` 使用 resident RL generation。完成后 owner 依次等待 RL terminal zero 与 ACT 模式
  zero claim 的 STM32 ACK，不退出进程、不释放重开串口；
- 点击 ACT 段；这次明确点击就是本地运动授权。默认完成 130 个 10 Hz step。在最后 20 steps 时，
  PC 后台准备 DUMP 轨迹；
- 点击“前往倾倒并倾倒”时先完成 ACT terminal zero → RL target zero 的 ACK 交接，再激活准备好的
  DUMP 轨迹。轨迹未及时就绪、已过期或当前起点不兼容时保持归零，回退到普通 live Plan；
- 点击“RL 返回挖点”，继续复用同一 resident RL Runtime 返回本轮目标，不发生模型或容器冷启动。

整个流程只有 resident owner 能写串口；策略切换只增加 `control_generation` 并更换候选动作语义。
PC 每 0.4 秒续约一次 1.5 秒 Mission lease。Web UI/PC 退出或控制网线断开后，Orin 会 terminal
disarm、等待最终零命令 ACK 并释放串口；点击“安全停止”时先禁止新的续约，再执行相同终态路径。
分段全部通过
后才能使用“自动装车”。自动模式可选择 1～9 铲；页面选中的 DIG 点是第一铲，后续按 Mission
配置中的 DIG 顺序循环。例如从 `dig_02` 开始的 5 铲依次为
`dig_02 → dig_03 → dig_01 → dig_02 → dig_03`。RL 返回阶段直接去下一铲的交接位姿，并在途中
保持常驻 ACT Worker ready；最后一铲仍返回该铲使用的挖点。任一阶段失败即停止，不会跳过故障
继续下一铲。ACT step 上限集中在
`config/hybrid_mission.pc.json`，不得通过页面临时随意扩大。该有界 step 规则只是当前实验的
最小完成条件，不代表任务成功识别；真实土壤验收仍记录是否完成挖掘、是否带土和最终交接位姿。
原生 RViz 不作为 iframe 嵌入，Web UI 也不显示 RViz/Foxglove 扩展占位。
需要三维状态或录屏时，使用页面“启动 RL + RViz”打开的原生 RViz 窗口。

更新 `excavator-il` Python 代码后，Orin 必须用
`docker/act-inference.incremental.Dockerfile` 重建 `excavator-act-inference:jp72-pytorch261`；Git
同步不会修改已有镜像。精确构建与 import 核验命令见仓库 README 的“RL + ACT 混合 Mission”节。

Wi-Fi 局域网避免了公网路由，但不是实时总线：无线信道争用/重传/省电、驱动与内核队列，以及
Orin 进程调度都可能把若干 20 Hz 包延后后再成批交付。`episode_0004` 中 PC 采样间隔仍约
49.9 ms，而 Orin 应用层一次接收间隔达到 167 ms，随后出现约 16 ms 的成批到达；现有证据只能
确认延迟发生在 PC 采样之后，不能把无线传输与 Orin 调度精确分开。150 ms 超时继续保持安全
门槛。离线构建对每个 timeout 要求可定位的安全回零和连续 10 个有效手柄包作为恢复证据，排除
恢复窗口并生成 Training Segment；事件无法定位或恢复时 `validate` 拒绝该条。不要通过插值、
沿用旧动作或直接忽略 timeout 计数来挽救数据。

### 3.4 非正式链路诊断命令

正式 demonstration Episode 应从 WebUI 逐条发起，使操作者填写的 `dig_point_id` 与 AiryLidar
权威目标坐标在创建 Episode 前一并校验。WebUI 不读取或强制 200 条 campaign 槽位。
`collect_guided_episode.py` 和下面的分端命令只用于诊断 Collector/teleop/Recorder，显式标记为
`diagnostic`，不会进入训练集。

Orin 终端 1——启动 Collector：

```bash
conda activate excavator-il-collector
cd /home/jetson16/workspace_excavator/excavator-il
excavator-il collect --config config/collection.orin.json
```

PC——启动双手柄 20 Hz 发送端：

```bash
conda activate excavator-il
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
excavator-il teleop --config config/teleop.pc.json --print-every 20
```

Orin 终端 2——记录一条诊断 Episode：

```bash
conda activate excavator-il-collector
cd /home/jetson16/workspace_excavator/excavator-il

excavator-il episode --config config/collection.orin.json start \
  --task ExecuteDig --operator zhaoshuai \
  --dig-target-m 0.8 0.0 -0.2 \
  --material-id soil_01 \
  --recording-purpose diagnostic

# 人工完成一次完整挖掘后：
excavator-il episode --config config/collection.orin.json stop --success
```

失败或中止：

```bash
excavator-il episode --config config/collection.orin.json stop \
  --failure-reason bucket_empty

excavator-il episode --config config/collection.orin.json abort \
  --reason emergency_stop
```

## 4. 生成样本和质量校验

在保存 Episode 的机器上执行：

```bash
EPISODE=/home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw/episode_0001

excavator-il build-steps "$EPISODE"
excavator-il validate "$EPISODE"
python -m json.tool "$EPISODE/quality_report.json"
```

WebUI/引导脚本在上述两步通过后会自动调用 `record-collection-run`，生成不可变
`experiment_run.v1`。单独复核或补录证据时可显式执行同一幂等命令：

```bash
excavator-il record-collection-run "$EPISODE" \
  --config config/collection_evidence.orin.json

python scripts/manage_experiment_run.py verify \
  --root /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/experiment-runs \
  --run-id collection_episode_0001
```

证据写入失败会让正式流程失败，但不会删除或改写 raw Episode。修复配置后可重试同一命令；若
Episode、质量报告或 TaskContext 已发生漂移，重试会拒绝，而不是把新内容嫁接到旧 Run。
artifact 使用 `experiment_run_artifact.v2`：`source_path` 仅保留原始来源，不能作为可验证证据；
注册时先在 Run 自己管理的 `artifact_snapshots/` 中生成不可变内容快照，拷贝前后源指纹与快照指纹
必须一致，`snapshot_path` 才供 finalize 和统一 Evaluation Harness 使用。实现会优先尝试独立 CoW
reflink，并把实际结果记为 `snapshot_method=reflink`；不支持 reflink 或跨文件系统时使用标准库逐字节
copy 并明确记录 `snapshot_method=copy`，不会假装节省了空间。目录中不同文件采用不同方法时记录为
`mixed`。

fallback copy 意味着 raw Episode 和 Experiment Run 证据快照可能占用接近两份容量。正式 200 条
采集前，先用 2--4 条 Pilot 的 `du -sh` 估算单条上界，用 campaign inspector 确认剩余槽位，再用
`df -h` 确认 raw、快照和临时复制峰值均有余量：

```bash
python scripts/inspect_collection_campaign.py \
  --collection-config config/collection.orin.json --next
du -sh /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw
du -sh /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/experiment-runs
df -h /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1
```

正式配置还要求 tracked 的 `config/icra2027_collection_campaign_provenance.json`，它冻结本批次
AiryLidar 目标配置 commit/path/SHA、F407 固件 commit、共享机器配置 SHA 与三个 Dig 坐标。每条 Run
发布前会把该文件纳入配置快照，并逐项对照 Episode 中本次真正使用的 target source provenance；
本地采集仓库存在未提交修改时也会 fail closed。`quality_report` 是必需的 integrity-only evidence，
统一 Evaluation Harness 会先验证其内容哈希，但不会由此伪造额外性能指标。

`episode.json` 为 `pending_review` 时不得构建或校验；必须先分类为 `complete`、`failed` 或
`aborted`。`rejection_reasons.action_stale` 表示某个 10 Hz 新状态找不到不晚于它且年龄不超过
100 ms 的有效 deadman 动作。旧的 `episode_0004` 在松开 deadman 后仍等待约 6.2 秒人工输入，
期间 Recorder 尚未关闭，因此产生 61 个尾部 `action_stale`；新引导流程先封存、后询问结果。

批量校验时逐条处理，任一命令失败都不应把该 Episode 加入训练集：

```bash
for episode in /home/jetson16/workspace_excavator/data/icra2027-dual-rgb-campaign-v1/raw/episode_*; do
  excavator-il build-steps "$episode" || break
  excavator-il validate "$episode" || break
done
```

## 5. 转换 LeRobotDataset

先把通过验证的成功 Episode 从 Orin 只读同步到训练 PC。此次采集包含两个不同任务，必须转换为
两个互不混合的数据集并分别训练模型：

- `episode_0001` 至 `episode_0110` 是仅挖掘任务，只使用 `front`；
- `episode_0111` 至 `episode_0145` 是挖掘、运转、倾倒任务，使用 `front,dump`。

转换前必须根据 `episode.json` 与 `quality_report.json` 只选择 `complete + success=true` 的 Episode。
不得再用 `"$RAW"/episode_*` 直接转换整个目录。当前原始标签不能可靠区分两类任务，因此这次转换
在完成质量筛选后使用显式 `--task-variant-override` 修正派生数据集标签；原始 Episode 保持只读。

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
conda activate excavator-il

RAW=data/raw/icra2027-dual-rgb-campaign-v1

# EPISODES 数组必须来自本次质量审计结果；以下省略具体展开。
# 仅挖掘：104 条通过质量门禁的 Episode，仅保留前视相机。
excavator-il convert "${DIG_ONLY_EPISODES[@]}" \
  --output-root data/lerobot/icra2027_dig_only_front_full \
  --repo-id local/icra2027_dig_only_front_full --fps 10 \
  --camera-roles front \
  --task-variant-override dig_only

excavator-il prepare-training-split \
  --dataset-root data/lerobot/icra2027_dig_only_front_full \
  --repo-id local/icra2027_dig_only_front_full \
  --output data/lerobot/icra2027_dig_only_front_split.json \
  --train-ratio 0.8 --seed 2027 --grouping episode

excavator-il materialize-training-split \
  --manifest data/lerobot/icra2027_dig_only_front_split.json \
  --output-root data/lerobot/icra2027_dig_only_front_split

# 挖掘、运转、倾倒：35 条通过质量门禁的 Episode，保留双路 RGB。
excavator-il convert "${TRANSPORT_DUMP_EPISODES[@]}" \
  --output-root data/lerobot/icra2027_transport_dump_dual_rgb_full \
  --repo-id local/icra2027_transport_dump_dual_rgb_full --fps 10 \
  --camera-roles front,dump \
  --task-variant-override dig_transport_dump

excavator-il prepare-training-split \
  --dataset-root data/lerobot/icra2027_transport_dump_dual_rgb_full \
  --repo-id local/icra2027_transport_dump_dual_rgb_full \
  --output data/lerobot/icra2027_transport_dump_dual_rgb_split.json \
  --train-ratio 0.8 --seed 2027 --grouping episode

excavator-il materialize-training-split \
  --manifest data/lerobot/icra2027_transport_dump_dual_rgb_split.json \
  --output-root data/lerobot/icra2027_transport_dump_dual_rgb_split
```

输出目录必须是新的空目录；不要在已有 LeRobotDataset 上重复运行转换。默认 `auto` 只有在全部
输入均为同一双相机契约时才保留双路；混合 v1/v2 会明确失败。本批数据的 block/zone 标记不是
训练分组依据，因此显式使用 `--grouping episode`，保证同一 parent Episode 不跨 train/validation；
materialize 会重算数据指纹并拒绝漂移或泄漏。LeRobot 原生 Episode 边界会
限制 ACT future-action delta indices，并通过 `action_is_pad` 屏蔽片段末端越界动作。

### 5.1 重复数据的离线管线验证

正式采集前可以将一条已经通过 `build-steps` 和 `validate` 的测试 Episode 复制为唯一 ID，专门
检查 10-Episode 转换、ACT checkpoint 和推理链路：

```bash
excavator-il synthesize-episodes data/raw/pipeline_source/episode_0004 \
  --output-root data/raw/synthetic_episode_0004_x10 --count 10

for episode in data/raw/synthetic_episode_0004_x10/synthetic_episode_*; do
  excavator-il build-steps "$episode" || break
  excavator-il validate "$episode" || break
done

excavator-il convert data/raw/synthetic_episode_0004_x10/synthetic_episode_* \
  --output-root data/lerobot/synthetic_episode_0004_x10 \
  --repo-id local/synthetic_episode_0004_x10 --fps 10 --allow-synthetic
```

每个副本的 `episode.json` 都包含 `synthetic_provenance` 且
`training_eligible=false`。转换器默认拒绝它们，只有显式 `--allow-synthetic` 才能用于离线
管线测试，并在 LeRobotDataset 根目录写入 `pipeline_validation.json`。转换器拒绝真实与合成
Episode 混合。十份完全相同的数据不是十次独立示教，不能用于评价收敛、泛化、成功率或正式
checkpoint，也不得与 Pilot 数据混合。原生 `lerobot-train` 不读取该门禁文件，启动训练前仍需
人工确认数据集路径。图像通过硬链接共享，只允许读取，禁止原地编辑。

以下是 16 GB 显存训练机验证过的 3-step checkpoint smoke 命令。它刻意使用小模型，只验证读取
4850 帧、反向传播、保存 checkpoint，不用于评价 loss：

```bash
lerobot-train \
  --dataset.repo_id=local/synthetic_episode_0004_x10 \
  --dataset.root=data/lerobot/synthetic_episode_0004_x10 \
  --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=20 --policy.n_action_steps=10 \
  --policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=null \
  --policy.dim_model=64 --policy.n_heads=4 --policy.dim_feedforward=128 \
  --policy.n_encoder_layers=1 --policy.n_decoder_layers=1 \
  --policy.latent_dim=16 --policy.n_vae_encoder_layers=1 \
  --output_dir=outputs/act_synthetic_episode_0004_x10_smoke \
  --job_name=act_synthetic_episode_0004_x10_smoke \
  --batch_size=2 --num_workers=0 --steps=3 --log_freq=1 \
  --save_checkpoint=true --save_freq=3 --eval_freq=0 --wandb.enable=false

excavator-il smoke-infer \
  outputs/act_synthetic_episode_0004_x10_smoke/checkpoints/last/pretrained_model \
  --dataset-root data/lerobot/synthetic_episode_0004_x10 \
  --repo-id local/synthetic_episode_0004_x10 --sample-index 0 --device cuda
```

## 6. ACT 训练

先验证特征契约和 ACT 安装：

```bash
excavator-il smoke-train
```

使用 LeRobot 0.5.2 在训练 PC 上启动 ACT。训练必须读取 materialized train，而不是转换后的
全量数据集；repo ID 由各自的 `split_provenance.json` 产生，不手写猜测。此次必须训练两个互相
独立的模型，不能把两类 Episode 合并后只训练一个 checkpoint。

推荐使用已经做过数据指纹、CUDA、权重缓存和输出目录门禁的训练脚本。脚本支持单独训练一个模型，
不会因为仅挖掘模型已经完成而阻止完整动作模型启动。原始 split 的相机 dtype 是 `image`，JPEG 实际
嵌在 Parquet 中；在这种数据上设置 `--dataset.video_backend=pyav` 不会自动变成视频读取。先生成
新的、带 source/output 指纹的视频派生 split，再测量输入吞吐：

```bash
PYTHONPATH=src python scripts/derive_video_training_split.py \
  data/lerobot/icra2027_transport_dump_dual_rgb_split \
  data/lerobot/icra2027_transport_dump_dual_rgb_split_video

FULL_VIDEO_SPLIT=data/lerobot/icra2027_transport_dump_dual_rgb_split_video
FULL_VIDEO_REPO=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["train_repo_id"])' \
  "$FULL_VIDEO_SPLIT/split_provenance.json")
PYTHONPATH=src python scripts/benchmark_act_training_input.py \
  "$FULL_VIDEO_SPLIT/train" "$FULL_VIDEO_REPO" \
  --batch-size 4 --num-workers 2

bash scripts/train_icra2027_two_act_models.sh \
  --model full --dataset-format video --preflight-only
bash scripts/train_icra2027_two_act_models.sh \
  --model full --dataset-format video
```

若训练在至少保存过一个 checkpoint 后被中断，使用同一模型、数据表示、seed 和 batch 环境变量续训，
不要删除输出目录或训练计划，也不要重新执行一份同名新训练：

```bash
bash scripts/train_icra2027_two_act_models.sh \
  --model full --dataset-format video --resume
```

`--resume` 只允许单模型，严格要求 `checkpoints/last` 与原训练计划同时存在，并由 LeRobot 恢复
模型、优化器、scheduler、随机数状态和已完成 step；实时输出追加到原日志。

终端实时显示完整训练输出并同时写日志。默认样本曝光预算仍为：仅挖掘 800,000、挖掘—运转—倾倒
600,000。步数由 `ceil(sample_budget / batch_size)` 计算，因此完整动作模型从 batch 2 提升为 batch 4
后训练 150,000 updates，仍约 104.6 等效 epoch，并不是把旧 300,000 updates 原样放大成双倍训练量。
checkpoint 间隔也按 20,000 个已见样本换算，完整动作模型每 5,000 updates 保存一次。默认 workers
仍为 2，避免当前 32 GB RAM / 8 GB swap 在大 Parquet 和双路解码下继续增加内存压力。seed 为 2027。
网络和优化器显式冻结为上一轮成功配置：ResNet18 ImageNet 初始化、
`dim_model=512`、8 heads、4 层 encoder、1 层 decoder、latent 32、KL 10，以及 AdamW
`lr=1e-5` / `weight_decay=1e-4`；本轮不引入未经真机验证的新 scheduler 或图像增强。训练完成后
脚本会自动在各自严格隔离的 validation Episode 上扫描
全部数值 checkpoint，按有限、`[-1,1]` 安全门禁后的最低 `deployment_prior_l1` 选优，而不是默认
采用最后一步。输入优化后的预计时长必须以 `benchmark_act_training_input.py` 和正式训练前 500--1000
updates 的 `samples/s` 为准，不再拿不同 batch 的 `steps/s` 直接比较。脚本拒绝覆盖任何已有正式输出
目录；完整动作的新 job 名包含 `video_b4`，不会覆盖之前中断的 image/batch2 输出。下面保留展开命令
供审计和单独重现实验使用。

2026-08-27 在 RTX 5070 Ti 主机上，用完整动作 train partition、batch 4、workers 2、warmup 20、
测量 200 batches 的纯输入热缓存 A/B 结果为：Parquet 内嵌 JPEG 约 76.5 samples/s，H.264/GOP 2
约 128.4 samples/s（约 `1.68x`）。该值只证明解码路径改善，不包含 ACT 前向/反向，不能直接当作正式
训练吞吐或论文性能结果；正式启动前仍需做有限 updates 端到端短跑并检查 loss、gradient、GPU/RAM。

评估的完整实时输出写入两个 `*_checkpoint_evaluation.log`，纯 JSON 结果写入对应
`*_checkpoint_evaluation.json`；其中 `selected_checkpoint` 才是后续离线回载和真机门禁的候选，
不是 `checkpoints/last`。

仅挖掘模型沿用已经在真机验证过的 swing-zero 动作约束，并且只读取 front：

```bash
excavator-il derive-zero-swing-split \
  --source-root data/lerobot/icra2027_dig_only_front_split \
  --output-root data/lerobot/icra2027_dig_only_front_split_swing_zero \
  --repo-suffix swing_zero

SPLIT=data/lerobot/icra2027_dig_only_front_split_swing_zero
TRAIN_REPO=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["train_repo_id"])' \
  "$SPLIT/split_provenance.json")

lerobot-train \
  --dataset.repo_id="$TRAIN_REPO" \
  --dataset.root="$SPLIT/train" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --output_dir=outputs/icra2027_dig_only_front_swing_zero_seed2027 \
  --job_name=icra2027_dig_only_front_swing_zero_seed2027 \
  --batch_size=2 \
  --num_workers=2 \
  --steps=400000 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=100 \
  --seed=2027 \
  --wandb.enable=false
```

挖掘、运转、倾倒模型保留 swing 标签并同时读取 front 与 dump：

```bash
SPLIT=data/lerobot/icra2027_transport_dump_dual_rgb_split_video
TRAIN_REPO=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["train_repo_id"])' \
  "$SPLIT/split_provenance.json")

lerobot-train \
  --dataset.repo_id="$TRAIN_REPO" \
  --dataset.root="$SPLIT/train" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --output_dir=outputs/icra2027_transport_dump_dual_rgb_video_b4_seed2027 \
  --job_name=icra2027_transport_dump_dual_rgb_video_b4_seed2027 \
  --batch_size=4 \
  --num_workers=2 \
  --steps=150000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --log_freq=100 \
  --seed=2027 \
  --wandb.enable=false
```

LeRobot 0.5.2 的 ACT 默认要求 Hub repo；本地训练必须显式设置
`--policy.push_to_hub=false`。训练完成后分别用各自数据集验证预处理、checkpoint 回载、动作块
推理和反归一化，不能交叉使用 repo ID：

```bash
DIG_SPLIT=data/lerobot/icra2027_dig_only_front_split_swing_zero
DIG_TRAIN_REPO=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["train_repo_id"])' \
  "$DIG_SPLIT/split_provenance.json")
excavator-il smoke-infer \
  outputs/icra2027_dig_only_front_swing_zero_seed2027/checkpoints/last/pretrained_model \
  --dataset-root "$DIG_SPLIT/train" \
  --repo-id "$DIG_TRAIN_REPO" \
  --sample-index 0 --device cuda

FULL_SPLIT=data/lerobot/icra2027_transport_dump_dual_rgb_split_video
FULL_TRAIN_REPO=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["train_repo_id"])' \
  "$FULL_SPLIT/split_provenance.json")
excavator-il smoke-infer \
  outputs/icra2027_transport_dump_dual_rgb_video_b4_seed2027/checkpoints/last/pretrained_model \
  --dataset-root "$FULL_SPLIT/train" \
  --repo-id "$FULL_TRAIN_REPO" \
  --sample-index 0 --device cuda
```

预期 `predicted_chunk_shape` 为 `[1, 20, 4]`、`action_dim` 为 4 且
`all_finite=true`。这一步只验证离线推理接口，不授权向 Orin 或 STM32 发送模型动作。
双相机模型投入真机前还必须把 `dump` 接入 Resident ACT 的因果观测缓冲；当前常驻真机入口只构造
`front` 观测，因此双相机离线 checkpoint 通过不等于已经具备真机部署条件。

完整动作默认使用已通过 preflight 后的 `--batch_size=4`；若真实短跑发生 CUDA OOM，先回退到 2，
同时让脚本按同一 600,000 样本预算自动恢复为 300,000 updates，不能只改 batch 而改变曝光预算。
长训练预算不代表应固定选最后 checkpoint；
checkpoint 选择只使用 materialized validation partition。训练开始前创建 `run_kind=training` 的
Experiment Run，记录 GPU、LeRobot 版本、split manifest/provenance、配置、代码 commit、随机种子、
checkpoint 和完整命令。正式实验至少运行 3 个预先冻结的随机种子；第一轮只需跑通一个种子。

完成态 Experiment Runs 使用统一 Harness 生成机器可读 JSON 与论文表格 CSV：

```bash
python scripts/evaluate_experiment_runs.py \
  EvaluationReport/experiment_runs/runs/<run_id_1> \
  EvaluationReport/experiment_runs/runs/<run_id_2> \
  --aggregate-mode homogeneous \
  --output-dir EvaluationReport/evaluations/<evaluation_id>
```

`training_internal` 与 `held_out_experiment` 不能混合聚合。论文最终任务结论只使用后者；代码或
配置仓库为 dirty 的 held-out Run 会被 Harness 拒绝，不能通过手工改 CSV 绕过。

## 7. 回归命令

`excavator-il`：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
conda activate excavator-il
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_cov \
  --cov=excavator_il --cov-report=term-missing
```

F407：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/F407
data_celect/Tests/run_host_tests.sh
```

Orin Runtime：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-orin-runtime
python3 -m unittest discover -s tests -v
python3 -m py_compile orin_state_sender.py orin_csv_replay.py edge_runtime/*.py
```

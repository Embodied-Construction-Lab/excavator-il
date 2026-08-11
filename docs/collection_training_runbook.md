# ACT 示教采集与训练运行手册

本文档描述 `excavator-il` 当前实现对应的下一阶段工作。权威字段、单位和时间语义见
[`data_contract_v1.md`](data_contract_v1.md)。

## 1. 阶段目标

### 阶段 A：设备与链路验收

1. 在 CubeIDE 中重新构建并烧录 `F407/data_celect`。
2. 确认双手柄、前视 UVC 相机和 STM32 串口设备名。
3. 完成 3 条 10～20 秒的短 Episode：成功、失败和人工中止各一条。
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

### 阶段 C：ACT v1 数据集

- 目标采集 60～100 条质量合格的成功 Episode；
- 额外保留至少 10 条失败或人工中止 Episode，但 ACT v1 暂不用于行为克隆训练；
- 操作员、初始姿态、挖掘点和材料条件尽量覆盖真实部署范围；
- 按 Episode 划分 80% 训练、10% 验证、10% 测试，禁止按帧随机拆分；
- 同一连续采集批次尽量只进入一个集合，降低相邻 Episode 的数据泄漏。

### 阶段 D：训练与离线评估

1. 分别转换 train/val/test 三个 LeRobotDataset。
2. 使用 train 数据训练 ACT，val 数据选择 checkpoint。
3. 在 test Episode 上报告动作 MAE、动作方向一致率和动作序列平滑度。
4. 保存数据清单、Git commit、配置、随机种子和 checkpoint。

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
同时核对 PC/Orin IP、串口、相机设备和 provenance 字段。

Orin：

```bash
cd ~/workspace/excavator-il
conda env create -f environment.orin.yml
conda activate excavator-il-collector
python -m pip install -e '.[collector]'
```

正式采集期间不能同时运行 `excavator-orin-runtime/orin_state_sender.py`，Collector 必须独占
STM32 串口。

### 网络、代理与防火墙迁移清单

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

Orin 终端 1——启动 Collector：

```bash
conda activate excavator-il-collector
cd ~/workspace/excavator-il
excavator-il collect --config config/collection.orin.json
```

PC——启动双手柄 20 Hz 发送端：

```bash
conda activate excavator-il
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
excavator-il teleop --config config/teleop.pc.json --print-every 20
```

Orin 终端 2——记录一条成功示教：

```bash
conda activate excavator-il-collector
cd ~/workspace/excavator-il

excavator-il episode --config config/collection.orin.json start \
  --task ExecuteDig --operator operator_01 \
  --dig-target-m 0.8 0.0 -0.2 \
  --material-id soil_01

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
EPISODE=~/excavator-data/raw/episode_0001

excavator-il build-steps "$EPISODE"
excavator-il validate "$EPISODE"
python -m json.tool "$EPISODE/quality_report.json"
```

批量校验时逐条处理，任一命令失败都不应把该 Episode 加入训练集：

```bash
for episode in ~/excavator-data/raw/episode_*; do
  excavator-il build-steps "$episode" || break
  excavator-il validate "$episode" || break
done
```

## 5. 转换 LeRobotDataset

先把通过验证的 Episode 从 Orin 同步到训练 PC，再显式列出每个集合的 Episode。以下编号仅为示例：

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
conda activate excavator-il

excavator-il convert \
  data/raw/episode_0001 data/raw/episode_0002 data/raw/episode_0003 \
  --output-root data/lerobot/excavator_rgb_v1_train \
  --repo-id local/excavator_rgb_v1_train --fps 10

excavator-il convert \
  data/raw/episode_0004 \
  --output-root data/lerobot/excavator_rgb_v1_val \
  --repo-id local/excavator_rgb_v1_val --fps 10

excavator-il convert \
  data/raw/episode_0005 \
  --output-root data/lerobot/excavator_rgb_v1_test \
  --repo-id local/excavator_rgb_v1_test --fps 10
```

输出目录必须是新的空目录；不要在已有 LeRobotDataset 上重复运行转换。

## 6. ACT 训练

先验证特征契约和 ACT 安装：

```bash
excavator-il smoke-train
```

使用 LeRobot 0.5.2 在训练 PC 上启动 ACT v1：

```bash
lerobot-train \
  --dataset.repo_id=local/excavator_rgb_v1_train \
  --dataset.root=data/lerobot/excavator_rgb_v1_train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --output_dir=outputs/act_excavator_rgb_v1 \
  --job_name=act_excavator_rgb_v1 \
  --batch_size=8 \
  --steps=100000 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false
```

显存不足时先把 `--batch_size` 降为 4 或 2。训练开始前记录 GPU、LeRobot 版本、数据集清单、
三个仓库的 commit 和完整命令。正式实验至少运行 3 个随机种子，第一轮只需跑通一个种子。

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

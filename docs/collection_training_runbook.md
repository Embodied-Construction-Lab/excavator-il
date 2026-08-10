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
python -m pip install -e '.[teleop,training,test]'
excavator-il list-joysticks
```

将手柄 GUID 写入 `config/teleop.pc.json` 和 Orin 上的
`config/collection.orin.json`。同时核对 PC/Orin IP、串口、相机设备和 provenance 字段。

Orin：

```bash
cd ~/workspace/excavator-il
conda env create -f environment.orin.yml
conda activate excavator-il-collector
python -m pip install -e '.[collector]'
```

正式采集期间不能同时运行 `excavator-orin-runtime/orin_state_sender.py`，Collector 必须独占
STM32 串口。

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

# ACT 人工训练操作手册

本文给出训练 PC 上从 LeRobotDataset 检查、ACT 训练、监控、checkpoint 回载到断点续训的完整
命令。命令已于 2026-08-12 在以下环境验证：

- GPU：NVIDIA GeForce RTX 5070 Ti，16 GB；
- Python 3.12.13、PyTorch 2.10.0+cu128、LeRobot 0.5.2；
- 输入：前视 RGB `3×480×640`、11 维状态；
- 输出：4 维动作 `[boom, stick, bucket, swing]`；
- ACT 频率 10 Hz，`chunk_size=20`，`n_action_steps=10`。

本文只涉及离线训练和推理，不启动 Orin Collector，不打开 STM32 串口，也不向真机发送模型动作。

## 1. 进入环境

```bash
cd /home/zhaoshuai/workspace_uinty/RL_prj/excavator-il
conda activate excavator-il

python --version
python -c 'import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)'
nvidia-smi
df -h .
```

预期 Python 为 3.10～3.12，`torch.cuda.is_available()` 为 `True`，磁盘应有足够空间。标准 ACT
每个完整 checkpoint（模型、optimizer 和状态）约占 580 MB；频繁保存会明显增加磁盘占用。

## 2. 确认数据集类型

当前用于管线验证的数据集为：

```text
data/lerobot/synthetic_episode_0004_x10
```

先检查合成数据门禁：

```bash
python -m json.tool \
  data/lerobot/synthetic_episode_0004_x10/pipeline_validation.json
```

应看到：

```json
{
  "contains_synthetic_episodes": true,
  "training_eligible": false
}
```

这十条数据是同一条 `episode_0004` 的精确副本，只能验证训练软件链。它们不能用于评价泛化、
成功率或选择部署 checkpoint。原生 `lerobot-train` 不读取上述门禁文件，因此操作人必须主动核对
数据集路径。

正式训练时必须换成由不同示教组成、按 parent Episode 划分的 train/val/test 数据集，并确认
数据集根目录不存在 `pipeline_validation.json`。禁止把合成数据与 Pilot 混合。

### 2.1 按 parent Episode 固定训练/验证集

转换完成后，先生成可复用拆分清单。拆分依据是 `source.episode_id`，因此同一条原始示教因链路
故障恢复切出的多个 LeRobot Episode 一定留在同一侧，不会形成数据泄漏：

```bash
excavator-il prepare-training-split \
  --dataset-root data/lerobot/excavator_rgb_v1 \
  --repo-id local/excavator_rgb_v1 \
  --output data/lerobot/split_manifests/excavator_rgb_v1.json \
  --train-ratio 0.8 \
  --seed 1000
```

若输出文件已存在，命令只接受完全相同的 dataset、seed、ratio 和 Episode 映射；参数漂移会失败，
不能静默重写拆分。`training_eligible=false` 的管线合成数据也会被拒绝。

再物化两个独立 LeRobotDataset：

```bash
excavator-il materialize-training-split \
  --manifest data/lerobot/split_manifests/excavator_rgb_v1.json \
  --output-root data/lerobot/excavator_rgb_v1_split
```

输出为 `..._split/train` 和 `..._split/validation`。LeRobot 会按各自 Episode 重新聚合
`meta/stats.json`，正式训练只能使用 train 的统计量。该过程需要重编码视频，因此占用额外磁盘；
它通过同目录暂存和原子改名发布，中断不会留下名称正常但内容不完整的数据集。

拆分清单不要写入训练的 `--output_dir`；LeRobot 要求首次训练时该目录不存在。

## 3. 首次标准 ACT 训练

输出目录必须不存在。以下命令使用标准规模 ACT 主体，关闭 Hub 上传和网络预训练权重，适合先做
100-step 本地闭环：

```bash
set -o pipefail

lerobot-train \
  --dataset.repo_id=local/synthetic_episode_0004_x10 \
  --dataset.root=data/lerobot/synthetic_episode_0004_x10 \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.pretrained_backbone_weights=null \
  --output_dir=outputs/act_synthetic_episode_0004_x10_standard_smoke \
  --job_name=act_synthetic_episode_0004_x10_standard_smoke \
  --batch_size=2 \
  --num_workers=2 \
  --steps=100 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=50 \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed=1000 \
  2>&1 | tee logs/act_synthetic_standard_100step.log
```

关键启动输出应包括：

```text
dataset.num_frames=4850
dataset.num_episodes=10
num_learnable_params=51559300
Effective batch size: 2 x 1 = 2
```

`--steps` 是目标总更新次数，不是 epoch 数。100 step、batch 2 只读取约 200 个训练样本，不等于
完整遍历 4850 帧。正式训练步数必须根据真实数据和验证集表现确定，不能照搬本例。

### 3.1 五条真实示教的 20-epoch 基线

2026-08-12 的相机新视角基线由五条相互独立、均通过 `build-steps` 和 `validate` 的真实示教组成：

```text
episode_0005 ... episode_0009
```

从 Orin 同步后，先把五条原始 Episode 放在
`data/raw/camera_angle_ep0005_0009/`，再转换并回载检查：

```bash
excavator-il convert \
  data/raw/camera_angle_ep0005_0009/episode_0005 \
  data/raw/camera_angle_ep0005_0009/episode_0006 \
  data/raw/camera_angle_ep0005_0009/episode_0007 \
  data/raw/camera_angle_ep0005_0009/episode_0008 \
  data/raw/camera_angle_ep0005_0009/episode_0009 \
  --output-root data/lerobot/camera_angle_ep0005_0009 \
  --repo-id local/camera_angle_ep0005_0009 \
  --fps 10

python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local/camera_angle_ep0005_0009",
    root="data/lerobot/camera_angle_ep0005_0009",
)
assert dataset.num_episodes == 5
assert dataset.num_frames == 623
assert dataset.features["observation.state"]["shape"] == (11,)
assert dataset.features["action"]["shape"] == (4,)
assert dataset.features["action"]["names"] == [
    "action_boom", "action_stick", "action_bucket", "action_swing"
]
print("dataset gate passed")
PY
```

输出目录必须事先不存在。转换进程退出后必须重新加载数据集并通过上述断言，不能根据转换期间的
中间文件或目录大小推断完成。

不要为了把 Episode 数量凑成 20 或 50 而复制这五条数据。复制不会增加信息量，还会破坏正式
数据集的 provenance。下面的 6250 step、batch 2 会读取约 12500 个样本，相当于对 623 帧执行
约 20.1 次等效遍历：

```bash
set -o pipefail

lerobot-train \
  --dataset.repo_id=local/camera_angle_ep0005_0009 \
  --dataset.root=data/lerobot/camera_angle_ep0005_0009 \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.pretrained_backbone_weights=null \
  --output_dir=outputs/act_camera_angle_ep0005_0009_20epochs \
  --job_name=act_camera_angle_ep0005_0009_20epochs \
  --batch_size=2 \
  --num_workers=2 \
  --steps=6250 \
  --log_freq=50 \
  --save_checkpoint=true \
  --save_freq=1250 \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed=1000 \
  2>&1 | tee logs/act_camera_angle_ep0005_0009_20epochs.log
```

这个基线使用全部五条数据训练，只适合验证训练和离线推理，不提供独立验证/测试集，也不能据此
评估泛化或放行真机闭环。正式部署候选必须增加不同初始位姿、完整四轴动作和独立留出 Episode。
本次实测最终 checkpoint 虽完成训练，但完整样本扫描发现部分 action chunk 超出 `[-1,1]`，已被
推理门禁拒绝；详见
`../../EvaluationReport/2026-08-12_act_camera_angle_real_episode_training.md`。不能只用一个样本通过
`smoke-infer` 就宣布 checkpoint 可部署，也不能在 STM32 中静默截断越界输出来绕过门禁。

## 4. 实时监控

另开一个终端监控 GPU：

```bash
watch -n 1 nvidia-smi
```

查看训练关键事件：

```bash
grep -E 'step:|Checkpoint policy|End of training|Traceback|Error' \
  logs/act_synthetic_standard_100step.log
```

每个日志窗口重点检查：

- `loss`、`grdn` 必须为有限值，不能出现 `nan` 或 `inf`；
- step 应持续增长，不能长期停在视频解码；
- GPU 显存不能持续增长至 OOM；
- 保存点应打印 `Checkpoint policy after step ...`；
- 最终应打印 `End of training`，进程退出码为 0。

短训练里 loss 可以波动，不能因为单个窗口上升就判定失败。合成重复数据上的 loss 下降也不代表
模型具备真机能力。

## 5. 检查 checkpoint 并离线推理

```bash
find outputs/act_synthetic_episode_0004_x10_standard_smoke/checkpoints \
  -mindepth 1 -maxdepth 1 -printf '%f -> %l\n' | sort

cat outputs/act_synthetic_episode_0004_x10_standard_smoke/checkpoints/last/\
training_state/training_step.json

excavator-il smoke-infer \
  outputs/act_synthetic_episode_0004_x10_standard_smoke/checkpoints/last/pretrained_model \
  --dataset-root data/lerobot/synthetic_episode_0004_x10 \
  --repo-id local/synthetic_episode_0004_x10 \
  --sample-index 484 \
  --device cuda
```

通过标准：

- `training_step.json` 中 step 与目标一致；
- `predicted_chunk_shape` 为 `[1, 20, 4]`；
- `action_dim` 为 4；
- `all_finite` 为 `true`；
- checkpoint 和数据集严格匹配 11 维状态、前视 RGB，以及动作顺序
  `[boom, stick, bucket, swing]`。

`smoke-infer` 只对一条离线样本推理，不连接 Orin/STM32。
将通过校验的 checkpoint 转移到 Orin 做 GPU 兼容性验证时，执行
[Orin ACT 离线推理操作手册](orin_act_inference.md)，不要从训练环境直接增加串口或运动设备权限。

### 5.1 用留出 Episode 选择 checkpoint

训练时把 dataset 明确指向物化后的 `train/`，并保存多个 checkpoint。例如：

```bash
set -o pipefail

lerobot-train \
  --dataset.repo_id=local/excavator_rgb_v1_train \
  --dataset.root=data/lerobot/excavator_rgb_v1_split/train \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.pretrained_backbone_weights=null \
  --output_dir=outputs/act_excavator_rgb_v1 \
  --job_name=act_excavator_rgb_v1 \
  --batch_size=2 \
  --num_workers=2 \
  --steps=5000 \
  --log_freq=50 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed=1000 \
  2>&1 | tee logs/act_excavator_rgb_v1.log
```

`--steps` 仍按 train 子集帧数和目标等效 epoch 计算：

```text
steps = ceil(target_epochs * train_num_frames / batch_size)
```

不要根据 validation loss 自动反向训练，也不要把 validation Episode 加回 train。训练结束后在
`validation/` 上统一评估：

```bash
excavator-il evaluate-checkpoints \
  outputs/act_excavator_rgb_v1/checkpoints/001000/pretrained_model \
  outputs/act_excavator_rgb_v1/checkpoints/002000/pretrained_model \
  outputs/act_excavator_rgb_v1/checkpoints/003000/pretrained_model \
  outputs/act_excavator_rgb_v1/checkpoints/004000/pretrained_model \
  outputs/act_excavator_rgb_v1/checkpoints/005000/pretrained_model \
  --split-root data/lerobot/excavator_rgb_v1_split \
  --device cuda --batch-size 4 --num-workers 2
```

命令按 `<split-root>/train` 和 `<split-root>/validation` 目录约定定位数据，并用
`split_provenance.json` 复核内容指纹；还要求每个 checkpoint 的 `train_config.json` 确实指向同一个
train 数据集。历史上用全量数据训练的 checkpoint 会被拒绝，不能冒充留出验证结果。评估结束前
会再次检查两个数据集的指纹，评估期间发生修改时不会产生选择结果。

验证指标 `deployment_prior_l1` 在死区后的归一化专家动作原始空间计算：checkpoint 自带的训练集
归一化器只用于模型输入和模型输出反归一化。评估器按验证集时间顺序逐帧调用 LeRobot
`select_action()`，在每个 LeRobot Episode 边界清空 action queue，并只比较该时刻线上真正会执行的
单步动作与当前专家标签。不同 checkpoint 的归一化统计不会改变 L1 比较尺度。这不是在线 rollout
成功率，也不是 LeRobot 0.5.2 的 VAE posterior loss。`--batch-size`、`--num-workers` 当前仅保留为
兼容参数；严格 replay 不会并行重排时间序列。

选择顺序是 fail-closed：checkpoint 必须先满足所有预测有限、所有反归一化动作保持在 `[-1,1]`，
然后才在安全候选中选择最低 L1。若没有候选，命令退出码为 3。选中的 checkpoint 后续仍必须通过
完整样本范围扫描、Orin GPU 延迟验证和真机上线前安全审查；validation loss 不能替代这些门禁。

## 6. 断点续训

假设 `last` 当前保存于 step 100，要继续训练到总 step 200：

```bash
set -o pipefail

lerobot-train \
  --config_path=outputs/act_synthetic_episode_0004_x10_standard_smoke/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=200 \
  --log_freq=10 \
  --save_freq=50 \
  --eval_freq=0 \
  --wandb.enable=false \
  2>&1 | tee logs/act_synthetic_standard_resume_to_200.log
```

恢复时不要重新指定 policy 结构、数据集或 output directory；它们从 checkpoint 的
`train_config.json` 恢复。`--steps=200` 表示训练到全局 step 200，因此从 step 100 恢复时只执行
100 个增量 step。启动日志应显示：

```text
resume: True
checkpoint_path: .../checkpoints/last
Training: 0/...  # 总数为目标 step 减去已保存 step
```

若按 `Ctrl+C` 中止，LeRobot 不保证立即产生新 checkpoint；只能从最近一次完整保存点恢复。不要
手工改写 `last`、optimizer state 或 `training_step.json`。

## 7. 常见问题

### 输出目录已存在

非 resume 训练会拒绝覆盖已有目录。使用新的 `--output_dir`，或按照第 6 节从完整 checkpoint
恢复；不要删除目录内的部分状态后强行重跑。

### `policy.repo_id` 缺失

本地训练必须设置：

```text
--policy.push_to_hub=false
```

### GPU 显存不足

先把 `--batch_size=2` 降为 `--batch_size=1`。修改后必须使用新的 output directory 重新训练，
不能用不同 batch 配置冒充原 checkpoint 的同一次实验。不要先改图像分辨率或动作/状态维度，
这些属于数据和模型契约变化。

### 视频解码错误或训练卡住

确认命令包含：

```text
--dataset.video_backend=pyav
```

先用 `excavator-il smoke-infer` 读取同一数据集样本；若同样失败，优先检查 LeRobotDataset 视频、
路径和磁盘，而不是修改 ACT。

### loss 出现 `nan`/`inf`

立即停止训练并保留日志与最近 checkpoint。依次检查数据集统计、状态/action 是否有限、输入契约和
损坏视频。不能把非有限 loss 的 checkpoint 用于后续推理或真机。

## 8. 本次实测结果

标准 ACT（51,559,300 参数、batch 2）已完成：

| 阶段 | 结果 |
|---|---|
| 首次训练 | step 0 → 10，checkpoint 正常保存 |
| 断点续训 | step 10 → 100，optimizer/step 正确恢复 |
| 吞吐 | 稳态约 4.2 step/s |
| loss | step 20 窗口 13.933；step 100 窗口 5.422 |
| 最终推理 | `[1,20,4]`，全部有限 |

最终管线测试 checkpoint 位于：

```text
outputs/act_synthetic_episode_0004_x10_standard_smoke/checkpoints/last/pretrained_model
```

该目录由 `.gitignore` 排除，不提交 Git；它是管线验证产物，不是可部署模型。

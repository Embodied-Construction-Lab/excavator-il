# Orin ACT 推理操作手册

第 1～6 节只验证 Orin 上的 NVIDIA PyTorch、LeRobotDataset、ACT checkpoint 回载和 GPU 前向；
这些离线命令不得映射 `/dev/ttyTHS1`、相机或其他控制设备。第 7 节单独说明需要现场确认的在线
Shadow 与 motion 验收。通过离线验证不等于在线 Runtime 获得运动授权。

## 1. 已验证基线

2026-08-12 的现场基线：

| 项目 | 实测值 |
|---|---|
| Orin 系统 | Ubuntu 24.04.4，L4T 39.2，JetPack 7.2 |
| 主机可见内存 | 7.4 GiB（`tegrastats` 总量 7547 MB） |
| 功耗模式 | 40 W |
| 基础镜像 | `nvcr.io/nvidia/pytorch:26.01-py3-igpu` |
| PyTorch / torchvision | `2.10.0a0...nv26.01` / `0.25.0a0...nv26.01` |
| LeRobot | `0.5.2`，commit `12b88fce029cc3a8a94b061cd9e790018873c769` |
| checkpoint | 标准 ACT，51,559,300 参数，11 维 state、前视 RGB、4 维 action |

虽然 PyTorch 会打印 CUDA 版本建议警告，CUDA 设备、FP32/FP16 matmul 和 480×640 FP16 Conv2d
均已实际通过。不要只根据警告判断环境失败，也不能只因 `torch.cuda.is_available()` 为真就跳过
算子测试。

## 2. GPU 基础自检

在 Orin 的 `excavator-il` 工作树运行仓库脚本：

```bash
cd /home/jetson16/workspace_excavator/excavator-il

sudo docker run --rm \
  --runtime=nvidia --gpus all \
  --network=none \
  --read-only \
  --user "$(id -u):$(id -g)" \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD:/workspace/excavator-il:ro" \
  nvcr.io/nvidia/pytorch:26.01-py3-igpu \
  python3 /workspace/excavator-il/scripts/verify_orin_pytorch_gpu.py
```

预期 `passed=true`，两种 matmul 和 Conv2d 的 `*_finite` 均为 `true`，设备能力为 `[8,7]`。

## 3. 构建固定依赖镜像

在 Orin 的 `excavator-il` 工作树执行：

```bash
cd /home/jetson16/workspace_excavator/excavator-il
git status --short

sudo docker build \
  --progress=plain \
  --network=host \
  --build-arg EXCAVATOR_IL_REVISION="$(git rev-parse HEAD)$(test -z "$(git status --porcelain)" || printf '%s' -dirty)" \
  -f docker/act-inference.Dockerfile \
  -t excavator-act-inference:jp72-pytorch261 \
  . 2>&1 | tee /home/jetson16/workspace_excavator/act_inference/docker-build.log
```

正式构建要求 `git status --short` 无输出；带 `-dirty` 的镜像只能用于开发诊断。镜像内固定推理
依赖，不向主机系统 Python 或现有 RL 环境安装包。Dockerfile 仅在镜像内优先
IPv4，避免现场网络访问 PyPI 的 IPv6 慢路径，不修改 Orin 主机网络配置。
基础镜像同时固定 OCI digest。Python 包目前固定版本，但在线 PyPI 构建尚未使用 wheel hash，
因此不应宣称 bit-reproducible；正式部署镜像还必须保存 wheelhouse `sha256sum` 清单并在构建前校验。

若网络不稳定，可先在联网 PC 为 Python 3.12/aarch64 下载 wheel 到忽略 Git 的
`docker/wheelhouse/`，并从权威 LeRobot 工作树导出固定 commit：

```bash
python -m pip download \
  --dest docker/wheelhouse \
  --platform manylinux_2_28_aarch64 \
  --platform manylinux2014_aarch64 \
  --platform any \
  --implementation cp --python-version 312 --abi cp312 \
  --only-binary=:all: \
  -r docker/act-inference.requirements.txt

git -C /path/to/lerobot archive \
  --format=tar.gz \
  --prefix=lerobot-12b88fce029cc3a8a94b061cd9e790018873c769/ \
  -o "$PWD/docker/wheelhouse/lerobot-12b88fce029cc3a8a94b061cd9e790018873c769.tar.gz" \
  12b88fce029cc3a8a94b061cd9e790018873c769
```

把完整构建上下文同步到 Orin 后，在上述 `docker build` 命令中增加
`--build-arg PIP_OFFLINE=1`。wheel 和源码包不提交 Git。

构建后检查版本：

```bash
sudo docker run --rm --runtime=nvidia --gpus all \
  excavator-act-inference:jp72-pytorch261 \
  python3 -c 'import torch, torchvision, lerobot, datasets, av, pyarrow; print(torch.__version__, torchvision.__version__, lerobot.__version__, datasets.__version__, av.__version__, pyarrow.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
```

## 4. 传输和校验离线输入

训练 PC 只传 `pretrained_model` 和一条已转换 LeRobotDataset；训练 optimizer state 不参与
推理。示例：

```bash
rsync -a --info=progress2 \
  outputs/act_synthetic_episode_0004_x10_standard_smoke/checkpoints/000100/pretrained_model/ \
  jetson16@192.168.31.10:/home/jetson16/workspace_excavator/act_inference/checkpoint/

rsync -a --info=progress2 \
  data/lerobot/orin_offline_smoke/ \
  jetson16@192.168.31.10:/home/jetson16/workspace_excavator/act_inference/data/orin_offline_smoke/

# cap-drop=ALL 的容器只能读取明确授予读取权限的离线输入。
# rsync 会保留 PC 文件权限；若 safetensors 是 0600，只调整刚复制的推理副本：
ssh jetson16@192.168.31.10 \
  'find /home/jetson16/workspace_excavator/act_inference/checkpoint -type d -exec chmod 0755 {} + &&
   find /home/jetson16/workspace_excavator/act_inference/checkpoint -type f -exec chmod 0644 {} + &&
   find /home/jetson16/workspace_excavator/act_inference/data/orin_offline_smoke -type d -exec chmod 0755 {} + &&
   find /home/jetson16/workspace_excavator/act_inference/data/orin_offline_smoke -type f -exec chmod 0644 {} +'
```

本次模型文件 SHA-256 为：

```text
3480d9696c8b7a7f9db6c9d9a6e7d7d8b53aeed45243e0e5b89795a10c6ab6c9
```

传输后在两端分别执行 `sha256sum model.safetensors`，结果必须一致。合成数据集必须保留
`pipeline_validation.json`，且 `training_eligible=false`；它只验证管线。

## 5. 只读离线推理与稳态时延

```bash
sudo docker run --rm \
  --runtime=nvidia --gpus all \
  --network=none \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/tmp/huggingface \
  -e XDG_CACHE_HOME=/tmp/cache \
  -v /home/jetson16/workspace_excavator/act_inference/checkpoint:/workspace/checkpoint:ro \
  -v /home/jetson16/workspace_excavator/act_inference/data/orin_offline_smoke:/workspace/dataset:ro \
  excavator-act-inference:jp72-pytorch261 \
  excavator-il smoke-infer /workspace/checkpoint \
    --dataset-root /workspace/dataset \
    --repo-id local/orin_offline_smoke \
    --sample-index 0 \
    --device cuda \
    --warmup-runs 1 \
    --timed-runs 3 \
    --max-inference-ms 100
```

通过标准：

- `predicted_chunk_shape=[1,20,4]`、`action_dim=4`、`all_finite=true`；
- checkpoint 与数据严格符合 11 维状态、前视 RGB 和
  `[boom, stick, bucket, swing]`；
- 预热后的 `inference_max_ms < 100`，才满足孤立环境下 10 Hz 的计算预算；
- 不显式映射串口、相机或液压控制设备；NVIDIA Runtime 只注入 GPU 计算所需设备。

现场实测证据与数值记录在
`../../EvaluationReport/2026-08-12_act_synthetic_training_pipeline_validation.md`。

首次 CUDA 前向包含惰性初始化/内核预热，不是可接受的控制周期。未来在线
Runtime 必须在运动授权前完成 checkpoint 校验、模型加载、至少一次预热和有限值检查；任一步失败
都保持零命令。

## 6. 生成正式部署清单

正式模型选择应在训练 PC 上运行 held-out Episode 评估，并由 evaluator 原子生成部署清单：

```bash
excavator-il evaluate-checkpoints \
  outputs/<run>/checkpoints/*/pretrained_model \
  --split-root data/lerobot/<materialized-split> \
  --device cuda --batch-size 4 --num-workers 0 \
  --deployment-manifest outputs/<run>/deployment_manifest.json \
  --machine-profile ../shared/machine_profile.json \
  --max-deployment-prior-l1 0.20
```

清单绑定 evaluator 实际选中的安全 checkpoint、全文件 SHA-256、训练/验证数据指纹、非合成资格、
`[boom,stick,bucket,swing]` 动作顺序、11 维状态字段、640×480 RGB、chunk 参数与
`shared/machine_profile.json`。任一不一致时 motion 入口拒绝启动。
这里的 `deployment-prior L1` 不是整块 `predict_action_chunk()` 的逐元素误差，而是按验证集时间顺序
逐帧 replay LeRobot `select_action()`、并在每个 LeRobot Episode 边界 `reset()` 后，对实际将要执行的
单步动作计算的 L1。这样 checkpoint 选择与线上 Runtime 的动作消费语义保持一致。
部署清单 schema v2 还锁定 `input_feature_keys` 只能是 11 维状态和单前视 RGB，并要求
`temporal_ensemble_coeff=null`；因此不会在换 checkpoint 后静默切换成 LeRobot temporal ensemble
或多相机语义。当前采用固定 action queue：10 Hz 状态丢失时清空队列并写零，不使用 RTC leftover
merge 或延迟重锚定。后者会改变动作时序，只有新模型和专项真机验证后才能引入。
`0.20` 仅是当前 5 条 Pilot 的阶段门限（当前选中模型 runtime replay 实测 `0.12668`），正式采集扩大后必须用
新的 held-out Pilot 和真机任务成功率重新标定，不能把该数值视为永久性能标准。

当前实验若明确要求跳过 held-out replay，也可以部署“人工授权的最低已保存训练 loss”checkpoint。
这种清单使用 `excavator_act_deployment.v3`，必须如实记录 `validation_performed=false`、checkpoint step、
训练 loss 和训练日志 SHA-256；Runtime 仍严格校验 checkpoint 全文件哈希、训练数据指纹、动作/状态/图像
契约、machine profile，并在每个在线 step 拒绝非有限或越过 `[-1,1]` 的动作。该选择只说明实验操作者
明确接受了未经 held-out 排名的模型，不能写成验证集性能结论，也不能替代之后的真机任务成功率评估。
首次切换仍按 Shadow → 发动机关闭 Motion → 受控真机短运行的顺序验收。

## 7. 在线 Shadow 验证

先停止 Collector、`orin_state_sender.py`、RL Runtime 和任何占用 `/dev/ttyTHS1`/相机的进程。
ACT 与 RL 使用同一个 `F407/data_celect` 统一固件，不需要在此处重烧 STM32；ACT motion Runtime
会同步 v2 telemetry 的 command sequence，并用 manual zero 从 velocity 模式切换到 manual 模式。
Shadow 使用真实相机与 STM32 遥测运行 ACT，但物理串口边界禁止全部写操作。ACT Runtime 的
正式结构是 Orin 本地独立推理，运行时不接收 PC 手柄、deadman、UDP 或 HMAC 数据；PC teleop
只属于人工示教采集与单独的手动控制诊断链路。

若 checkpoint 的 `pretrained_backbone_weights` 不是 `null`，LeRobot 在载入完整 checkpoint 前仍会先
构造 torchvision backbone。在线 Runtime 保持 `--network=none`，因此必须把训练时固定的权重部署到：

```text
/home/jetson16/workspace_excavator/act_inference/torch-cache/checkpoints/resnet18-f37072fd.pth
```

当前权威 SHA-256 是
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。Shadow/Motion 启动脚本都会在
Docker 启动前校验该文件，并把缓存只读挂载到 `/tmp/cache/torch/hub`；不得为了下载 backbone 而给
Motion 容器开放网络。

先在 PC 执行只读 USART2 验收；该命令不启动 Collector、不写串口，也不发送零命令：

```bash
python scripts/diagnose_stm32_link.py
```

只有输出同时满足 `passed=true`、`parse_failure_count=0`、`control_sequence_gap_count=0`、
`estimated_rate_hz` 与 `estimated_control_rate_hz` 均位于 18～22 Hz，并且
`max_receive_period_ms <= 80` 才能继续。固件的控制循环与遥测发送是两个独立 20 Hz 周期，遥测中
相邻 `control_seq` 合法出现一次重复（增量 0）并在下一帧前进两次（增量 2）；诊断器接受这种相位
交换，但仍拒绝增量大于 2、控制循环平均频率异常或真正的遥测长间断。解析失败时复位 STM32 并
检查 TX→RX/共地，不得修改 parser 去接受乱码或错误字段数。

```bash
# 在 Orin 仓库执行
bash scripts/run_act_shadow.sh
```

此命令省略 `--motion-authorization`，因此不得产生任何 STM32 串口写入。
验收包括：CUDA 预热通过、相机与 10 Hz 状态持续、无未来图像、推理小于 100 ms、输出有限且在
`[-1,1]`、序号断点会清空 LeRobot action queue，并且日志中 `serial_write_performed=false`。
Motion 日志同时记录推理 step 和异步安全命令事件；state timeout、unsafe telemetry、startup 与
shutdown 的实际零命令均使用 `excavator_act_runtime_command.v1` 记录。
运行时会先完成一次 synthetic CUDA warmup，再在真实相机 + STM32 新状态上完成一次 live warmup；
只有 live warmup 通过且其内部 action queue 已清空后，runtime 才会打印 `ACT hardware ready`。
若某个 10 Hz state 找不到因果相机帧，runtime 该 step 只记录 `observation_unavailable` 并保持零命令，
不得用未来图像补齐，也不应沿用上一非零动作。

运行至少 30 秒并按 `Ctrl+C` 正常退出后，在 Orin 仓库执行离线日志验收：

```bash
bash scripts/inspect_latest_act_runtime_log.sh shadow
```

脚本自动选择日志目录中最新的 `act_runtime_shadow_*.jsonl`，无需激活 Conda 环境，也不访问串口、
相机或 GPU。只有输出 `passed=true`、`serial_write_count=0`、`command_event_count=0`、step 频率位于
8～12 Hz，且时间因果和配置中的 state/camera age 门限全部通过，Shadow 才算验收成功。

Motion 入口只在现场确认后使用。首次验收必须保持发动机关闭，并确认 Collector、RL Runtime、
历史串口脚本和相机进程全部停止；直接在 Orin 启动 Runtime：

```bash
bash scripts/run_act_motion.sh
```

脚本拒绝竞争的 Collector/RL/ACT Runtime 和被占用的串口/相机，并要求操作者在 Orin 终端输入
精确的一次性启动授权 `ALLOW_ACT_MACHINE_MOTION`。容器使用 `--network=none`，启动后不依赖 PC
或现场 Wi-Fi。该授权会让 ACT 在硬件 ready 后直接向 STM32 写入模型杆量，不能再把 motion 模式当作
“全程零命令”测试；首次必须保持发动机关闭。运行 30 秒后按 `Ctrl+C` 停止 Orin Runtime，检查日志中
动作有限且位于 `[-1,1]`、四轴到六轴映射正确、无 stale/future 输入、startup 与 shutdown 为零、
state timeout/unsafe telemetry 会归零，且终态零之后没有非零写入。以上发动机关闭验收通过后，才能在
新的现场确认下启动发动机安排最小非零动作验收。

正常退出后执行：

```bash
bash scripts/inspect_latest_act_runtime_log.sh motion
```

Motion 检查器要求首条实际写入为 `act_runtime_startup` 零命令、末条为
`act_runtime_shutdown` 零命令，命令序号连续，并逐步核对 ACT 动作
`[boom,stick,bucket,swing]` 到 STM32 `[X1,Y1,Z1,X2,Y2,Z2]` 的映射
`[swing,stick,0,bucket,boom,0]`。推理完成前安全门发生变化而把请求降为零属于正确的 fail-closed
行为；检查器会接受有效的零写入，但仍会在报告中保留实际非零写入数和丢弃状态数供人工复核。

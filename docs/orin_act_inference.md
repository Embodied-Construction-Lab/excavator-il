# Orin ACT 离线推理操作手册

本文只验证 Orin 上的 NVIDIA PyTorch、LeRobotDataset、ACT checkpoint 回载和 GPU 前向。所有
命令都不得映射 `/dev/ttyTHS1`、相机或其他控制设备，不启动 Collector、RL Runtime 或 STM32
通信，也不产生真机动作。通过本文不等于在线 Runtime 获得运动授权。

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
都保持零命令。本文没有实现在线相机/state 接入、action chunk 调度、deadman/运动授权或 STM32
转发，这些仍是后续独立的安全验收阶段。

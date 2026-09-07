# 本地模型仓库

该目录保存 PC 侧的 ACT 模型备份与部署候选。除本说明外，目录内容默认被 Git 忽略，禁止把
`model.safetensors` 等大文件提交到源码仓库。

每个版本使用独立目录，并保持以下布局：

```text
<version>/
├── checkpoint/                    # 完整 LeRobot pretrained_model
├── deployment/                    # deployment_manifest.json
├── config/                        # 与该模型绑定的 act_runtime.orin.json
└── metadata/                      # 来源、时间和核验结果等非运行时资料
```

约束：

- PC 是模型备份与版本归档位置；Orin 只保存当前验收所需的部署副本。
- 切换模型时必须同时切换 checkpoint、deployment manifest 与 runtime config。
- 复制后必须比较两端 SHA-256；不得只凭目录名判断模型版本。
- 未完成 held-out 评估的模型必须明确标为 engineering candidate，不能作为论文正式结果。

## 当前 V3-B 候选

`icra2027_transport_dump_dual_rgb_step115000/` 是 V3-B
`act_dig_transport_dump_reference` 工程参考任务的本地部署候选：

- 输入：按固定角色排序的 `front`、`dump` 两路 640×480 RGB 与 11D proprioception；
- 输出：`[boom, stick, bucket, swing]` 四维归一化动作；
- checkpoint：150k 训练中的 step 115000，由隔离 validation split 上最低安全 L1 选出；
- `model.safetensors` SHA-256：
  `54a3ba90e6c2186787b8b7eb1b9e5211e2bcf81e41551e866283ace41ed04f4a`；
- 状态：PC smoke/evaluator 已通过，尚未完成发动机关闭 HIL 与真机 Commissioning。

该目录的权重仍被 Git 忽略；源码仓库只跟踪与它绑定的 runtime/deployment/evidence 配置和本说明。

## 三阶段 ACT 工程参考候选

`icra2027_transport_dump_three_phase_step140000/` 是与主线平行的探索模型：

- 任务：ACT 连续完成挖掘、运转、倾倒，RL 只负责到达当前/下一挖掘点；
- 输入：`front`、`dump` 双 RGB + 11D proprioception +
  `[phase_is_dig, phase_is_transport, phase_is_dump]`，合计 14D 状态；
- 在线阶段表：`0..92=DIG`、`93..141=TRANSPORT`、`142..240=DUMP`；切换阶段时丢弃
  上一阶段未消费的 action chunk；
- 输出：`[boom, stick, bucket, swing]` 四维归一化杆量；
- checkpoint：step 140000，隔离 validation split 的 deployment-prior L1 为
  `0.1062048085`；
- `model.safetensors` SHA-256：
  `8faf364022695312714ccbac3299635e8fc8ee30b1935b0f3122b1c1fdedd637`；
- 状态：PC 静态资产预检、真实 checkpoint 加载和 CPU warmup 已通过；尚未同步到 Orin、重建容器或
  完成发动机关闭 HIL，因此只能称 engineering candidate。

该模型使用独立 Mission、runtime config、deployment manifest 和 UI config，不会替换
`fixed_target_hybrid` 的 `act_dig_lift` 模型。

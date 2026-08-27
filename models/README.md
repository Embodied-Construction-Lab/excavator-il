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

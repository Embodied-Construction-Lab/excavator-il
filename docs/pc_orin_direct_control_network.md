# PC–Orin 独立有线控制网配置

本文档记录现场已验证的双网络方案：PC 与 Orin 通过独立网线承载控制、采集和 SSH，双方继续通过
Wi-Fi 访问外网。该方案避免 Wi-Fi 排队、重传和频段切换影响 20 Hz 手柄链路，同时不要求 PC
为 Orin 提供 NAT 或网络共享。

## 1. 拓扑与地址

```text
CPE --网线--> 小米路由器 WAN
                   |-- Wi-Fi dr202 --> PC / Orin 外网与备用 SSH

PC USB 网卡 ================= Orin 有线网口
enx00e04c266130                    enP8p1s0
192.168.50.1/24                    192.168.50.2/24
              独立控制网，无网关、无 DNS
```

当前现场配置：

| 用途 | PC | Orin |
| --- | --- | --- |
| 有线接口 | `enx00e04c266130` | `enP8p1s0` |
| 有线地址 | `192.168.50.1/24` | `192.168.50.2/24` |
| NetworkManager profile | `excavator-control-pc` | `excavator-control-orin` |
| Wi-Fi 接口 | `wlp128s20f3` | `wlP1p1s0` |
| Wi-Fi 地址 | `192.168.31.219` | `192.168.31.10` |

换电脑、扩展坞或 Orin 后必须先通过 `ip -br link` 和 `nmcli device status` 重新发现接口名，不能
直接照抄本表。选择控制网网段时还要避开雷达、Docker、Jetson USB bridge 和现场 Wi-Fi 网段。

## 2. 首次配置

配置过程中保持双方 Wi-Fi 连接，不执行 `nmcli networking off` 或 `nmcli radio wifi off`。
建议保留一个使用 Orin Wi-Fi 地址的 SSH 终端，作为有线配置完成前的备用连接。

PC：

```bash
sudo nmcli connection add \
  type ethernet \
  ifname enx00e04c266130 \
  con-name excavator-control-pc \
  connection.autoconnect yes \
  ipv4.method manual \
  ipv4.addresses 192.168.50.1/24 \
  ipv4.never-default yes \
  ipv6.method disabled
```

Orin：

```bash
sudo nmcli connection add \
  type ethernet \
  ifname enP8p1s0 \
  con-name excavator-control-orin \
  connection.autoconnect yes \
  ipv4.method manual \
  ipv4.addresses 192.168.50.2/24 \
  ipv4.never-default yes \
  ipv6.method disabled
```

连接网线后启用 profile：

```bash
# PC
sudo nmcli connection up excavator-control-pc

# Orin
sudo nmcli connection up excavator-control-orin
```

如果 profile 已存在，不要重复 `add`。使用以下命令查看并按需重新启用：

```bash
nmcli connection show excavator-control-pc       # PC
nmcli connection show excavator-control-orin     # Orin
```

## 3. 验收

### 地址和物理链路

```bash
# PC
ip -4 -br address show enx00e04c266130
ethtool enx00e04c266130 | grep -E 'Speed:|Duplex:|Link detected:'

# Orin
ip -4 -br address show enP8p1s0
ethtool enP8p1s0 | grep -E 'Speed:|Duplex:|Link detected:'
```

当前硬件应报告 `1000Mb/s`、`Full` 和 `Link detected: yes`。

### 路由隔离

```bash
# PC：控制地址必须走 USB 网卡；主路由仍保留 Wi-Fi
ip route get 192.168.50.2
ip route show default table main

# Orin：控制地址必须走有线；外网必须走 Wi-Fi
ip route get 192.168.50.1
ip route get 1.1.1.1
```

控制地址的结果必须分别包含 `enx00e04c266130` 和 `enP8p1s0`。Orin 的外网结果必须包含
`wlP1p1s0`。PC 使用 Mihomo TUN 时，`ip route get 1.1.1.1` 可能显示 `Mihomo`；此时以
`ip route show default table main` 仍指向 Wi-Fi，并结合实际 HTTP 请求验证：

```bash
curl -sS --connect-timeout 5 --max-time 10 \
  -o /dev/null -w 'http_code=%{http_code}\n' \
  https://www.google.com/generate_204
```

预期为 `http_code=204`。

### 双向延迟

```bash
# PC -> Orin
ping -c 300 -i 0.1 -W 1 192.168.50.2

# Orin -> PC
ping -c 300 -i 0.1 -W 1 192.168.50.1
```

必须 0% 丢包。千兆直连的平均延迟通常低于 1 ms，不应出现接近 Collector 150 ms 陈旧命令门限
的尖峰。不要通过放宽陈旧命令门限掩盖链路问题。

## 4. 首次通过有线地址 SSH

新 IP 尚未写入 `~/.ssh/known_hosts` 时，首次 SSH 可能提示主机密钥未知或校验失败。先通过旧的
Wi-Fi SSH 查询 Orin ED25519 指纹：

```bash
ssh jetson16@192.168.31.10 \
  'ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub'
```

然后执行：

```bash
ssh jetson16@192.168.50.2
```

确认终端显示的指纹一致后再接受。自动化迁移完成前，可显式复用已验证的 Wi-Fi host key：

```bash
ssh -o HostKeyAlias=192.168.31.10 jetson16@192.168.50.2
```

## 5. 应用配置迁移

只有在地址、双向 ping、外网和 SSH 全部通过后，才把活动控制链路切换到有线地址。ACT Runtime
在 Orin 本地推理，不依赖 PC 网络；需要迁移的是采集和其他 PC–Orin 通信入口。

`excavator-il` 当前需要核对：

- `config/teleop.pc.json`：`orin_host` 改为 `192.168.50.2`；
- `config/guided_episode.pc.json`：`ssh_host` 改为 `jetson16@192.168.50.2`；
- `config/collection.orin.json`：`allowed_pc_host` 改为 `192.168.50.1`。
- `config/collection_ui.pc.json`：`camera_preview_url` 指向
  `http://192.168.50.2:18092/camera/front.mjpg`。

修改前用搜索确认没有遗漏，不盲目替换文档中的历史证据或代理地址：

```bash
rg -n '192\.168\.31\.(10|219)' config scripts docs README.md
```

若 Orin 启用了 UFW，仅为当前 Interface 和端口增加最小规则：

```bash
sudo ufw allow in on enP8p1s0 \
  proto udp from 192.168.50.1 to 192.168.50.2 port 18090 \
  comment 'excavator joystick direct control'

sudo ufw allow in on enP8p1s0 \
  proto tcp from 192.168.50.1 to 192.168.50.2 port 18092 \
  comment 'excavator collector camera preview'
```

不要为整个有线子网开放所有端口。AiryLidar/RL 的端口在迁移对应活动入口时按其权威配置逐项处理。
PC 局域网代理仍可沿 Wi-Fi 使用 `192.168.31.219:7897`，无需给独立有线网设置网关。

## 6. Wi-Fi 省电与回滚

有线控制网不依赖 Wi-Fi，但双方 Wi-Fi profile 仍建议显式关闭省电：

```bash
nmcli -g 802-11-wireless.powersave connection show dr202
```

预期为 `disable`；运行态也可检查：

```bash
iw dev wlp128s20f3 get power_save  # PC
iw dev wlP1p1s0 get power_save     # Orin
```

若需要临时撤销有线链路，只关闭新 profile，不修改 Wi-Fi：

```bash
# PC
sudo nmcli connection down excavator-control-pc

# Orin
sudo nmcli connection down excavator-control-orin
```

需要完全删除时，再分别执行 `sudo nmcli connection delete <profile>`。删除前先把应用配置恢复到仍可达
的地址；否则引导采集脚本可能无法 SSH 到 Orin。

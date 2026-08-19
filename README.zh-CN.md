# Piper FP3 Pipeline（中文）

这是一个面向 **AgileX/松灵 Piper + D405 + D455 + FP3** 的完整数据到部署工程。

它不是简单的“几个 Python 文件”，而是完整链：

```text
人工遥操作
→ D405/D455/Piper 状态采集
→ RAW Episode
→ FP3 H5 转换
→ 数据审计
→ FP3 微调
→ 自动保留历史最低 Loss checkpoint
→ FP3 TCP 推理服务
→ Windows 实机客户端
→ Piper 执行动作
```

## 当前状态

- [x] Piper 主臂 → 从臂遥操作
- [x] D405 + D455 RGB-D 数据采集
- [x] FP3 HDF5 数据转换
- [x] 数据集与 Action 审计
- [x] FP3 LoRA 微调
- [x] Piper 实机推理与执行链路
- [x] FP3 rot6d 到 Piper 位姿的正确解码
- [ ] 稳定可靠的自动叠毛巾策略
- [ ] 更大规模的示范数据集
- [ ] 定量实验评估

## Demo 演示

### 主臂 → 从臂遥操作叠毛巾

以下为主臂控制从臂完成叠毛巾动作的遥操作示范：

![Teleoperation Demo](assets/teleoperation_demo_privacy.gif)

[查看高清 MP4](assets/teleoperation_demo_privacy.mp4)

### FP3 模型实机 Rollout

以下为 FP3 模型在线推理并控制 Piper 实机运动的实验片段：

![FP3 Rollout](assets/fp3_rollout_privacy.gif)

[查看高清 MP4](assets/fp3_rollout_privacy.mp4)

> 公开演示素材已经进行隐私处理：非必要背景区域经过模糊处理，公开视频不包含原始音频。

## GitHub 上需要完整数据吗？

**需要完整的数据链，但不需要把完整数据本体塞进 GitHub。**

GitHub 应该公开：

- 采集脚本
- H5 转换
- 数据 schema
- 数据检查
- 训练配置与训练入口
- checkpoint 选择策略
- 推理 server
- 实机 client
- 环境说明
- 示例配置
- 实验结果和 checksum

不要提交：

- 真实 `.h5`
- RAW RGB-D
- 11GB `.pth`
- FP3 pretrained weight
- `.env`
- SSH key
- access token
- 私人路径配置

## 当前主线

- 手眼相机：Intel RealSense D405
- 外部相机：Intel RealSense D455
- 数据：camera-local / uncalibrated XYZRGB point cloud
- 每相机每帧：8000 points
- `action[t] = state[t+1]`
- 10D action = 3 position + 6 rotation + 1 gripper
- rot6d 使用旋转矩阵前两**行**
- 训练：2000 epoch × 5 step = 10000 optimizer steps
- checkpoint：只保留历史最低 epoch train loss
- server：默认不启用 `replan_every_step`
- client：正确 rot6d 解码 + Ctrl+C 回启动初始关节位

## 最短命令

Windows：

```powershell
.\pipeline.ps1 collect -Episode 21 -Duration 30
.\pipeline.ps1 upload -Episode 21
```

RTX 主机：

```bash
./pipeline_remote.sh prepare
./pipeline_remote.sh train
./pipeline_remote.sh serve
```

Windows：

```powershell
.\pipeline.ps1 run
```

详细内容请看英文 `README.md` 和 `docs/`。

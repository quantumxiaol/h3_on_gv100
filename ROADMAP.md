# ROADMAP — h3_on_gv100

## 背景

MiniMax H3（海螺 3）是 MiniMax 于 2026 年 7 月底发布、8 月初开放权重的全模态生成模型：33B 稠密参数，单次前向同时生成视频与 32kHz 立体声音频，本地基准输出为 768p（短边 768、32 的倍数对齐）、4–15 秒、24 FPS。开放权重只包含 H3-Base（FL2VA 与 Ref2VA 两个分区）；Context-IR（prompt 增强）与 Regenerate-2K（2K 精修）是托管服务，不在开放权重内。

目标：把 H3 推理在 GV100 工作站上稳定跑起来，作为**创意草图机**——本地批量出草稿、选定的镜头走 MiniMax 官方 API 的 Regenerate-2K（¥0.30/秒）精修出片。

学校计算节点（8×A6000，驱动 535 无 root、CentOS 7、无容器）已放弃本地部署，仅保留 API 使用可能。本机路线是确认的最终方案，不要回头建议换硬件。

## 技术路线（已锁定）

```
MiniMax-H3 权重（modelsweights/，scripts/download_modelscope.py 从 modelscope 下载）
    │
    ▼
diffusers ModularPipeline（workflow="fl2va"）
    + ComponentsManager.enable_auto_cpu_offload  ← 权重 ~123GB FP16 驻内存，
    │                                              逐组件换入 32GB 显存
    ├─ dtype: 全链路 FP16（V100 无 BF16；--dit-dtype float32 为排障开关）
    ├─ attention: SDPA（强制；Volta 不支持 FA2/SageAttention，禁装 flash-attn）
    ├─ 步数: 50 步基线 → Turbo LoRA 8 步（ModelTC Minimax-H3-Turbo，M1 接入）
    └─ 网格: 帧数 17n+5（73/90/107/124…），宽高 32 倍数；CFG 蒸馏模型，无 negative prompt
    │
    ▼
产物：video_only.mp4（24fps H.264）+ audio.wav（32kHz）→ ffmpeg 合流成片
```

**明确排除的路线**：SGLang / vLLM（SM70 无验证，依赖 FlashAttention-2 / SageAttention）；量化（NF4/INT8，504GB 内存下无必要，徒增 Volta 兼容性风险）。

## 里程碑

### M0 冒烟跑通（当前）

- 任务：跑通 `infer.py`，出第一条 3 秒 960×544、50 步的视频。
- 预期难点：H3 进 diffusers 仅数月，ModularPipeline 调用细节（参数名/返回结构）可能随版本漂移，需按报错适配 1–2 处；首次推理 20–40 分钟。
- 验收：`out/smoke.mp4` 可播放，音画同步，画面无 NaN 花屏。

### M0.5 API 服务化（已建骨架，待 M0 后验证）

- 任务：FastAPI 异步 job 服务（`h3-server`）+ 本机客户端（`h3-client`）+ 整体工作流（`h3-imagine`：想法 → 生图 API 首帧 → 图生视频）。文件走 base64。公网入口由用户配置内网穿透。
- 状态：代码完成（h3_gv100/server.py、client.py、tools/），客户端链路在本机有 pytest 覆盖；服务端依赖 M0 的引擎验证。
- 验收：本机 curl/客户端提交 → 轮询 → 取回 base64 mp4 闭环；`h3-imagine` 一条命令从想法到成片。

### M1 Turbo LoRA 接入

- 任务：按 ModelTC/Minimax-H3-Turbo 仓库 examples 接入 8 步 LoRA，确认 diffusers 下 LoRA 加载与 fuse 方式。
- 验收：8 步出片，单条 5 秒耗时 ≤30 分钟；与 50 步样片对比，运动/音质下降可接受即采用为默认档。

### M2 双卡分片（被占卡回归后）

- 任务：DiT FP16（66GB）分片双卡常驻（每卡 ~33G），编码器/VAE 走 offload，卡间搬运走 NV4。
- 预期：3–5 分钟/条。
- 验收：双卡模式稳定跑 3 条不 OOM、不静默降质。

### M3（可选）API 精修管道

- 任务：封装 MiniMax 开放平台 Regenerate-2K 异步调用（提交→轮询→下载），输入本地 768p 草稿 + 增强 prompt，输出 2K。
- 验收：一条本地草稿完成 2K 精修闭环，成本可预期（~¥3/10 秒片）。

### M4（可选）意图调度层

- 任务：CLI/agent 入口——一句自然语言描述 → 组织 prompt → 调本地生成（或 API）→ 取回成片归档。
- 验收：非技术用户给一句话能拿到成片。
- 备注：`h3-imagine` 已覆盖其雏形（想法→首帧→视频），M4 视需要再加深（prompt 组织/归档/批量）。

### M5（可选）ComfyUI 对照组

- 仅当 M1 后仍嫌 diffusers 可控性不足时考虑；ComfyUI 0.30+ 有官方 H3 模板，但注意其数值基线与官方 checkpoint 不同，性能结论不可混用。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| FP16 动态范围溢出（花屏/NaN） | `--dit-dtype float32` 定位；必要时 DiT fp32 + 其余 fp16 混合 |
| diffusers API 漂移导致脚本失效 | 改动前 `pip show diffusers` 记录版本；调用细节以 site-packages 内源码为准；`h3_gv100/pipeline.py` 已做签名自省适配 |
| 525 驱动边界问题 | torch 已实测可跑；若新依赖触发 CUDA 报错，先在当前环境最小复现再升级驱动（535 是 GV100 最后支持分支） |
| 共享机资源冲突 | 跑前 `nvidia-smi` 查空卡，`--device` 指定；长任务 nohup/screen |
| 模型下载中断 | 相同命令重跑即可（断点续传） |

## 里程碑间的关系

M0 是全部后续的前提（M0.5 的 API 也建立在同一引擎上）；M1 决定这台机器是否"日常可用"；M2 决定它是否"好用"；M3/M4 是向生产工作流的延伸，与本地性能无关，可并行推进。

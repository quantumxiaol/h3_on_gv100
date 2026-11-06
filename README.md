# h3_on_gv100

MiniMax H3（33B 全模态视频+音频生成）在 **2×Quadro GV100 32GB** 工作站上的本地部署：
diffusers + CPU offload 推理，FastAPI 封装成**图生视频 API**；本机（mac）持有客户端，
一个想法 → 调生图 API 出首帧 → 调 H3 API 出视频。

```
本机 mac（客户端，无 GPU 依赖）                GV100 工作站（服务端）
─────────────────────────────                ─────────────────────────
h3-imagine "想法"
  1. GENIMG API (gpt-image-2.5)  ──生首帧图──▶
  2. POST /v1/videos  (base64 首帧 + prompt) ──────────▶  H3Engine
                                                            ModularPipeline(fl2va)
                                                            + auto_cpu_offload
                                                            FP16 / SDPA / 17n+5
  3. GET  /v1/videos/{id}  ◀──── 轮询（单条 20–40 min）──    job 队列，单 worker
  4. GET  /v1/videos/{id}/content  ◀── base64 mp4+wav ──    ffmpeg 合流
```

## 硬件档案（实测，勿凭假设改动）

| 项目 | 实测值 |
| --- | --- |
| 机型 | Dell Precision 7920 Tower，双路 Xeon（40 物理核） |
| GPU | 2× Quadro GV100 32GB（Volta，SM70，**无 BF16**，Tensor Core 仅 FP16） |
| 卡间互联 | NV4（NVLink 2.0 ×4，单向约 100GB/s）——双卡形态才可用 |
| PCIe | Gen3 x16 满血——单卡 offload 瓶颈 ~12–14GB/s |
| 驱动 | 525.105.17（CUDA ceiling 12.0）；torch 2.8.0+cu126 自带 12.x runtime，minor-version compat 踩线达标，**已实测**（fish-speech 同环境在跑） |
| 内存 | 504GB DDR4 ECC，权重 FP16 ~123GB 全量驻内存做 offload |
| NUMA | 双卡同挂 NUMA 0 → 推理进程加 `numactl --cpunodebind=0 --membind=0` 前缀 |
| 系统 | Ubuntu 桌面版，**4 人共享工作站**，无 root |

## 技术路线（已锁定，偏离需先论证）

1. **框架**：diffusers `ModularPipeline`（workflow="fl2va"）+ `ComponentsManager.enable_auto_cpu_offload`。不走 SGLang/vLLM（SM70 无验证、依赖 FA2/SageAttention）。
2. **精度**：全链路 FP16。**严禁 bfloat16**。注意官方权重发布精度是 BF16，FP16 是加载时强制降 cast（动态范围收窄，溢出风险见排障）。
3. **attention**：SDPA。**不要安装/启用 flash-attn、SageAttention、FlashAttention-2**。
4. **网格约束**：帧数 `17n+5` 且落在 5–15 秒窗口（**124/141/…/345**，更低会被夹到 124；73 帧 ≈ 3 秒不满足模型时长约束），宽高 32 的倍数；H3 是 CFG 蒸馏模型，**不要加 negative prompt / guidance**。
5. **产物**：视频 24fps + 音频 32kHz 立体声，ffmpeg 合流 MP4。
6. **画布**：960×544 比训练画布 1344×768 每步快约 2.3×（官方文档数据），草图机默认用小画布。

## 目录结构

```
h3_gv100/            # 包：schemas(共用) / pipeline(引擎) / server(API) / client(客户端)
tools/               # 入口：h3-server / h3-client / h3-imagine（装好后是可执行命令）
scripts/             # 权重下载：download_modelscope.py / download_huggingface.py（断点续传）
.agents/skills/      # h3-prompt-writing：官方 H3 提示词写作 skill（含 LOCAL-API.md 本地适配）
infer.py             # M0 冒烟脚本（与 API 共用同一引擎）
modelsweights/       # 模型权重（git 忽略，fl2va 范围 ~144GB）
ROADMAP.md           # 里程碑 M0–M5 与验收标准
AGENT_PROMPT.md      # 交接给 GV100 上 coding agent 的提示词
```

## 一、GV100 服务端部署

```bash
cd ~/h3_on_gv100
uv sync --extra server --extra cu126 --extra download   # torch 2.8.0+cu126，与 fish-speech 实测环境一致
# 权重 -> modelsweights/MiniMax-H3（fl2va 范围 ~144GB，断点续传；两个脚本同一目标目录，二选一/换源续传均可）
uv run python scripts/download_modelscope.py     # ModelScope 镜像（MiniMax/MiniMax-H3）
uv run python scripts/download_huggingface.py    # HuggingFace 官方（MiniMaxAI/MiniMax-H3，默认走 hf-mirror.com）
# 磁盘空间足够且要 Ref2VA 才加 --full（整仓约 500GB，含原始 checkpoint 分区与 ref2va transformer）

nvidia-smi                                   # 共享机纪律：先确认目标卡空闲
numactl --cpunodebind=0 --membind=0 uv run h3-server --device cuda:0 --port 8010
# 长任务用 nohup / tmux 挂载；引擎后台加载 ~144GB 需数分钟，GET /health 看 engine: ready
```

冒烟（M0，不经 API 直接验证引擎）：

```bash
numactl --cpunodebind=0 --membind=0 uv run python infer.py \
  --prompt "A red panda sipping tea in a sunlit kitchen, cinematic lighting" \
  --frames 124 --steps 50 --out out/smoke
```

首次推理预计 **20–40 分钟**（PCIe offload + 50 步）。参考：官方公布 2×RTX 5090 32GB + TP2 layerwise 卸载跑 50 步端到端 559s；我们的 diffusers 按组件 offload、PCIe Gen3 带宽更低，预期更慢。**不要因慢而打断或加并行优化。**

服务端环境变量（可写进 `.env`）：`H3_MODEL_PATH` `H3_DEVICE` `H3_DIT_DTYPE` `H3_RESERVE`
`H3_PORT` `H3_OUT_DIR` `H3_API_KEY`（公网暴露时务必设置）。

## 二、本机（mac）客户端

```bash
uv sync                # 只装 httpx/pydantic/dotenv，无 torch
cp .env.example .env   # 填 GENIMG_*（生图）与 H3_API_BASE_URL（你做的内网穿透/公网地址）
```

### 整体工作流：一个想法 → 成片

```bash
uv run h3-imagine "一只红色小熊猫在洒满阳光的厨房里喝茶，电影感打光" \
  --frames 124 --steps 50
# 产物：out/imagine/<时间戳>/keyframe.png + final.mp4（+ audio.wav）
```

### 单独调 H3 API

```bash
uv run h3-client health
uv run h3-client submit --prompt "..." --image keyframe.png --frames 124 --wait
# 首尾帧（FL2VA）/ 仅尾帧（L2VA）：加 --last-image last.png
uv run h3-client status <job_id>
uv run h3-client download <job_id> --kind mp4
```

## 三、API 一览（服务端，文件一律 base64）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 引擎状态（loading/ready/error）、GPU、队列长度 |
| POST | `/v1/videos` | 提交任务 → 202 `{job_id}`。body：`prompt`, `image_b64?`(首帧), `last_image_b64?`(尾帧), `width=960`, `height=544`, `frames=124`(17n+5, 124–345), `steps=50`, `seed=42`, `fps=24` |
| GET | `/v1/videos/{id}` | 轮询状态（queued/running/done/error + elapsed_s） |
| GET | `/v1/videos/{id}/content` | 产物 JSON：`video_b64`(合流 mp4)、`audio_b64`、`meta` |
| GET | `/v1/videos/{id}/download?kind=mp4` | 直接下载文件 |

设置了 `H3_API_KEY` 后，除 `/health` 外都需要 `Authorization: Bearer <key>`。

curl 示例：

```bash
curl -X POST $H3_API_BASE_URL/v1/videos -H "Authorization: Bearer $H3_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"prompt": "A red panda sipping tea", "frames": 124, "steps": 50}'
curl $H3_API_BASE_URL/v1/videos/<job_id> -H "Authorization: Bearer $H3_API_KEY"
```

## 四、排障

| 症状 | 对策 |
| --- | --- |
| 花屏 / NaN | 官方权重是 BF16、我们强制 cast FP16，动态范围收窄。服务端以 `H3_DIT_DTYPE=float32` 重启定位；必要时 DiT fp32 + 其余 fp16 混合 |
| `pipe.__call__` 参数报错 | diffusers 版本漂移。报错信息里带实际签名；对照 `site-packages/diffusers` 源码更新 [pipeline.py](h3_gv100/pipeline.py) 的 `_PARAM_CANDIDATES` |
| CUDA 相关报错 | 先 `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"` 最小复现；驱动 535 是 GV100 最后支持分支，谨慎升级 |
| offload 异常慢 | 带宽实测：1GB pinned tensor 连续 H2D `copy_` 计时，预期 12–14GB/s，低于 6GB/s 先修链路/锁页 |
| 下载中断 | 同一下载命令重跑（断点续传）；怀疑缓存损坏用 `--force-download`，不要静默换 revision |
| frames 报越界 | 模型时长窗口 5–15s：传入值会被对齐到 17n+5 并夹到 [124, 345]，不需要手动调 |
| 需要可复现基线 | 用 `download_huggingface.py --revision <commit>` 固定 commit；ModelScope 镜像 revision 与 HF 一致性无公开保证 |
| `Missing dependencies for SOCKS support` / 下载超时 | 本机出口走企业 VPN 的 SOCKS5 代理（`ALL_PROXY=socks5://…`），**直连出不去，不要 unset 代理**。`download` extra 已带 `pysocks`+`socksio`，保留代理环境变量重跑即可 |

版本可回溯：环境跑通后 `uv pip freeze > requirements.txt` 提交；改动前 `pip show diffusers` 记录版本。

## 里程碑

见 [ROADMAP.md](ROADMAP.md)：M0 冒烟 → M1 Turbo LoRA（8 步）→ M2 双卡分片 → M3 官方 2K 精修管道 → M4 意图调度层。

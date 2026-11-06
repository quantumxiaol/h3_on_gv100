# LOCAL-API — 在 h3_on_gv100 项目上使用本 skill

> 本文件是 h3_on_gv100 项目对官方 `h3-prompt-writing` skill 的**本地适配层**，
> 不改变官方 prompt 结构与写作规则，只说明写好的 prompt 如何落到本项目的 API 与约束上。
> 官方正文见 SKILL.md 与 references/base-en.txt、references/ref-en.txt。

## 1. 本项目的角色

本机（mac）是客户端，GV100 工作站跑本地 H3-Base（FL2VA 分区）。官方 H3-Context-IR
（prompt 增强）是托管服务、不在开放权重内——**本 skill 就是我们本地的 Context-IR 替代品**：
agent 用本 skill 把用户的一句话想法改写成 H3 结构化 prompt，再交给本地 API。

标准工作流（对应 `h3-imagine`）：

```
用户想法 → (本 skill 改写为 H3 prompt 结构) → 生图 API 出首帧 → GV100 H3 API 图生视频
```

## 2. 模式 → API 字段映射

本地只部署了 **FL2VA 分区**（diffusers `workflow="fl2va"`，含 t2va/i2va/fl2va/l2va 四种用法）；
Ref2VA 未部署（需要 `transformer_ref/` 分区， roadmap 待定）。

| skill 模式 | `POST /v1/videos` 字段 | 说明 |
| --- | --- | --- |
| T2VA | `prompt` | 不带图 |
| I2VA | `prompt` + `image_b64` | 首帧（h3-imagine 的主路径） |
| FL2VA | `prompt` + `image_b64` + `last_image_b64` | 首尾帧 |
| L2VA | `prompt` + `last_image_b64` | 仅尾帧 |
| Ref2VA | — | 本地 API 未暴露，需 ref2va 分区 |

base-en.txt 里 I2VA/FL2VA/L2VA 的对齐指令（`<Picture 1>` 等）按原文写进 `prompt` 字符串即可，
图片本体经 `image_b64` / `last_image_b64` 以 base64 传输，二者按顺序对应 `<Picture 1>` / `<Picture 2>`。

## 3. 本地硬约束（与官方不同处注意）

| 项 | 值 | 备注 |
| --- | --- | --- |
| 时长 | **5–15 秒** | 本地 diffusers 约束；`frames` 会被对齐+夹到 124–345（17n+5） |
| 帧率 | 24 FPS 固定 | `fps` 字段保留但勿改 |
| 宽高 | 32 的倍数 | 不给时由首帧比例推导（短边 768 训练画布） |
| 推荐画布 | **960×544** | 比训练画布 1344×768 每步快约 2.3×（草图机定位） |
| negative prompt / guidance | **不要传** | CFG 蒸馏模型，无此参数 |
| 步数 | 默认 50 | Turbo LoRA（M1）后 8 |
| 时长-帧数换算 | `frames = round(秒 * 24)` | 再向上对齐 17n+5 |

## 4. 调用方式

整体工作流（推荐）：

```bash
# .env 配好 GENIMG_*（生图）与 H3_API_BASE_URL（GV100 服务）
h3-imagine "想法" --video-prompt "$(本 skill 改写后的完整 prompt)" --frames 124
```

直接提交（已有首帧图）：

```bash
h3-client submit --prompt "<skill 改写后的 prompt>" --image keyframe.png \
  [--last-image last.png] --frames 124 --wait
```

curl（base64 传图）：

```bash
curl -X POST $H3_API_BASE_URL/v1/videos -H "Authorization: Bearer $H3_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"prompt": "...", "image_b64": "<base64>", "frames": 124, "steps": 50}'
# -> {"job_id": "..."}; 轮询 GET /v1/videos/{id}; 取回 GET /v1/videos/{id}/content
```

注意：本地单条生成 20–40 分钟（GV100 + CPU offload + 50 步），job 异步，轮询间隔 30s 即可。

## 5. 写作要点（继承官方，强调本地相关）

- 改写段落用英文；台词、歌词、画面文字保留原语言（H3 稳定支持中/英等 11 种语言）。
- I2VA 以首帧为锚：画面从首帧向前发展，写清「运动路径」而非罗列静态描述。
- 音画同出是 H3 特性：`overall_soundscape` 与 `non_diegetic_music` 不要留空，
  环境声、动作音效、配乐分开写，时间记号（"At 00:04.500"）与镜头时长一致。
- 本地无 Regenerate-2K：选定草稿后可走官方 API 精修（roadmap M3）。

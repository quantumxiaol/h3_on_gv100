#!/usr/bin/env python3
"""从 HuggingFace 下载 MiniMax-H3 权重到 modelsweights/（与 modelscope 脚本同一目标目录）。

默认只下 diffusers FL2VA 需要的部分（~144GB），忽略原始 checkpoint 分区与 ref2va 分区：
  FL2VA/ (~144GB)  Ref2VA/ (~144GB)  transformer_ref/ (~66GB)
`--full` 下载整仓（约 500GB，注意磁盘空间）。

用法：
    uv sync --extra download
    python scripts/download_huggingface.py
    # 固定 revision 复现（推荐内网/可复现基线）：
    python scripts/download_huggingface.py --revision <commit>
    # 中断后重跑同一命令续传；怀疑缓存损坏时加 --force-download

- 国内网络默认走 hf-mirror.com（--endpoint 或 HF_ENDPOINT 覆盖，官方源 https://huggingface.co）
- 需要授权时：export HF_TOKEN=hf_xxx
- 与 modelscope 脚本混用：已完整下载的文件互相跳过；分片临时文件互不干扰可换源续传
  （modelscope 用 ____temp，HF 用 .cache/huggingface）
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

SKIP_DIRS = ["FL2VA/*", "Ref2VA/*", "transformer_ref/*"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--local-dir", default="modelsweights/MiniMax-H3")
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="默认 hf-mirror.com；官方源 https://huggingface.co",
    )
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--full", action="store_true", help="下载整仓（约 500GB；默认只下 fl2va 的 diffusers 部分 ~144GB）")
    p.add_argument("--force-download", action="store_true", help="忽略已有文件重新下载（修复缓存损坏时用）")
    args = p.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint  # 需在 import huggingface_hub 之前设置
    from huggingface_hub import snapshot_download

    path = Path(args.local_dir)
    path.mkdir(parents=True, exist_ok=True)
    scope = "整仓 ~500GB" if args.full else "fl2va diffusers 部分 ~144GB"
    rev = f"，revision={args.revision}" if args.revision else ""
    print(f"下载 {args.repo} -> {path}（{scope}{rev}，中断后重跑同一命令续传）")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(path),
        revision=args.revision,
        max_workers=args.max_workers,
        ignore_patterns=None if args.full else SKIP_DIRS,
        force_download=args.force_download,
    )
    print(f"完成。校验：ls {path} 应有 model_index.json、transformer/、text_encoder/、vae/ 等")


if __name__ == "__main__":
    main()

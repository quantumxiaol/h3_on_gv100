#!/usr/bin/env python3
"""从 HuggingFace 下载 MiniMax-H3 权重到 modelsweights/（与 modelscope 脚本同一目标目录）。

用法：
    uv sync --extra download
    python scripts/download_huggingface.py

- 国内网络默认走 hf-mirror.com（--endpoint 或环境变量 HF_ENDPOINT 覆盖，官方源用 https://huggingface.co）
- 断点续传：中断后重跑同一命令即可（部分文件在 .cache/huggingface 里续）
- 需要授权时：export HF_TOKEN=hf_xxx
- 与 modelscope 脚本混用说明：已完整下载的文件会互相跳过；各工具的分片临时文件
  互不干扰（modelscope 用 ____temp，HF 用 .cache/huggingface），可放心换源续传。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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
    args = p.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint  # 需在 import huggingface_hub 之前设置
    from huggingface_hub import snapshot_download

    path = Path(args.local_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"下载 {args.repo} -> {path}（endpoint={args.endpoint}，~123GB，中断后重跑同一命令续传）")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(path),
        revision=args.revision,
        max_workers=args.max_workers,
    )
    print(f"完成。校验：ls {path} 应有 FL2VA/、model_index.json 等")


if __name__ == "__main__":
    main()

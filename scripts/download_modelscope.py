#!/usr/bin/env python3
"""从 ModelScope 下载 MiniMax-H3 权重到 modelsweights/（断点续传：中断后原命令重跑即可）。

用法：
    uv sync --extra download
    python scripts/download_modelscope.py
    # 或指定别的仓库/目录
    python scripts/download_modelscope.py --repo MiniMaxAI/MiniMax-H3 --local-dir modelsweights/MiniMax-H3

HuggingFace 路线（备选，同一目标目录）：python scripts/download_huggingface.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--local-dir", default="modelsweights/MiniMax-H3")
    p.add_argument("--revision", default=None)
    args = p.parse_args()

    from modelscope import snapshot_download

    path = Path(args.local_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"下载 {args.repo} -> {path}（~123GB，中断后重跑同一命令续传）")
    snapshot_download(args.repo, local_dir=str(path), revision=args.revision)
    print(f"完成。校验：ls {path} 应有 FL2VA/、model_index.json 等")


if __name__ == "__main__":
    main()

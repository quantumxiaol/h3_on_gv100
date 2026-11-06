#!/usr/bin/env python3
"""从 ModelScope 下载 MiniMax-H3 权重到 modelsweights/（断点续传：中断后原命令重跑即可）。

默认只下 diffusers FL2VA 需要的部分（~144GB）：
  transformer/ text_encoder/ vae/ audio_vae/ scheduler/ audio_scheduler/
  tokenizer/ processor/ model_index.json modular_model_index.json
忽略（不需要、体积巨大）：
  FL2VA/ Ref2VA/      —— 原始 checkpoint 分区（SGLang/vLLM 用），每个 144GB
  transformer_ref/    —— ref2va 分区（本项目只部署 fl2va）
`--full` 可下载整仓（约 500GB，注意磁盘空间）。

注意：ModelScope 是中国大陆镜像，其 revision 与 HF 固定 commit 的一致性无公开保证；
追求可复现基线时优先用 HuggingFace 固定 revision 那条（scripts/download_huggingface.py）。

用法：
    uv sync --extra download
    python scripts/download_modelscope.py
    python scripts/download_modelscope.py --full --repo MiniMax/MiniMax-H3
"""

from __future__ import annotations

import argparse
from pathlib import Path

# diffusers fl2va 用不到的目录（原始 checkpoint 分区 + ref2va transformer）
SKIP_DIRS = ["FL2VA/*", "Ref2VA/*", "transformer_ref/*"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="MiniMax/MiniMax-H3")
    p.add_argument("--local-dir", default="modelsweights/MiniMax-H3")
    p.add_argument("--revision", default=None)
    p.add_argument("--full", action="store_true", help="下载整仓（约 500GB；默认只下 fl2va 的 diffusers 部分 ~144GB）")
    args = p.parse_args()

    from modelscope import snapshot_download

    path = Path(args.local_dir)
    path.mkdir(parents=True, exist_ok=True)
    ignores = None if args.full else SKIP_DIRS
    scope = "整仓 ~500GB" if args.full else "fl2va diffusers 部分 ~144GB"
    print(f"下载 {args.repo} -> {path}（{scope}，中断后重跑同一命令续传）")
    try:
        snapshot_download(args.repo, local_dir=str(path), revision=args.revision,
                          ignore_patterns=ignores)
    except TypeError:
        # 老版 modelscope 不支持 ignore_patterns：退化为全量下载并提示
        if ignores:
            print("[warn] 当前 modelscope 不支持 ignore_patterns，将下载整仓（~500GB）")
        snapshot_download(args.repo, local_dir=str(path), revision=args.revision)
    print(f"完成。校验：ls {path} 应有 model_index.json、transformer/、text_encoder/、vae/ 等")


if __name__ == "__main__":
    main()

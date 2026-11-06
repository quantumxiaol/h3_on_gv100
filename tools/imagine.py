#!/usr/bin/env python3
"""h3-imagine — 整体工作流：一个想法 -> 生图（首帧）-> 图生视频。

    h3-imagine "一只红色小熊猫在洒满阳光的厨房里喝茶，电影感打光"

流程：
  1. 调生图 API（GENIMG_*，OpenAI 兼容，如 gpt-image-2.5）生成首帧图
  2. 首帧图 base64 + prompt 提交到 GV100 的 H3 API（H3_API_BASE_URL）
  3. 轮询任务直到完成（单条 20–40 分钟，耐心等），取回 base64 mp4 落盘

产物在 out/imagine/<时间戳>/：keyframe.png、request.json、final.mp4（+ audio.wav）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser(description="想法 -> 生图 -> H3 图生视频")
    p.add_argument("idea", help="一句话想法（同时作生图与视频 prompt，可用 --image-prompt/--video-prompt 分开指定）")
    p.add_argument("--image-prompt", default=None, help="生图 prompt（默认同 idea）")
    p.add_argument("--video-prompt", default=None, help="视频 prompt（默认同 idea）")
    p.add_argument("--size", default="1536x1024", help="生图尺寸（横版适合视频首帧）")
    p.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"])
    p.add_argument("--width", type=int, default=960, help="视频宽（自动对齐 32）")
    p.add_argument("--height", type=int, default=544, help="视频高（自动对齐 32）")
    p.add_argument("--frames", type=int, default=124, help="向上对齐 17n+5，范围 124–345；124≈5.2s")
    p.add_argument("--steps", type=int, default=50, help="基线 50；Turbo LoRA 后 8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="out/imagine")
    p.add_argument("--poll-interval", type=float, default=30.0)
    args = p.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    from h3_gv100.client import GenImgClient, H3Client, imagine

    genimg_base = os.environ.get("GENIMG_API_BASE_URL")
    genimg_key = os.environ.get("GENIMG_API_KEY")
    genimg_model = os.environ.get("GENIMG_MODEL", "gpt-image-2.5")
    h3_base = os.environ.get("H3_API_BASE_URL")
    if not genimg_base or not genimg_key:
        sys.exit("缺少 GENIMG_API_BASE_URL / GENIMG_API_KEY（见 .env.example）")
    if not h3_base:
        sys.exit("缺少 H3_API_BASE_URL（见 .env.example）")

    genimg = GenImgClient(genimg_base, genimg_key, genimg_model)
    h3 = H3Client(h3_base, api_key=os.environ.get("H3_API_KEY") or None)

    def on_event(event: str, **kw):
        ts = time.strftime("%H:%M:%S")
        if event == "genimg.start":
            print(f"[{ts}] 1/3 生图中: {kw['prompt'][:80]}", flush=True)
        elif event == "genimg.done":
            print(f"[{ts}] 2/3 首帧已保存: {kw['path']}，提交 H3…", flush=True)
        elif event == "h3.submitted":
            print(f"[{ts}]    job_id={kw['job_id']}，轮询中（单条约 20–40 分钟）…", flush=True)
        elif event == "h3.poll":
            info = kw["info"]
            state = info.get("status")
            elapsed = info.get("elapsed_s")
            line = f"[{ts}]    {state}"
            if elapsed:
                line += f"  已运行 {elapsed:.0f}s"
            print(line, flush=True)
        elif event == "h3.done":
            print(f"[{ts}] 3/3 完成: {kw['paths']}", flush=True)

    saved = imagine(
        genimg,
        h3,
        args.idea,
        image_prompt=args.image_prompt,
        video_prompt=args.video_prompt,
        size=args.size,
        quality=args.quality,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
        out_dir=args.out,
        poll_interval=args.poll_interval,
        on_event=on_event,
    )
    if "video" in saved:
        print(f"\n成片: {saved['video']}")


if __name__ == "__main__":
    main()

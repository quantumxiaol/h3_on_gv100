#!/usr/bin/env python3
"""h3_on_gv100/infer.py — M0 冒烟脚本：MiniMax H3 (33B) on Quadro GV100。

Volta 特化设计（与 AGENT_PROMPT 锁定路线一致）：
  1. 全链路 FP16（V100 无 BF16 单元）；--dit-dtype float32 用于排查数值问题
  2. 强制 SDPA attention（Volta 不支持 FlashAttention-2 / SageAttention）
  3. diffusers 官方 ModularPipeline + ComponentsManager.enable_auto_cpu_offload
     权重 ~123GB FP16 驻 512GB 内存，逐组件换入 32GB 显存
  4. H3 为 CFG 蒸馏模型：无 negative prompt，guidance 恒 1

引擎实现在 h3_gv100/pipeline.py（H3Engine），本脚本只是其 CLI 冒烟入口，
与服务端（tools/h3_server.py）共用同一份引擎代码，保证 M0 验证的就是 API 跑的逻辑。

用法：
    numactl --cpunodebind=0 --membind=0 python infer.py \
      --prompt "A red panda sipping tea in a sunlit kitchen, cinematic lighting" \
      --frames 73 --steps 50 --out out/smoke
"""

from __future__ import annotations

import argparse
import sys

from h3_gv100.pipeline import H3Engine
from h3_gv100.schemas import CreateVideoRequest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="modelsweights/MiniMax-H3")
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", default=None, help="首帧参考图路径（图生视频）")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=544)
    p.add_argument("--frames", type=int, default=124, help="会自动对齐到 17n+5")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--device", default="cuda:0", help="共享机先看 nvidia-smi 哪张空")
    p.add_argument("--dit-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--turbo-lora", default=None,
                   help="ModelTC Minimax-H3-Turbo 的 lora 路径（用法以该仓库 examples 为准）")
    p.add_argument("--out", default="out/h3")
    args = p.parse_args()

    image_b64 = None
    if args.image:
        from pathlib import Path

        from h3_gv100.common import b64encode

        image_b64 = b64encode(Path(args.image).read_bytes())

    req = CreateVideoRequest(
        prompt=args.prompt,
        image_b64=image_b64,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
        fps=args.fps,
    ).normalized()
    print(f"[cfg] {req.width}x{req.height}, {req.frames} frames "
          f"(~{req.frames / req.fps:.1f}s), {req.steps} steps, seed {req.seed}")

    engine = H3Engine(
        model_path=args.model,
        device=args.device,
        dit_dtype=args.dit_dtype,
        turbo_lora=args.turbo_lora,
    )
    engine.load()  # 内含 env_probe() 自检

    # generate() 会把产物写到 <work_dir>/h3.*，对齐成 --out 指定的前缀
    from pathlib import Path

    out = Path(args.out)
    work_dir = out.parent
    prefix = out.name
    result = engine.generate(req, work_dir)
    if prefix != "h3":  # 重命名到请求的前缀
        renamed = {}
        for kind, path in result["artifacts"].items():
            src = Path(path)
            dst = src.with_name(src.name.replace("h3.", f"{prefix}.", 1))
            src.rename(dst)
            renamed[kind] = str(dst)
        result["artifacts"] = renamed
    for kind, path in result["artifacts"].items():
        print(f"[done] {kind}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

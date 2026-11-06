#!/usr/bin/env python3
"""h3-server — 在 GV100 工作站上启动 H3 API 服务。

推荐启动方式（双卡挂 NUMA 0，共享机先 nvidia-smi 确认空卡）：

    numactl --cpunodebind=0 --membind=0 h3-server --device cuda:0 --port 8010

长任务请 nohup / screen / tmux 挂载。
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser(description="MiniMax H3 API server (GV100)")
    p.add_argument("--host", default=os.environ.get("H3_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("H3_PORT", "8010")))
    p.add_argument("--model", default=None, help="模型路径（默认 env H3_MODEL_PATH 或 modelsweights/MiniMax-H3）")
    p.add_argument("--device", default=None, help="cuda:0 / cuda:1，共享机先看 nvidia-smi")
    p.add_argument("--dit-dtype", default=None, choices=["float16", "float32"],
                   help="float32 仅作排障开关")
    p.add_argument("--reserve", default=None, help="显存预留，默认 4GB")
    p.add_argument("--turbo-lora", default=None, help="ModelTC Minimax-H3-Turbo LoRA 路径（M1）")
    args = p.parse_args()

    import uvicorn

    from h3_gv100.server import ServerSettings, create_app

    settings = ServerSettings(
        model_path=args.model,
        device=args.device,
        dit_dtype=args.dit_dtype,
        reserve=args.reserve,
        turbo_lora=args.turbo_lora,
    )
    print(f"[h3-server] model={settings.model_path} device={settings.device} "
          f"dtype={settings.dit_dtype} out={settings.out_dir}")
    app = create_app(settings)
    # workers 恒为 1：GPU/引擎是进程内单例
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

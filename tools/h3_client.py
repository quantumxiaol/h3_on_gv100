#!/usr/bin/env python3
"""h3-client — 本机调用 GV100 上的 H3 API。

子命令：
  health                          查看服务端引擎/队列状态
  submit  --prompt ... [--image keyframe.png] [--wait]   提交图生视频
  status  JOB_ID                  查询状态
  wait    JOB_ID                  阻塞等到完成
  download JOB_ID [--kind mp4] [--out path]              下载产物

配置来自环境变量或 .env：H3_API_BASE_URL、H3_API_KEY（可选）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _client(args):
    from dotenv import load_dotenv

    load_dotenv()
    import os

    from h3_gv100.client import H3Client

    base = args.base_url or os.environ.get("H3_API_BASE_URL")
    if not base:
        sys.exit("缺少 H3_API_BASE_URL（.env 或 --base-url）")
    return H3Client(base, api_key=args.api_key or os.environ.get("H3_API_KEY") or None)


def _print_poll(info: dict) -> None:
    state = info.get("status")
    elapsed = info.get("elapsed_s")
    msg = f"[{time.strftime('%H:%M:%S')}] {state}"
    if elapsed:
        msg += f"  已运行 {elapsed:.0f}s"
    print(msg, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="H3 API client")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")

    s = sub.add_parser("submit")
    s.add_argument("--prompt", required=True)
    s.add_argument("--image", default=None, help="首帧参考图路径（图生视频）")
    s.add_argument("--width", type=int, default=960)
    s.add_argument("--height", type=int, default=544)
    s.add_argument("--frames", type=int, default=73)
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--wait", action="store_true", help="提交后轮询到完成并下载")
    s.add_argument("--out", default="out/client")

    s = sub.add_parser("status")
    s.add_argument("job_id")

    s = sub.add_parser("wait")
    s.add_argument("job_id")
    s.add_argument("--interval", type=float, default=30.0)

    s = sub.add_parser("download")
    s.add_argument("job_id")
    s.add_argument("--kind", default="mp4", choices=["mp4", "video_only", "audio"])
    s.add_argument("--out", default=None, help="输出文件路径，默认 out/client/<job>.<kind>")

    args = p.parse_args()
    client = _client(args)

    if args.cmd == "health":
        import json

        print(json.dumps(client.health(), indent=2, ensure_ascii=False))
        return

    if args.cmd == "submit":
        from h3_gv100.common import b64encode
        from h3_gv100.schemas import CreateVideoRequest

        image_b64 = b64encode(Path(args.image).read_bytes()) if args.image else None
        req = CreateVideoRequest(
            prompt=args.prompt,
            image_b64=image_b64,
            width=args.width,
            height=args.height,
            frames=args.frames,
            steps=args.steps,
            seed=args.seed,
        )
        job_id = client.submit(req)
        print(f"job_id: {job_id}")
        if args.wait:
            client.wait(job_id, on_poll=_print_poll)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{job_id}.mp4"
            path.write_bytes(client.download(job_id, "mp4"))
            print(f"saved: {path}")
        return

    if args.cmd == "status":
        import json

        print(json.dumps(client.status(args.job_id), indent=2, ensure_ascii=False))
        return

    if args.cmd == "wait":
        client.wait(args.job_id, interval=args.interval, on_poll=_print_poll)
        print("done")
        return

    if args.cmd == "download":
        path = Path(args.out) if args.out else Path("out/client") / f"{args.job_id}.{args.kind}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(client.download(args.job_id, args.kind))
        print(f"saved: {path}")
        return


if __name__ == "__main__":
    main()

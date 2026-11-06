"""H3Engine — MiniMax H3 在 GV100 上的推理封装。

Volta 特化（与 AGENT_PROMPT 锁定路线一致）：
  1. 全链路 FP16（官方发布精度 BF16，V100 无 BF16 单元故 cast 到 FP16 加载，
     动态范围收窄有溢出风险——花屏/NaN 时用 dit_dtype=float32 定位）
  2. SDPA attention；严禁 flash-attn / SageAttention（Ampere 起步，Volta 会崩）
  3. diffusers ModularPipeline + ComponentsManager.enable_auto_cpu_offload
     权重 ~123GB FP16 驻 512GB 内存，逐组件换入 32GB 显存
  4. H3 是 CFG 蒸馏模型：不加 negative prompt / guidance

注意：ModularPipeline 的调用细节（参数名/返回结构）可能随 diffusers 版本漂移。
本模块在调用前用 inspect 读取所装版本的签名做参数适配；若适配失败会抛出
带签名信息的异常，按 README「排障」节对照 site-packages 源码改 _PARAM_CANDIDATES。
"""

from __future__ import annotations

import inspect
import json
import math
import subprocess
import threading
import time
from pathlib import Path

from .schemas import CreateVideoRequest

# pipe.__call__ 参数名候选（按优先级）。image/last_image 已对 diffusers v0.42.1 源码核实；
# 其余以所装版本签名（或 pipe.doc）为准。
_PARAM_CANDIDATES: dict[str, tuple[str, ...]] = {
    "prompt": ("prompt",),
    "image": ("image", "first_frame", "first_frames", "condition_image", "reference_image", "images"),
    "last_image": ("last_image", "last_frame", "final_frame"),
    "height": ("height",),
    "width": ("width",),
    "num_frames": ("num_frames", "frames"),
    "num_inference_steps": ("num_inference_steps", "steps"),
    "generator": ("generator", "generators"),
    "output": ("output", "outputs"),
}

# 请求传给 pipe 的 output 字段
_OUTPUT_FIELDS = ["videos", "audio", "sampling_rate"]


def env_probe() -> None:
    """环境自检：torch/CUDA/arch list/FP16 matmul 探针。挂了先把输出发出来再排查。"""
    import torch

    print("=" * 60)
    print(f"torch {torch.__version__}  cuda-runtime {torch.version.cuda}")
    print(f"arch list: {torch.cuda.get_arch_list()}")
    print(
        f"gpu: {torch.cuda.get_device_name(0)}  "
        f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
    )
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    y = (x @ x).sum().item()
    assert not math.isnan(y), "FP16 matmul returned NaN"
    print(f"fp16 matmul probe: ok ({y:.1f})")
    print("=" * 60)


def _pick(container, *keys):
    """从 dict 或对象属性里防御性取字段。"""
    for k in keys:
        if isinstance(container, dict) and k in container:
            return container[k]
        if hasattr(container, k):
            return getattr(container, k)
    return None


def save_outputs(result, out_prefix: str, fps: int = 24) -> dict[str, str]:
    """从 pipe 返回里健壮地拆出视频/音频并 ffmpeg 合流。返回 {kind: path}。"""
    import numpy as np
    import torch

    videos = _pick(result, "videos", "video")
    audios = _pick(result, "audio", "audios")
    sr = _pick(result, "sampling_rate", "sample_rate") or 32000

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    if videos is not None:
        v = videos[0] if isinstance(videos, (list, tuple)) else videos
        if isinstance(v, torch.Tensor):
            v = v.float().cpu().numpy()
        v = np.asarray(v)
        if v.ndim == 4 and v.shape[-1] not in (1, 3, 4):  # [C,T,H,W] -> [T,H,W,C]
            v = v.transpose(1, 2, 3, 0)
        if v.dtype != np.uint8:
            v = v - v.min()
            v = (v / max(v.max(), 1e-6) * 255).astype(np.uint8)
        if not np.isfinite(v).all():
            raise RuntimeError("视频帧含 NaN/Inf —— FP16 溢出，用 --dit-dtype float32 复现排查")
        import imageio.v2 as imageio

        video_path = f"{out_prefix}.video_only.mp4"
        imageio.mimsave(video_path, v, fps=fps, codec="libx264", quality=8)
        saved["video_only"] = video_path
        print(f"[save] video -> {video_path}  ({len(v)} frames @ {fps}fps)")

    if audios is not None:
        a = audios[0] if isinstance(audios, (list, tuple)) else audios
        if isinstance(a, torch.Tensor):
            a = a.float().cpu().numpy()
        a = np.asarray(a)
        import soundfile as sf

        audio_path = f"{out_prefix}.audio.wav"
        sf.write(audio_path, a.T if a.ndim > 1 else a, int(sr))
        saved["audio"] = audio_path
        print(f"[save] audio -> {audio_path}  ({int(sr)} Hz)")

    if "video_only" in saved and "audio" in saved:
        final = f"{out_prefix}.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", saved["video_only"],
                    "-i", saved["audio"],
                    "-c:v", "copy", "-c:a", "aac", "-shortest", final,
                ],
                check=True,
                capture_output=True,
            )
            saved["mp4"] = final
            print(f"[save] muxed -> {final}")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(
                "[warn] ffmpeg 不可用或合流失败，请手动执行：\n"
                f"  ffmpeg -i {saved['video_only']} -i {saved['audio']} "
                f"-c:v copy -c:a aac -shortest {final}"
            )
    elif "video_only" in saved:
        saved["mp4"] = saved["video_only"]

    return saved


class H3Engine:
    """H3 推理引擎（进程内单例）。线程安全：加载与推理各自串行。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dit_dtype: str = "float16",
        reserve: str = "4GB",
        turbo_lora: str | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.dit_dtype = dit_dtype
        self.reserve = reserve
        self.turbo_lora = turbo_lora

        self._pipe = None
        self._call_params: set[str] | None = None
        self._accepts_kwargs = False
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.ready = threading.Event()
        self.error: str | None = None

    # ── 加载 ────────────────────────────────────────────────
    def load(self) -> None:
        with self._load_lock:
            if self._pipe is not None:
                return
            try:
                self._load_inner()
            except Exception as e:  # 任何加载失败都要记录并暴露给 /health
                self.error = f"{type(e).__name__}: {e}"
                raise
            self.ready.set()

    def _load_inner(self) -> None:
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        env_probe()

        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(
            self.model_path, workflow="fl2va", components_manager=manager
        )
        dtype = {"float16": torch.float16, "float32": torch.float32}[self.dit_dtype]
        pipe.load_components(dtype=dtype)  # V100: 严禁 bf16
        # SDPA 是 SM70 的安全选择；不要在环境里装 flash-attn
        manager.enable_auto_cpu_offload(device=self.device, memory_reserve_margin=self.reserve)

        if self.turbo_lora:
            # TODO(ModelTC/Minimax-H3-Turbo): 以官方 examples 的调用为准，可能不是 load_lora_weights
            pipe.load_lora_weights(self.turbo_lora)
            print(f"[lora] loaded {self.turbo_lora}")

        sig = inspect.signature(pipe.__call__)
        self._call_params = set(sig.parameters)
        self._accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        print(f"[engine] pipe.__call__ params: {sorted(self._call_params)}")
        doc = getattr(pipe, "doc", None)
        if doc:  # MiniMaxH3ModularPipeline.doc：当前 workflow 的精确输入/输出说明
            print(f"[engine] pipe.doc:\n{doc}")
        self._pipe = pipe

    # ── 推理 ────────────────────────────────────────────────
    def _build_kwargs(self, req: CreateVideoRequest, image, last_image=None) -> dict:
        """按所装 diffusers 版本的签名组装调用参数。"""
        import torch

        assert self._call_params is not None
        values = {
            "prompt": req.prompt,
            "image": image,
            "last_image": last_image,
            "height": req.height,
            "width": req.width,
            "num_frames": req.frames,
            "num_inference_steps": req.steps,
            "generator": torch.Generator(device="cpu").manual_seed(req.seed),
            "output": _OUTPUT_FIELDS,
        }
        kwargs: dict = {}
        missing: list[str] = []
        for logical, value in values.items():
            if value is None:
                continue
            for name in _PARAM_CANDIDATES[logical]:
                if name in self._call_params:
                    kwargs[name] = value
                    break
            else:
                if self._accepts_kwargs:
                    kwargs[_PARAM_CANDIDATES[logical][0]] = value
                elif logical in ("prompt", "num_frames", "num_inference_steps"):
                    missing.append(logical)
        for logical, b64 in (("image", req.image_b64), ("last_image", req.last_image_b64)):
            if b64 and not any(k in kwargs for k in _PARAM_CANDIDATES[logical]):
                raise RuntimeError(
                    f"当前 diffusers 版本的 pipe.__call__ 不接受任何已知的 {logical} 参数"
                    f"（候选 {_PARAM_CANDIDATES[logical]}），签名：{sorted(self._call_params)}。"
                    "请按 site-packages 源码（或 pipe.doc）更新 h3_gv100/pipeline.py 的 _PARAM_CANDIDATES。"
                )
        if missing:
            raise RuntimeError(
                f"pipe.__call__ 缺少必要参数 {missing}，签名：{sorted(self._call_params)}。"
                "请按 site-packages 源码更新 _PARAM_CANDIDATES。"
            )
        return kwargs

    def generate(self, req: CreateVideoRequest, work_dir: str | Path) -> dict:
        """同步执行一次生成，返回 {artifacts, meta}。调用方负责串行（JobManager 单 worker）。"""
        if not self.ready.is_set():
            raise RuntimeError(f"engine 未就绪（error={self.error}）")
        import torch

        req = req.normalized()
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        image = None
        if req.image_b64:
            image = self._decode_image(req.image_b64, req.width, req.height)
        last_image = None
        if req.last_image_b64:
            last_image = self._decode_image(req.last_image_b64, req.width, req.height)

        kwargs = self._build_kwargs(req, image, last_image)
        print(f"[engine] call kwargs: {sorted(kwargs)}")

        t0 = time.time()
        with self._infer_lock:
            result = self._pipe(**kwargs)
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"[engine] inference {elapsed:.0f}s")

        artifacts = save_outputs(result, str(work_dir / "h3"), fps=req.fps)

        meta = {
            "width": req.width,
            "height": req.height,
            "frames": req.frames,
            "steps": req.steps,
            "seed": req.seed,
            "fps": req.fps,
            "elapsed_s": round(elapsed, 1),
        }
        (work_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return {"artifacts": artifacts, "meta": meta}

    @staticmethod
    def _decode_image(image_b64: str, width: int, height: int):
        """base64 -> PIL Image，等比裁剪填充到目标网格（首帧条件图）。"""
        from io import BytesIO

        from PIL import Image, ImageOps

        from .common import b64decode

        raw = b64decode(image_b64)
        img = Image.open(BytesIO(raw)).convert("RGB")
        return ImageOps.fit(img, (width, height), Image.Resampling.LANCZOS)

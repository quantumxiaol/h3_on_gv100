"""H3 API 服务（跑在 GV100 工作站上）。

- POST /v1/videos            提交图生视频任务（base64 首帧图 + prompt）-> 202 {job_id}
- GET  /v1/videos/{job_id}   轮询状态
- GET  /v1/videos/{job_id}/content   取产物（base64 mp4 + wav）
- GET  /v1/videos/{job_id}/download?kind=mp4|video_only|audio  直接下载文件
- GET  /health               引擎状态（loading/ready/error）+ 队列长度

推理是 20–40 分钟级长任务，所以是异步 job 模式；单 worker 串行执行（GPU 单例）。
引擎在后台线程加载（~123GB FP16 进内存需数分钟），加载期间 submit 仍接受任务，
排队等就绪后执行。

共享机纪律：启动前先 nvidia-smi 确认目标卡空闲；启动命令见 README（numactl 前缀）。
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from .common import b64encode
from .pipeline import H3Engine

logger = logging.getLogger(__name__)
from .schemas import (
    CreateVideoRequest,
    CreateVideoResponse,
    HealthResponse,
    JobInfo,
    JobState,
    VideoContent,
)


class ServerSettings:
    """全部可用环境变量覆盖（见 .env.example）。"""

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        dit_dtype: str | None = None,
        reserve: str | None = None,
        turbo_lora: str | None = None,
        out_dir: str | None = None,
        api_key: str | None = None,
        max_frames: int | None = None,
    ):
        self.model_path = model_path or os.environ.get("H3_MODEL_PATH", "modelsweights/MiniMax-H3")
        self.device = device or os.environ.get("H3_DEVICE", "cuda:0")
        self.dit_dtype = dit_dtype or os.environ.get("H3_DIT_DTYPE", "float16")
        self.reserve = reserve or os.environ.get("H3_RESERVE", "4GB")
        self.turbo_lora = turbo_lora if turbo_lora is not None else os.environ.get("H3_TURBO_LORA") or None
        self.out_dir = Path(out_dir or os.environ.get("H3_OUT_DIR", "out/jobs"))
        self.api_key = api_key if api_key is not None else os.environ.get("H3_API_KEY") or None
        self.max_frames = max_frames or int(os.environ.get("H3_MAX_FRAMES", "345"))  # ~14.4s @24fps


class JobRecord:
    def __init__(self, job_id: str, request: CreateVideoRequest):
        self.job_id = job_id
        self.request = request
        self.status = JobState.QUEUED
        self.detail = ""
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.outputs: dict[str, str] = {}
        self.meta: dict = {}
        self.work_dir: Path | None = None

    def to_info(self) -> JobInfo:
        elapsed = None
        if self.started_at:
            elapsed = round((self.finished_at or time.time()) - self.started_at, 1)
        return JobInfo(
            job_id=self.job_id,
            status=self.status,
            detail=self.detail,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            request=self.request,
            outputs=self.outputs,
            elapsed_s=elapsed,
        )


class JobManager:
    """内存 job 表 + 单 worker 线程。产物落盘 out/jobs/<id>/，重启后文件仍在。"""

    def __init__(self, engine: H3Engine, out_dir: Path):
        self.engine = engine
        self.out_dir = out_dir
        self.jobs: dict[str, JobRecord] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="h3-worker")
        self._worker.start()

    def submit(self, req: CreateVideoRequest) -> JobRecord:
        job = JobRecord(uuid.uuid4().hex[:12], req.normalized())
        job.work_dir = self.out_dir / job.job_id
        with self._lock:
            self.jobs[job.job_id] = job
        self._q.put(job.job_id)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self.jobs.get(job_id)

    def counts(self) -> tuple[int, int]:
        with self._lock:
            queued = sum(1 for j in self.jobs.values() if j.status is JobState.QUEUED)
            running = sum(1 for j in self.jobs.values() if j.status is JobState.RUNNING)
        return queued, running

    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self.engine.ready.wait()  # 引擎加载完才开跑；加载失败则快速失败
                if self.engine.error:
                    raise RuntimeError(f"engine 加载失败: {self.engine.error}")
                job.status = JobState.RUNNING
                job.started_at = time.time()
                result = self.engine.generate(job.request, job.work_dir)
                job.outputs = {
                    k: str(Path(v).relative_to(job.work_dir)) for k, v in result["artifacts"].items()
                }
                job.meta = result["meta"]
                job.status = JobState.DONE
                job.detail = "ok"
            except Exception as e:  # noqa: BLE001 —— 任何失败都要落到 job 状态里
                job.status = JobState.ERROR
                job.detail = f"{type(e).__name__}: {e}"
            finally:
                job.finished_at = time.time()


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    engine = H3Engine(
        model_path=settings.model_path,
        device=settings.device,
        dit_dtype=settings.dit_dtype,
        reserve=settings.reserve,
        turbo_lora=settings.turbo_lora,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 后台线程加载 ~123GB 权重，uvicorn 端口立刻可用（健康检查报 loading）
        threading.Thread(target=_load_safe, daemon=True, name="h3-loader").start()
        yield

    def _load_safe() -> None:
        try:
            engine.load()
        except Exception:  # engine.error 已记录，/health 会暴露
            logger.exception("H3 engine 加载失败")

    app = FastAPI(title="h3-on-gv100", version="0.1.0", lifespan=lifespan)
    manager = JobManager(engine, settings.out_dir)

    async def auth(authorization: str | None = Header(default=None)) -> None:
        if settings.api_key and authorization != f"Bearer {settings.api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        queued, running = manager.counts()
        state = "error" if engine.error else ("ready" if engine.ready.is_set() else "loading")
        return HealthResponse(
            engine=state,
            engine_detail=engine.error or "",
            gpu=_gpu_name(),
            queued=queued,
            running=running,
        )

    @app.post("/v1/videos", response_model=CreateVideoResponse, status_code=202,
              dependencies=[Depends(auth)])
    async def create_video(req: CreateVideoRequest) -> CreateVideoResponse:
        if engine.error:
            raise HTTPException(status_code=503, detail=f"engine 加载失败: {engine.error}")
        if req.frames > settings.max_frames:
            raise HTTPException(
                status_code=422,
                detail=f"frames 超过上限 {settings.max_frames}（H3-Base 最长约 15s）",
            )
        job = manager.submit(req)
        return CreateVideoResponse(job_id=job.job_id, status=job.status)

    @app.get("/v1/videos", response_model=list[JobInfo], dependencies=[Depends(auth)])
    async def list_videos() -> list[JobInfo]:
        with manager._lock:
            jobs = sorted(manager.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_info() for j in jobs[:50]]

    @app.get("/v1/videos/{job_id}", response_model=JobInfo, dependencies=[Depends(auth)])
    async def get_video(job_id: str) -> JobInfo:
        job = _must_get(manager, job_id)
        return job.to_info()

    @app.get("/v1/videos/{job_id}/content", response_model=VideoContent,
             dependencies=[Depends(auth)])
    async def get_content(job_id: str) -> VideoContent:
        job = _must_get(manager, job_id)
        if job.status is not JobState.DONE:
            raise HTTPException(status_code=409, detail=f"job 状态 {job.status}，未完成")

        def read_b64(kind: str) -> str | None:
            rel = job.outputs.get(kind)
            if not rel:
                return None
            path = job.work_dir / rel
            return b64encode(path.read_bytes()) if path.exists() else None

        return VideoContent(
            job_id=job.job_id,
            video_b64=read_b64("mp4"),
            video_only_b64=read_b64("video_only"),
            audio_b64=read_b64("audio"),
            sampling_rate=32000 if job.outputs.get("audio") else None,
            meta=job.meta,
        )

    @app.get("/v1/videos/{job_id}/download", dependencies=[Depends(auth)])
    async def download(job_id: str, kind: str = "mp4") -> FileResponse:
        job = _must_get(manager, job_id)
        if job.status is not JobState.DONE:
            raise HTTPException(status_code=409, detail=f"job 状态 {job.status}，未完成")
        rel = job.outputs.get(kind)
        if not rel:
            raise HTTPException(status_code=404, detail=f"job 没有 {kind} 产物: {sorted(job.outputs)}")
        path = job.work_dir / rel
        if not path.exists():
            raise HTTPException(status_code=410, detail="产物文件已不存在")
        return FileResponse(path, filename=path.name)

    return app


def _must_get(manager: JobManager, job_id: str) -> JobRecord:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未知 job_id: {job_id}")
    return job


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001 —— torch 未装（纯客户端环境）等情况不算错误
        logger.debug("gpu name unavailable: %s", e)
    return ""


app = create_app()

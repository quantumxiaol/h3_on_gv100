"""API 请求/响应模型，服务端与客户端共用（pydantic，无重依赖）。

传输约定：文件一律 base64。视频生成是 20–40 分钟级的长任务，
因此是「提交 job -> 轮询 -> 取内容」的异步模式（与 M3 官方 API 精修管道同构）。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .common import MAX_FRAMES, MIN_FRAMES, snap_frames, snap_to_32


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class CreateVideoRequest(BaseModel):
    """图生视频请求。image_b64 为首帧参考图（图生视频）；不传则退化为文生视频。"""

    prompt: str
    image_b64: str | None = Field(default=None, description="首帧参考图 PNG/JPEG/WebP 的 base64")
    last_image_b64: str | None = Field(default=None, description="尾帧参考图 base64；与 image_b64 组合成 FL2VA 首尾帧，单独传则为 L2VA")
    width: int = 960
    height: int = 544
    frames: int = Field(
        default=MIN_FRAMES,
        description=f"向上对齐 17n+5 并夹到 [{MIN_FRAMES}, {MAX_FRAMES}]（5–15s @24fps）",
    )
    steps: int = Field(default=50, description="基线 50；Turbo LoRA 后 8")
    seed: int = 42
    fps: int = 24

    def normalized(self) -> CreateVideoRequest:
        return self.model_copy(
            update={
                "frames": snap_frames(self.frames),
                "width": snap_to_32(self.width),
                "height": snap_to_32(self.height),
            }
        )


class CreateVideoResponse(BaseModel):
    job_id: str
    status: JobState


class JobInfo(BaseModel):
    job_id: str
    status: JobState
    detail: str = ""
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    request: CreateVideoRequest | None = None
    outputs: dict[str, str] = Field(default_factory=dict, description="产物文件名 -> 相对路径")
    elapsed_s: float | None = None


class VideoContent(BaseModel):
    """生成产物（base64）。video_b64 是 ffmpeg 合流后的成片 mp4。"""

    job_id: str
    video_b64: str | None = None
    video_only_b64: str | None = None
    audio_b64: str | None = None
    sampling_rate: int | None = None
    meta: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    engine: str = Field(description="loading | ready | error")
    engine_detail: str = ""
    gpu: str = ""
    queued: int = 0
    running: int = 0

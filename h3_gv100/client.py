"""本机客户端（mac，无 torch）：

- GenImgClient: OpenAI 兼容生图 API（参考 dsh-umamusume-agent 的 image.ts）
    无参考图 -> POST {base}/images/generations (JSON)
    有参考图 -> POST {base}/images/edits (multipart, image/image[])
    响应 data[0].b64_json 优先，url 次之；edits 404 时回退 generations
- H3Client:     GV100 上 H3 API 的提交/轮询/取回（base64）
- imagine():    整体工作流——一个想法 -> 生图（首帧）-> 图生视频 -> 本地落盘
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .common import b64decode, b64encode, sniff_image_media_type
from .schemas import CreateVideoRequest, JobState

MAX_IMAGE_BYTES = 32 * 1024 * 1024


class ApiHttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status


def _check(resp: httpx.Response) -> httpx.Response:
    if resp.status_code >= 400:
        raise ApiHttpError(resp.status_code, resp.text)
    return resp


# ── 生图 API ────────────────────────────────────────────────
@dataclass
class GeneratedImage:
    data: bytes
    media_type: str


class GenImgClient:
    """OpenAI 兼容图像 API（gpt-image-2.5 等）。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        size: str = "1536x1024",
        quality: str = "high",
        references: list[tuple[str, bytes]] | None = None,
        reference_description: str | None = None,
    ) -> GeneratedImage:
        try:
            return self._request(prompt, size=size, quality=quality, references=references)
        except ApiHttpError as e:
            # 网关不支持 /images/edits（404）时回退 generations + 文字描述；其他错误不重复付费请求
            if not references or e.status != 404:
                raise
            fallback_prompt = (
                f"{reference_description}\n{prompt}" if reference_description else prompt
            )
            return self._request(fallback_prompt, size=size, quality=quality, references=None)

    def _request(
        self,
        prompt: str,
        *,
        size: str,
        quality: str,
        references: list[tuple[str, bytes]] | None,
    ) -> GeneratedImage:
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
            "user-agent": "h3-on-gv100/0.1.0",
        }
        fields = {
            "model": self.model,
            "prompt": prompt,
            "n": "1",
            "size": size,
            "quality": quality,
            "response_format": "b64_json",
        }
        if references:
            url = f"{self.base_url}/images/edits"
            field = "image" if len(references) == 1 else "image[]"
            files = [(field, (name, data, "image/png")) for name, data in references]
            resp = httpx.post(url, headers=headers, data=fields, files=files, timeout=self.timeout)
        else:
            url = f"{self.base_url}/images/generations"
            headers["content-type"] = "application/json"
            resp = httpx.post(url, headers=headers, json=fields, timeout=self.timeout)
        payload = _check(resp).json()
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("图像 API 未返回图片。")
        item = data[0]
        if isinstance(item.get("b64_json"), str) and item["b64_json"]:
            return self._decode(b64decode(item["b64_json"]))
        if isinstance(item.get("url"), str) and item["url"]:
            img_resp = _check(httpx.get(item["url"], timeout=self.timeout, follow_redirects=True))
            length = int(img_resp.headers.get("content-length") or 0)
            if length > MAX_IMAGE_BYTES:
                raise RuntimeError("生成图片超过 32 MB 限制。")
            return self._decode(img_resp.content)
        raise RuntimeError("图像 API 响应缺少 b64_json 或 url。")

    @staticmethod
    def _decode(data: bytes) -> GeneratedImage:
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise RuntimeError("生成图片为空或超过 32 MB 限制。")
        return GeneratedImage(data=data, media_type=sniff_image_media_type(data))


# ── H3 API（GV100）─────────────────────────────────────────
class H3Client:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"accept": "application/json"}
        if api_key:
            self.headers["authorization"] = f"Bearer {api_key}"

    def health(self) -> dict:
        return _check(httpx.get(f"{self.base_url}/health", headers=self.headers, timeout=30)).json()

    def submit(self, req: CreateVideoRequest) -> str:
        resp = httpx.post(
            f"{self.base_url}/v1/videos",
            headers=self.headers,
            json=req.model_dump(),
            timeout=self.timeout,
        )
        return _check(resp).json()["job_id"]

    def status(self, job_id: str) -> dict:
        return _check(
            httpx.get(f"{self.base_url}/v1/videos/{job_id}", headers=self.headers, timeout=30)
        ).json()

    def wait(
        self,
        job_id: str,
        *,
        interval: float = 30.0,
        timeout: float | None = None,
        on_poll=None,
    ) -> dict:
        """轮询直到 done/error。timeout=None 表示不限（H3 单条 20–40 分钟）。"""
        t0 = time.time()
        while True:
            info = self.status(job_id)
            state = info["status"]
            if on_poll:
                on_poll(info)
            if state == JobState.DONE:
                return info
            if state == JobState.ERROR:
                raise RuntimeError(f"远端生成失败: {info.get('detail')}")
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError(f"等待 {job_id} 超时（{timeout}s），任务仍在 {state}")
            time.sleep(interval)

    def content(self, job_id: str) -> dict:
        return _check(
            httpx.get(
                f"{self.base_url}/v1/videos/{job_id}/content",
                headers=self.headers,
                timeout=max(self.timeout, 600.0),  # base64 mp4 可能几十 MB
            )
        ).json()

    def download(self, job_id: str, kind: str = "mp4") -> bytes:
        return _check(
            httpx.get(
                f"{self.base_url}/v1/videos/{job_id}/download",
                headers=self.headers,
                params={"kind": kind},
                timeout=max(self.timeout, 600.0),
                follow_redirects=True,
            )
        ).content


# ── 整体工作流 ──────────────────────────────────────────────
def imagine(
    genimg: GenImgClient,
    h3: H3Client,
    idea: str,
    *,
    image_prompt: str | None = None,
    video_prompt: str | None = None,
    size: str = "1536x1024",
    quality: str = "high",
    width: int = 960,
    height: int = 544,
    frames: int = 124,
    steps: int = 50,
    seed: int = 42,
    out_dir: str | Path = "out/imagine",
    poll_interval: float = 30.0,
    on_event=None,
) -> dict[str, Path]:
    """想法 -> 生图（首帧）-> 图生视频 -> 本地落盘。返回产物路径。"""

    def emit(event: str, **kw):
        if on_event:
            on_event(event, **kw)

    out = Path(out_dir) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    # 1) 生图（视频首帧）
    img_prompt = image_prompt or idea
    emit("genimg.start", prompt=img_prompt)
    image = genimg.generate(img_prompt, size=size, quality=quality)
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}[
        image.media_type
    ]
    keyframe = out / f"keyframe.{ext}"
    keyframe.write_bytes(image.data)
    emit("genimg.done", path=str(keyframe))

    # 2) 图生视频（base64 传首帧 + prompt）
    req = CreateVideoRequest(
        prompt=video_prompt or idea,
        image_b64=b64encode(image.data),
        width=width,
        height=height,
        frames=frames,
        steps=steps,
        seed=seed,
    )
    job_id = h3.submit(req)
    emit("h3.submitted", job_id=job_id)
    (out / "request.json").write_text(
        json.dumps({**req.model_dump(exclude={"image_b64"}), "job_id": job_id}, indent=2, ensure_ascii=False)
    )

    h3.wait(job_id, interval=poll_interval, on_poll=lambda info: emit("h3.poll", info=info))

    # 3) 取回产物（base64）
    content = h3.content(job_id)
    saved: dict[str, Path] = {"keyframe": keyframe}
    if content.get("video_b64"):
        saved["video"] = out / "final.mp4"
        saved["video"].write_bytes(b64decode(content["video_b64"]))
    if content.get("audio_b64"):
        saved["audio"] = out / "audio.wav"
        saved["audio"].write_bytes(b64decode(content["audio_b64"]))
    emit("h3.done", paths={k: str(v) for k, v in saved.items()})
    return saved

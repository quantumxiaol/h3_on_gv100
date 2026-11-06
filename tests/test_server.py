"""服务端 API 测试：fake engine 替换 H3Engine（不加载 torch/权重），验证 job 闭环与鉴权。

在本机跑需要临时 `uv pip install fastapi`；GV100 上 `--extra server` 自带。
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import h3_gv100.server as server_mod

MP4 = b"\x00\x00\x00\x18ftypmp42fake-video"
WAV = b"RIFFfake-wave"


class FakeEngine:
    """模拟引擎：ready 立即置位，generate 写出假产物。"""

    def __init__(self, **kwargs):
        self.ready = threading.Event()
        self.ready.set()
        self.error = None
        self.kwargs = kwargs

    def load(self):
        return None

    def generate(self, req, work_dir):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "h3.mp4").write_bytes(MP4)
        (work_dir / "h3.audio.wav").write_bytes(WAV)
        time.sleep(0.05)  # 让 RUNNING 状态可被观察
        return {
            "artifacts": {"mp4": str(work_dir / "h3.mp4"), "audio": str(work_dir / "h3.audio.wav")},
            "meta": {"frames": req.frames, "elapsed_s": 0.05},
        }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "H3Engine", FakeEngine)
    settings = server_mod.ServerSettings(out_dir=str(tmp_path), api_key="secret")
    app = server_mod.create_app(settings)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["engine"] == "ready"


def test_auth_required(client):
    assert client.post("/v1/videos", json={"prompt": "x"}).status_code == 401
    assert client.get("/v1/videos/nope").status_code == 401


def test_job_roundtrip_base64(client):
    h = {"authorization": "Bearer secret"}
    png_b64 = base64.b64encode(b"\x89PNGfake").decode()
    r = client.post(
        "/v1/videos",
        headers=h,
        json={"prompt": "red panda", "image_b64": png_b64, "last_image_b64": png_b64,
              "frames": 80, "steps": 50},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    for _ in range(100):
        info = client.get(f"/v1/videos/{job_id}", headers=h).json()
        if info["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert info["status"] == "done", info.get("detail")
    assert info["request"]["frames"] == 124  # 服务端对齐 17n+5 并夹到 5–15s 窗口
    assert info["request"]["last_image_b64"] == png_b64  # FL2VA 尾帧透传
    assert set(info["outputs"]) == {"mp4", "audio"}

    content = client.get(f"/v1/videos/{job_id}/content", headers=h).json()
    assert base64.b64decode(content["video_b64"]) == MP4
    assert base64.b64decode(content["audio_b64"]) == WAV
    assert content["sampling_rate"] == 32000

    dl = client.get(f"/v1/videos/{job_id}/download?kind=mp4", headers=h)
    assert dl.status_code == 200 and dl.content == MP4


def test_frames_cap(client):
    h = {"authorization": "Bearer secret"}
    r = client.post("/v1/videos", headers=h, json={"prompt": "x", "frames": 9999})
    assert r.status_code == 422


def test_unknown_job(client):
    h = {"authorization": "Bearer secret"}
    assert client.get("/v1/videos/deadbeef", headers=h).status_code == 404

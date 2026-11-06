"""客户端链路测试：本地 stub HTTP 服务模拟生图 API 与 H3 API，不触网、不需要 torch。"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from h3_gv100.client import ApiHttpError, GenImgClient, H3Client
from h3_gv100.common import MAX_FRAMES, MIN_FRAMES, b64encode, snap_frames, snap_to_32
from h3_gv100.schemas import CreateVideoRequest

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zq2cAAAAASUVORK5CYII="
)
MP4 = b"\x00\x00\x00\x18ftypmp42fake-video"


def test_grid():
    # 向上对齐 17n+5 并夹到 5–15s 窗口 [124, 345]
    assert snap_frames(124) == 124
    assert snap_frames(125) == 141
    assert snap_frames(73) == MIN_FRAMES      # 低于 5s 被抬到下限
    assert snap_frames(80) == MIN_FRAMES      # 对齐到 90 仍不足 5s
    assert snap_frames(107) == MIN_FRAMES     # 4.46s 也低于窗口
    assert snap_frames(345) == MAX_FRAMES
    assert snap_frames(1000) == MAX_FRAMES
    assert snap_to_32(960) == 960
    assert snap_to_32(545) == 544
    assert snap_to_32(100) == 160


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静默
        pass

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "engine": "ready", "gpu": "GV100", "queued": 0, "running": 0})
        if self.path.startswith("/v1/videos/") and self.path.endswith("/content"):
            return self._send(200, {
                "job_id": "abc", "video_b64": b64encode(MP4),
                "video_only_b64": None, "audio_b64": None, "sampling_rate": None,
                "meta": {"elapsed_s": 1},
            })
        if self.path.startswith("/v1/videos/"):
            return self._send(200, {
                "job_id": "abc", "status": "done", "detail": "ok",
                "created_at": 0, "started_at": 0, "finished_at": 1,
                "request": None, "outputs": {"mp4": "h3.mp4"}, "elapsed_s": 1,
            })
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length)
        ctype = self.headers.get("content-type", "")
        if self.path == "/v1/images/generations":
            req = json.loads(body)
            assert req["response_format"] == "b64_json"
            assert req["model"] == "gpt-image-2.5"
            return self._send(200, {"data": [{"b64_json": b64encode(PNG)}]})
        if self.path == "/v1/images/edits":
            assert ctype.startswith("multipart/form-data")
            assert b'name="image[]"' in body or b'name="image"' in body
            return self._send(200, {"data": [{"b64_json": b64encode(PNG)}]})
        if self.path == "/v1/videos":
            req = json.loads(body)
            assert req["prompt"]
            return self._send(202, {"job_id": "abc", "status": "queued"})
        return self._send(404, {"error": "not found"})


def _serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


def test_genimg_generations_b64():
    c = GenImgClient(_serve() + "/v1", api_key="k", model="gpt-image-2.5")
    img = c.generate("a red panda", size="1536x1024", quality="high")
    assert img.data == PNG
    assert img.media_type == "image/png"


def test_genimg_edits_with_references():
    c = GenImgClient(_serve() + "/v1", api_key="k", model="gpt-image-2.5")
    img = c.generate("make it move", references=[("ref.png", PNG), ("ref2.png", PNG)])
    assert img.media_type == "image/png"


def test_genimg_404_fallback_to_generations():
    # edits 返回 404 的 stub：应回退 generations 且不再带参考图
    class Stub404(Stub):
        def do_POST(self):
            if self.path == "/v1/images/edits":
                return self._send(404, {"error": "no edits endpoint"})
            return self._send(200, {"data": [{"b64_json": b64encode(PNG)}]})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub404)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = GenImgClient(f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="k", model="gpt-image-2.5")
    img = c.generate("make it move", references=[("ref.png", PNG)],
                     reference_description="a red panda")
    assert img.media_type == "image/png"


def test_genimg_http_error_has_status():
    class Stub500(Stub):
        def do_POST(self):
            return self._send(500, {"error": "boom"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub500)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = GenImgClient(f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="k", model="m")
    try:
        c.generate("x")
        raise AssertionError("should raise")
    except ApiHttpError as e:
        assert e.status == 500


def test_h3_client_roundtrip():
    c = H3Client(_serve(), api_key="k")
    assert c.health()["engine"] == "ready"
    job_id = c.submit(CreateVideoRequest(prompt="hi", image_b64=b64encode(PNG), last_image_b64=b64encode(PNG)))
    assert job_id == "abc"
    info = c.wait(job_id, interval=0.01)
    assert info["status"] == "done"
    content = c.content(job_id)
    assert base64.b64decode(content["video_b64"]) == MP4


def test_request_normalization():
    req = CreateVideoRequest(prompt="x", frames=80, width=1000, height=545).normalized()
    assert req.frames == 124 and req.width == 992 and req.height == 544

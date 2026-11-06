"""纯函数工具：H3 网格约束 + base64 编解码。客户端/服务端共用，无重依赖。"""

from __future__ import annotations

import base64
import binascii
import math

# diffusers 文档：num_frames 向上对齐 17n+5，且时长须在 5–15 秒（24fps）
MIN_FRAMES = 124  # 17*7+5 ≈ 5.2s
MAX_FRAMES = 345  # 17*20+5 ≈ 14.4s


def snap_frames(frames: int) -> int:
    """H3 帧数：向上对齐 17n+5，并夹到 [124, 345]（5–15s @24fps）。"""
    n = max(0, math.ceil((frames - 5) / 17))
    snapped = 17 * n + 5
    return max(MIN_FRAMES, min(MAX_FRAMES, snapped))


def snap_to_32(v: int) -> int:
    """宽高对齐 32 的倍数（H3 基准 768p，短边 768）。"""
    return max(160, (v // 32) * 32)


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(text: str) -> bytes:
    """容错解码：允许 data: URL 前缀与空白字符。"""
    if "," in text and text.lstrip().startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"base64 解码失败: {e}") from e


def sniff_image_media_type(data: bytes) -> str:
    """按魔数识别图片类型，与参考实现 dsh-umamusume-agent 保持一致。"""
    if len(data) >= 8 and data[0] == 0x89 and data[1:4] == b"PNG":
        return "image/png"
    if len(data) >= 3 and data[0] == 0xFF and data[1] == 0xD8 and data[2] == 0xFF:
        return "image/jpeg"
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 6 and data[0:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    raise ValueError("不是受支持的 PNG、JPEG、WebP 或 GIF 图片。")

"""h3_on_gv100 — MiniMax H3 (33B) 图生视频 on 2x Quadro GV100.

包分两半：
- 服务端（GV100 工作站）：h3_gv100.pipeline / h3_gv100.server
- 客户端（本机 mac）：h3_gv100.client（生图 API + H3 API + 整体工作流）
"""

__version__ = "0.1.0"

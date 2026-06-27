"""FastAPI 中间件"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from ..utils.memory import MemoryMonitor

logger = logging.getLogger(__name__)


class MemoryCheckMiddleware(BaseHTTPMiddleware):
    """在处理推理请求前检查 GPU 内存是否充足。"""

    INFERENCE_PREFIX = "/v1/chat/completions"

    def __init__(self, app, monitor: "MemoryMonitor") -> None:
        super().__init__(app)
        self._monitor = monitor

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(self.INFERENCE_PREFIX):
            if not self._monitor.is_memory_available():
                stats = self._monitor.get_stats()
                return JSONResponse(
                    status_code=503,
                    content={"detail": f"GPU memory exhausted: {stats}"},
                )
        return await call_next(request)

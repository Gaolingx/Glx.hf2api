"""FastAPI 中间件"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class MemoryCheckMiddleware(BaseHTTPMiddleware):
    """
    在处理推理请求前检查 GPU 内存是否充足。

    monitor 通过 getter 延迟获取，避免 app 创建时 MemoryMonitor
    尚未实例化的问题（lifespan 启动后才初始化）。
    """

    INFERENCE_PREFIX = "/v1/chat/completions"

    def __init__(self, app, monitor_getter: Callable) -> None:
        """
        Args:
            monitor_getter: 无参可调用对象，返回当前 MemoryMonitor 实例或 None。
                            例如：lambda: _monitor
        """
        super().__init__(app)
        self._get_monitor = monitor_getter

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(self.INFERENCE_PREFIX):
            monitor = self._get_monitor()
            if monitor is not None and not monitor.is_memory_available():
                stats = monitor.get_stats()
                return JSONResponse(
                    status_code=503,
                    content={"detail": f"GPU memory exhausted: {stats}"},
                )
        return await call_next(request)

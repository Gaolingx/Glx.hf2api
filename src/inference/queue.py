"""
RequestQueue：asyncio 请求队列 + ThreadPoolExecutor 调度。

设计原则：
  - asyncio.Future 只在事件循环线程中创建和 resolve（通过 call_soon_threadsafe）
  - 工作线程只调用 engine，不碰 asyncio 原语
  - 流式请求：工作线程把 StreamGenerator 放入 Future，
    事件循环侧由 StreamHandler 异步消费
  - 非流式请求：工作线程把 GenerationResult 放入 Future
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union

from ..configs import ServerConfig
from .engine import InferenceEngine
from .types import GenerationResult, InferenceRequest, StreamGenerator

logger = logging.getLogger(__name__)


class RequestQueue:
    """
    asyncio 侧：
      enqueue()  → 把请求放入 asyncio.Queue，返回 Future（等待结果）
    工作线程侧：
      _worker()  → 从队列取请求，调用 engine，通过 call_soon_threadsafe 回写结果
    """

    def __init__(self, engine: InferenceEngine, server_cfg: ServerConfig) -> None:
        self._engine = engine
        self._limits = server_cfg.limits
        self._request_timeout: float = server_cfg.limits.request_timeout

        # 单工作线程：GPU 推理天然串行，不需要多线程
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(
            maxsize=server_cfg.limits.max_queue_size
        )
        self._running = False
        self._dispatch_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ #
    # 生命周期                                                              #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """在事件循环启动后调用（lifespan startup）。"""
        self._running = True
        self._dispatch_task = asyncio.ensure_future(self._dispatch_loop())
        logger.info("RequestQueue started.")

    async def stop(self) -> None:
        """优雅停止：等待队列清空后关闭线程池。"""
        self._running = False
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=True)
        logger.info("RequestQueue stopped.")

    # ------------------------------------------------------------------ #
    # 入队（asyncio 侧）                                                    #
    # ------------------------------------------------------------------ #

    async def enqueue(
        self, request: InferenceRequest
    ) -> Union[GenerationResult, StreamGenerator]:
        """
        将请求放入队列并等待结果。
        Future 在此处（事件循环线程）创建，保证线程安全。
        """
        loop = asyncio.get_running_loop()
        request.future = loop.create_future()
        request.loop = loop

        try:
            await asyncio.wait_for(
                self._queue.put(request),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            raise _queue_full_error()
        except asyncio.QueueFull:
            raise _queue_full_error()

        try:
            return await asyncio.wait_for(
                request.future,
                timeout=self._request_timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Request %s timed out.", request.request_id)
            from fastapi import HTTPException
            raise HTTPException(status_code=504, detail="Request timeout")

    # ------------------------------------------------------------------ #
    # 调度循环（asyncio 侧）                                                #
    # ------------------------------------------------------------------ #

    async def _dispatch_loop(self) -> None:
        """
        从队列中取请求，提交到线程池执行。
        使用 run_in_executor 而非直接 submit，保持与 asyncio 取消机制兼容。
        """
        logger.info("Dispatch loop started.")
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                request = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # 提交到线程池；不 await，立即接受下一个请求
            # （GPU 串行由单线程 executor 保证，不会有两个任务并发执行）
            loop.run_in_executor(
                self._executor,
                self._run_request,
                request,
            )

        logger.info("Dispatch loop exited.")

    # ------------------------------------------------------------------ #
    # 工作线程侧                                                            #
    # ------------------------------------------------------------------ #

    def _run_request(self, request: InferenceRequest) -> None:
        """在工作线程中执行，通过 call_soon_threadsafe 回写 Future。"""
        try:
            if request.stream:
                result = self._engine.generate_stream(
                    messages=request.messages,
                    gen_config=request.gen_config,
                    tools=request.tools,
                    open_thinking=request.open_thinking,
                )
            else:
                result = self._engine.generate(
                    messages=request.messages,
                    gen_config=request.gen_config,
                    tools=request.tools,
                    open_thinking=request.open_thinking,
                )
            self._resolve(request, result)
        except Exception as exc:
            self._reject(request, exc)

    def _resolve(self, request: InferenceRequest, result: Any) -> None:
        if request.loop and not request.future.done():
            request.loop.call_soon_threadsafe(request.future.set_result, result)

    def _reject(self, request: InferenceRequest, exc: Exception) -> None:
        logger.error(
            "Request %s failed: %s", request.request_id, exc, exc_info=True
        )
        if request.loop and not request.future.done():
            request.loop.call_soon_threadsafe(request.future.set_exception, exc)

    # ------------------------------------------------------------------ #
    # 监控                                                                  #
    # ------------------------------------------------------------------ #

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()


def _queue_full_error():
    from fastapi import HTTPException
    return HTTPException(status_code=503, detail="Server overloaded, queue is full")

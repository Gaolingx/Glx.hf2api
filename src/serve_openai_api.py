"""
服务入口：FastAPI app 定义、路由、lifespan。
全局状态仅在此文件维护，其他模块通过参数传递依赖。
"""
from __future__ import annotations

import argparse
import os
import logging
import sys
import time

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

__package__ = "src"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .api.middleware import MemoryCheckMiddleware
from .api.models import ChatRequest
from .configs import load_config
from .inference import InferenceEngine, RequestQueue, StreamHandler
from .inference.types import InferenceRequest
from .utils import MemoryMonitor, setup_logging

logger = logging.getLogger(__name__)

# ── 全局单例（仅 server.py 持有） ───────────────────────────────────────
_engine: InferenceEngine | None = None
_queue: RequestQueue | None = None
_monitor: MemoryMonitor | None = None
_config = None


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queue, _monitor

    logger.info("Server starting up...")

    _monitor = MemoryMonitor(_config.memory)
    _queue = RequestQueue(_engine, _config.server)
    _queue.start()

    import asyncio
    cleanup_task = asyncio.ensure_future(_monitor.run_cleanup_loop())

    logger.info("Server ready.")
    yield

    logger.info("Server shutting down...")
    cleanup_task.cancel()
    await _queue.stop()
    logger.info("Server stopped.")


# ── App 创建：中间件在此处注册，通过 lambda 延迟访问 _monitor ─────────────

app = FastAPI(lifespan=lifespan)

# lambda 捕获模块级变量名，每次调用时才读取 _monitor 的当前值
# lifespan 启动后 _monitor 被赋值，中间件即可正常使用
app.add_middleware(MemoryCheckMiddleware, monitor_getter=lambda: _monitor)


# ── 路由 ──────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        gen_defaults = {
            "temperature": _config.generation.temperature,
            "top_p": _config.generation.top_p,
            "top_k": _config.generation.top_k,
            "max_tokens": _config.generation.max_tokens or 8192,
            "repetition_penalty": _config.generation.repetition_penalty,
            "num_beams": _config.generation.num_beams,
            "do_sample": _config.generation.do_sample,
            "no_repeat_ngram_size": _config.generation.no_repeat_ngram_size,
        }

        inference_req = InferenceRequest(
            request_id=InferenceRequest.make_id(),
            messages=request.normalized_messages(),
            gen_config=request.build_generation_config(gen_defaults),
            tools=request.tools or None,
            open_thinking=request.get_open_thinking(),
            stream=request.stream,
        )

        result = await _queue.enqueue(inference_req)

        # ── 流式响应 ──────────────────────────────────────────────────
        if request.stream:
            handler = StreamHandler(open_thinking=request.get_open_thinking())

            async def event_stream():
                async for chunk in handler.run(result):
                    yield chunk

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # ── 非流式响应 ────────────────────────────────────────────────
        message: dict = {"role": "assistant", "content": result.content}
        if result.reasoning_content:
            message["reasoning_content"] = result.reasoning_content
        if result.tool_calls:
            message["tool_calls"] = result.tool_calls

        return {
            "id": f"chatcmpl-{inference_req.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if result.tool_calls else "stop",
            }],
            "usage": _compute_usage(inference_req, result),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled error in chat_completions: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _config.model.name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": _config.title,
        }],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": _engine is not None and _engine._loaded,
        "memory": _monitor.get_stats() if _monitor else {},
        "queue_size": _queue.queue_size if _queue else 0,
    }


@app.get("/metrics")
def metrics():
    stats = _monitor.get_stats() if _monitor else {}
    queue_size = _queue.queue_size if _queue else 0
    return (
        f"# HELP gpu_memory_allocated_gb GPU memory allocated\n"
        f"# TYPE gpu_memory_allocated_gb gauge\n"
        f"gpu_memory_allocated_gb {stats.get('allocated_gb', 0)}\n\n"
        f"# HELP gpu_memory_total_gb Total GPU memory\n"
        f"# TYPE gpu_memory_total_gb gauge\n"
        f"gpu_memory_total_gb {stats.get('total_gb', 0)}\n\n"
        f"# HELP gpu_memory_ratio GPU memory utilization\n"
        f"# TYPE gpu_memory_ratio gauge\n"
        f"gpu_memory_ratio {stats.get('ratio', 0)}\n\n"
        f"# HELP request_queue_size Current queue depth\n"
        f"# TYPE request_queue_size gauge\n"
        f"request_queue_size {queue_size}\n"
    )


# ── 工具函数 ──────────────────────────────────────────────────────────────

def _compute_usage(req: InferenceRequest, result) -> dict:
    """计算实际 token 用量。"""
    try:
        tok = _engine.tokenizer
        prompt_text = tok.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = len(tok.encode(prompt_text))
        completion_tokens = len(tok.encode(result.content or ""))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    except Exception as exc:
        logger.warning("Failed to compute token usage: %s", exc)
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ── 主入口 ────────────────────────────────────────────────────────────────

def main():
    global _engine, _config

    parser = argparse.ArgumentParser(description="LLMs API Server")
    parser.add_argument("--config", default="config/production.yaml")
    parser.add_argument("--device", type=str)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level", type=str)
    args = parser.parse_args()

    _config = load_config(args.config)

    log_level = args.log_level or _config.server.log_level
    setup_logging(_config.title, log_level)

    if args.device:
        _config.device.device = args.device
    if args.dtype:
        _config.device.dtype = args.dtype
    if args.gpus:
        _config.device.gpus = [int(g) for g in args.gpus.split(",")]

    host = args.host or _config.server.host
    port = args.port or _config.server.port

    logger.info("Initializing model...")
    _engine = InferenceEngine()
    _engine.load(_config.model, _config.device)

    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=_config.server.log_level.lower(),
        timeout_keep_alive=_config.server.timeout_keep_alive,
    )


if __name__ == "__main__":
    main()

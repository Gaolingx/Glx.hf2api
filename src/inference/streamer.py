"""
StreamHandler：将 engine 返回的 token generator 转换为 SSE 事件流。

职责边界：
  - 只做格式转换（token → SSE JSON chunk）
  - 不感知 asyncio.Future / 队列
  - 状态机：thinking → content → tool_call（可循环）
"""
from __future__ import annotations

import json
import logging
import re
import asyncio
import uuid
from typing import AsyncGenerator, Optional

from ..utils.parser import generate_tool_call_id
from .types import StreamGenerator

logger = logging.getLogger(__name__)

# <tool_call> 开始标签及其长度（用于安全缓冲计算）
_TC_OPEN = "<tool_call>"
_TC_OPEN_LEN = len(_TC_OPEN)
_TC_CLOSE = "</tool_call>"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _delta(content_key: str, content_value) -> str:
    return _sse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {content_key: content_value},
            "finish_reason": None,
        }],
    })


def _finish(reason: str) -> str:
    return _sse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": reason,
        }],
    })


class StreamHandler:
    """
    用法：
        handler = StreamHandler(open_thinking=True)
        async for chunk in handler.run(token_generator):
            yield chunk
    """

    def __init__(self, open_thinking: bool = False) -> None:
        self._open_thinking = open_thinking

    async def run(self, token_gen: StreamGenerator) -> AsyncGenerator[str, None]:
        """
        消费同步 token_gen（在事件循环线程中逐 token 推进），
        yield SSE 格式字符串。
        generator 执行完后检测是否有异常（OOM 等），若有则 yield error chunk。
        """
        full_text = ""
        phase = "thinking" if self._open_thinking else "content"

        # 游标
        content_sent_len = 0        # full_text[:content_sent_len] 已发送
        search_start = 0            # 下一次搜索 <tool_call> 的起点
        tool_call_index = 0
        thinking_start = 0          # thinking 内容在 full_text 中的起始位置
        gen_error: Optional[Exception] = None

        try:
            # asyncio.to_thread 使每次 next() 不阻塞事件循环
            while True:
                try:
                    token = await asyncio.to_thread(next, token_gen, None)
                except StopIteration:
                    break
                except Exception as exc:
                    gen_error = exc
                    break

                if token is None:
                    break

                full_text += token

                # ── thinking 阶段 ─────────────────────────────────────
                if phase == "thinking":
                    end_tag = None
                    for tag in ("</thinking>", "</think>"):
                        if tag in full_text:
                            end_tag = tag
                            break

                    if end_tag is None:
                        continue  # 还在 thinking，等待更多 token

                    end_pos = full_text.index(end_tag)
                    raw = full_text[thinking_start:end_pos]
                    reasoning = re.sub(r"</?think(?:ing)?>", "", raw).strip()
                    if reasoning:
                        yield _delta("reasoning_content", reasoning)

                    content_sent_len = end_pos + len(end_tag)
                    search_start = content_sent_len
                    phase = "content"
                    # 不 continue：当前 token 可能带来 content，继续处理

                # ── content 阶段 ──────────────────────────────────────
                if phase == "content":
                    tc_pos = full_text.find(_TC_OPEN, search_start)

                    if tc_pos == -1:
                        # 保留末尾不安全缓冲，防止标签被截断
                        safe_end = max(content_sent_len, len(full_text) - (_TC_OPEN_LEN - 1))
                        if safe_end > content_sent_len:
                            yield _delta("content", full_text[content_sent_len:safe_end])
                            content_sent_len = safe_end
                        search_start = content_sent_len
                        continue

                    # 找到 <tool_call>：先发送之前未发送的 content
                    if tc_pos > content_sent_len:
                        yield _delta("content", full_text[content_sent_len:tc_pos])
                    content_sent_len = tc_pos
                    search_start = tc_pos
                    phase = "tool_call"
                    # 不 continue：继续处理 tool_call 阶段

                # ── tool_call 阶段 ────────────────────────────────────
                if phase == "tool_call":
                    tc_slice = full_text[search_start:]

                    if _TC_CLOSE not in tc_slice:
                        continue  # 还没收到完整 tool_call

                    tc_end_rel = tc_slice.index(_TC_CLOSE) + len(_TC_CLOSE)
                    single_raw = tc_slice[:tc_end_rel]

                    async for chunk in self._emit_tool_call(single_raw, tool_call_index):
                        yield chunk
                    tool_call_index += 1

                    abs_end = search_start + tc_end_rel
                    content_sent_len = abs_end
                    search_start = abs_end

                    # 检查是否紧跟下一个 <tool_call>
                    after = full_text[abs_end:]
                    if _TC_OPEN in after:
                        search_start = abs_end + after.index(_TC_OPEN)
                        # phase 保持 tool_call
                    else:
                        phase = "content"
                        search_start = abs_end
                    continue

        except Exception as exc:
            gen_error = exc

        # ── 错误处理 ──────────────────────────────────────────────────
        if gen_error is not None:
            logger.error("Stream generation error: %s", gen_error, exc_info=True)
            yield _sse({"error": {"message": str(gen_error), "type": "internal_error"}})
            yield _finish("error")
            yield "data: [DONE]\n\n"
            return

        # ── 冲刷剩余 content ──────────────────────────────────────────
        if phase == "content" and content_sent_len < len(full_text):
            remaining = full_text[content_sent_len:]
            if remaining.strip():
                yield _delta("content", remaining)

        finish_reason = "tool_calls" if tool_call_index > 0 else "stop"
        yield _finish(finish_reason)
        yield "data: [DONE]\n\n"

    @staticmethod
    async def _emit_tool_call(raw: str, index: int) -> AsyncGenerator[str, None]:
        """将单个完整 <tool_call>...</tool_call> 块转换为流式 SSE chunks。"""
        m = re.search(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL)
        if not m:
            logger.warning("_emit_tool_call: no tool_call tag found in: %s", raw[:100])
            return

        try:
            call = json.loads(m.group(1).strip())
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse tool_call JSON: %s | raw=%s", exc, m.group(1)[:200])
            return

        call_id = generate_tool_call_id()
        func_name = call.get("name", "")
        args_str = json.dumps(call.get("arguments", {}), ensure_ascii=False)

        # Chunk 1：id + name + 空 arguments
        yield _sse({
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": func_name, "arguments": ""},
                    }]
                },
                "finish_reason": None,
            }]
        })

        # Chunk 2~N：arguments 分块发送
        chunk_size = 16
        for start in range(0, len(args_str), chunk_size):
            yield _sse({
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": index,
                            "function": {"arguments": args_str[start:start + chunk_size]},
                        }]
                    },
                    "finish_reason": None,
                }]
            })

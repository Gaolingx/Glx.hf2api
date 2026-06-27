"""
模型输出解析：thinking 块提取、tool_call 解析。
无状态纯函数，可在任意线程调用。
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<(thinking|think)>(.*?)</(thinking|think)>", re.DOTALL)
_TC_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def parse_response(
    text: str,
) -> Tuple[str, Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    从模型输出中提取：
      - reasoning_content：<thinking>/<think> 块合并
      - tool_calls：<tool_call> JSON 块列表
      - content：去除上述标签后的纯文本

    Returns:
        (content, reasoning_content, tool_calls)
        tool_calls 为 None 表示无工具调用（非空列表）
    """
    # ── 提取 thinking ──────────────────────────────────────────────────
    reasoning_parts: List[str] = []

    def _collect(m: re.Match) -> str:
        reasoning_parts.append(m.group(2).strip())
        return ""

    text = _THINK_RE.sub(_collect, text)
    reasoning_content = "\n".join(reasoning_parts) if reasoning_parts else None

    # ── 提取 tool_calls ────────────────────────────────────────────────
    tool_calls: List[Dict[str, Any]] = []
    for raw in _TC_RE.findall(text):
        try:
            call = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed tool_call JSON: %s | raw=%s", exc, raw[:200])
            continue

        tool_calls.append({
            "id": generate_tool_call_id(),
            "type": "function",
            "function": {
                "name": call.get("name", ""),
                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
            },
        })

    if tool_calls:
        text = _TC_RE.sub("", text)

    return text.strip(), reasoning_content, tool_calls or None

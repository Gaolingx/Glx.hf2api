"""
API 层数据模型（Pydantic）
职责：HTTP 请求/响应的校验与序列化，不含任何推理逻辑。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ..inference.types import GenerationConfig

logger = logging.getLogger(__name__)


# ── 工具调用模型 ────────────────────────────────────────────────────────

class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string，与 OpenAI 规范一致


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# ── 消息模型 ────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None

    @model_validator(mode="after")
    def _validate_role(self):
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool message must have tool_call_id")
            if not self.content:
                raise ValueError("tool message must have content")
        return self


# ── 请求模型 ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[Message]

    # 生成参数：直接字段（优先级最高）
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0)
    num_beams: Optional[int] = Field(default=None, ge=1)
    do_sample: Optional[bool] = None
    no_repeat_ngram_size: Optional[int] = Field(default=None, ge=0)

    stream: bool = False
    tools: List[Any] = Field(default_factory=list)
    open_thinking: bool = False
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_tool_refs(self):
        """校验 tool message 的 tool_call_id 必须有对应的 assistant tool_call。"""
        pending: Dict[str, int] = {}
        consumed: set = set()

        for i, msg in enumerate(self.messages):
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id in consumed:
                        raise ValueError(
                            f"messages[{i}]: tool_call_id '{tc.id}' already consumed."
                        )
                    pending[tc.id] = i

            elif msg.role == "tool":
                tid = msg.tool_call_id
                if tid in consumed:
                    raise ValueError(
                        f"messages[{i}]: tool_call_id '{tid}' already consumed."
                    )
                if tid not in pending:
                    raise ValueError(
                        f"messages[{i}]: tool_call_id '{tid}' has no matching assistant tool_call. "
                        f"Pending: {set(pending.keys())}"
                    )
                consumed.add(tid)
                del pending[tid]

        if pending:
            logger.debug("Unresolved tool_calls at end of messages: %s", set(pending.keys()))

        return self

    def get_open_thinking(self) -> bool:
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return (
                self.chat_template_kwargs.get("open_thinking", False)
                or self.chat_template_kwargs.get("enable_thinking", False)
            )
        return False

    def build_generation_config(self, server_defaults: Dict[str, Any]) -> GenerationConfig:
        """
        合并优先级：server_defaults（最低） < 请求直接字段（最高）。
        只有请求中显式设置的字段才会覆盖默认值。
        """
        cfg = GenerationConfig(**{
            k: v for k, v in server_defaults.items()
            if hasattr(GenerationConfig, k)
        })

        overrides = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }
        for key, val in overrides.items():
            if val is not None:
                setattr(cfg, key, val)

        return cfg

    def normalized_messages(self) -> List[Dict[str, Any]]:
        """
        转换为 apply_chat_template 可接受的 dict 列表。
        tool_calls[].function.arguments 统一为合法 JSON str。
        """
        result = []
        for msg in self.messages:
            d = msg.model_dump(exclude_none=True)
            if d.get("tool_calls"):
                for tc in d["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, dict):
                        func["arguments"] = json.dumps(args, ensure_ascii=False)
                    elif isinstance(args, str):
                        try:
                            json.loads(args)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON in tool_call arguments; replacing with {}")
                            func["arguments"] = "{}"
            result.append(d)
        return result

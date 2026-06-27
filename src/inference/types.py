"""
内部推理数据类型
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union


@dataclass
class GenerationConfig:
    """
    单次请求的生成参数，从 ChatRequest 中提取后传入引擎。
    所有字段均有默认值，engine 可直接使用。
    """
    temperature: float = 0.7
    top_p: float = 0.92
    top_k: int = 50
    max_tokens: int = 8192
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    no_repeat_ngram_size: int = 0

    def to_hf_kwargs(self) -> Dict[str, Any]:
        """转换为 model.generate 可接受的 kwargs"""
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }

        if self.num_beams > 1:
            # beam search 模式：do_sample 必须为 False
            kwargs["num_beams"] = self.num_beams
            kwargs["do_sample"] = False
            kwargs["early_stopping"] = True
        else:
            kwargs["num_beams"] = 1
            kwargs["do_sample"] = self.do_sample
            if self.do_sample:
                kwargs["temperature"] = self.temperature
                kwargs["top_p"] = self.top_p
                kwargs["top_k"] = self.top_k

        return kwargs


# 非流式结果
@dataclass
class GenerationResult:
    content: str
    reasoning_content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]


# 流式结果：engine 返回同步 generator，由 StreamHandler 消费
StreamGenerator = Generator[str, None, None]


@dataclass
class InferenceRequest:
    """
    请求队列中的单个推理请求。
    future 由调用方在事件循环中创建，set_result/set_exception
    必须通过 loop.call_soon_threadsafe 从工作线程回调。
    """
    request_id: str
    messages: List[Dict[str, Any]]
    gen_config: GenerationConfig
    tools: Optional[List[Any]]
    open_thinking: bool
    stream: bool

    # 绑定到事件循环的 Future（调用方负责创建）
    future: asyncio.Future = field(default_factory=asyncio.Future)
    # 用于 call_soon_threadsafe 的事件循环引用
    loop: Optional[asyncio.AbstractEventLoop] = field(default=None)

    created_at: float = field(default_factory=time.time)

    @staticmethod
    def make_id() -> str:
        return f"req_{uuid.uuid4().hex}"

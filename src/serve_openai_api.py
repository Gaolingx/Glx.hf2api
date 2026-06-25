"""
LLMs API Server - Supports streaming, concurrency, KV cache, multi-device, and optimizations
"""

import argparse
import json
import re
import os
import sys
import yaml
import time
import uuid
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# Global Variables & Configuration
# ============================================================================

model = None
tokenizer = None
config = None
logger = None
inference_executor = None  # ThreadPoolExecutor for model inference
batch_processor = None
memory_monitor = None


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(title: str = "default", log_level: str = "INFO"):
    """Setup structured logging with configurable level"""
    global logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(title)
    logger.setLevel(numeric_level)

    # Suppress transformers warnings unless debug mode
    if numeric_level > logging.DEBUG:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)

    return logger


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ModelDeviceConfig:
    """Device configuration for model loading and inference"""
    device: str = "cuda"
    dtype: str = "auto"
    device_map: Optional[str] = None
    gpus: Optional[List[int]] = None


@dataclass
class InferenceRequest:
    """Internal request object for batching"""
    request_id: str
    messages: List[dict]
    gen_params: Dict[str, Any]
    tools: Optional[List] = None
    open_thinking: bool = False
    stream: bool = False
    future: asyncio.Future = field(default_factory=lambda: asyncio.Future())
    created_at: float = field(default_factory=time.time)


class GenerationParams(BaseModel):
    """Generation parameters"""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.92, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    num_beams: int = Field(default=1, ge=1)
    do_sample: bool = True
    early_stopping: bool = False
    no_repeat_ngram_size: int = Field(default=0, ge=0)


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None

    @model_validator(mode='after')
    def validate_role_constraints(self):
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool message must have tool_call_id")
            if not self.content:
                raise ValueError("tool message must have content")
        if self.role == "assistant":
            if self.tool_calls and self.content and self.content.strip():
                # assistant 同时有 tool_calls 和 content 是允许的，但记录警告
                pass
        return self


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    num_beams: Optional[int] = None
    do_sample: Optional[bool] = None
    stream: bool = False
    tools: list = []
    open_thinking: bool = False
    chat_template_kwargs: Optional[dict] = None
    generation_params: Optional[GenerationParams] = None

    def get_open_thinking(self) -> bool:
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return (
                    self.chat_template_kwargs.get('open_thinking', False) or
                    self.chat_template_kwargs.get('enable_thinking', False)
            )
        return False

    @model_validator(mode='after')
    def validate_tool_call_references(self):
        """
        校验 tool message 的 tool_call_id 必须有对应且尚未被消费的 assistant tool_call。

        多轮对话中同一 id 不应跨轮复用：
          Round1: assistant(call_A) -> tool(call_A)   # call_A 已消费
          Round2: assistant(call_B) -> tool(call_B)   # call_B 已消费
          ❌ tool(call_A) 出现第二次 -> 报错
        """
        # pending: 已发出但尚未被 tool 消费的 call_id
        # consumed: 已被 tool 消费的 call_id（不允许重复消费）
        pending: Dict[str, int] = {}   # call_id -> messages index（便于错误定位）
        consumed: set = set()

        for i, msg in enumerate(self.messages):
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id in consumed:
                        raise ValueError(
                            f"messages[{i}]: tool_call_id '{tc.id}' was already consumed "
                            f"by a previous tool message. IDs must be unique across turns."
                        )
                    pending[tc.id] = i

            elif msg.role == "tool":
                tid = msg.tool_call_id
                if tid in consumed:
                    raise ValueError(
                        f"messages[{i}]: tool_call_id '{tid}' has already been consumed "
                        f"by a previous tool message."
                    )
                if tid not in pending:
                    raise ValueError(
                        f"messages[{i}]: tool_call_id '{tid}' has no matching "
                        f"assistant tool_call. Pending IDs: {set(pending.keys())}"
                    )
                consumed.add(tid)
                del pending[tid]

        # 流程结束后 pending 非空说明有未被响应的 tool_call（允许，最后一轮可以尚未执行）
        # 不报错，只记录 debug 信息
        if pending:
            logger.debug(f"Unresolved tool_calls at end of messages: {set(pending.keys())}")

        return self

    def get_generation_params(self) -> Dict[str, Any]:
        gen_config = config.get('generation', {})
        params = {
            'temperature': gen_config.get('temperature', 0.7),
            'top_p': gen_config.get('top_p', 0.92),
            'top_k': gen_config.get('top_k', 50),
            'max_tokens': gen_config.get('max_tokens', 8192),
            'repetition_penalty': gen_config.get('repetition_penalty', 1.0),
            'num_beams': gen_config.get('num_beams', 1),
            'do_sample': gen_config.get('do_sample', True),
            'early_stopping': gen_config.get('early_stopping', False),
            'no_repeat_ngram_size': gen_config.get('no_repeat_ngram_size', 0),
        }

        if self.generation_params:
            gen_params_dict = self.generation_params.dict(exclude_unset=True)
            params.update(gen_params_dict)

        direct_params = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'max_tokens': self.max_tokens,
            'repetition_penalty': self.repetition_penalty,
            'num_beams': self.num_beams,
            'do_sample': self.do_sample,
        }
        for key, value in direct_params.items():
            if value is not None:
                params[key] = value

        return params


# ============================================================================
# Memory Monitor
# ============================================================================

class MemoryMonitor:
    """Monitor GPU memory and manage KV cache cleanup"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('memory', {})
        self.max_memory_ratio = self.config.get('max_gpu_memory_ratio', 0.8)
        self.kv_cache_timeout = self.config.get('kv_cache_timeout_minutes', 30)
        self.cleanup_interval = self.config.get('cleanup_interval', 60)
        self.last_cleanup = time.time()
        self.request_timestamps = {}  # request_id -> last_access_time

        logger.info(
            f"MemoryMonitor initialized: max_ratio={self.max_memory_ratio}, "
            f"kv_timeout={self.kv_cache_timeout}min"
        )

    def check_memory_available(self) -> bool:
        """Check if GPU memory is available"""
        if not torch.cuda.is_available():
            return True

        try:
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            memory_ratio = allocated_memory / total_memory

            available = memory_ratio < self.max_memory_ratio

            if not available:
                logger.warning(
                    f"GPU memory threshold exceeded: {memory_ratio:.2%} "
                    f"(max: {self.max_memory_ratio:.2%})"
                )

            return available
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return True  # Fail open

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return {}

        try:
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(device) / 1e9
            cached = torch.cuda.memory_reserved(device) / 1e9

            return {
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
                "ratio": round(allocated / total, 3)
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

    def cleanup_cache(self, force: bool = False):
        """Clean up KV cache and GPU memory"""
        current_time = time.time()

        # Check if cleanup is needed
        if not force and (current_time - self.last_cleanup) < self.cleanup_interval:
            return

        logger.info("Starting cache cleanup...")

        # Remove old request timestamps
        timeout_seconds = self.kv_cache_timeout * 60
        expired_requests = [
            req_id for req_id, timestamp in self.request_timestamps.items()
            if (current_time - timestamp) > timeout_seconds
        ]

        for req_id in expired_requests:
            del self.request_timestamps[req_id]

        if expired_requests:
            logger.info(f"Cleaned up {len(expired_requests)} expired request records")

        # Clear PyTorch cache
        if torch.cuda.is_available():
            before_stats = self.get_memory_stats()
            torch.cuda.empty_cache()
            after_stats = self.get_memory_stats()

            logger.info(
                f"GPU cache cleared: {before_stats.get('cached_gb', 0):.2f}GB -> "
                f"{after_stats.get('cached_gb', 0):.2f}GB"
            )

        self.last_cleanup = current_time

    def track_request(self, request_id: str):
        """Track request access time"""
        self.request_timestamps[request_id] = time.time()


# ============================================================================
# Batch Processor
# ============================================================================

class BatchProcessor:
    """Process inference requests in batches"""

    def __init__(self, config: Dict[str, Any], executor: ThreadPoolExecutor):
        self.config = config.get('batching', {})
        self.enabled = self.config.get('enabled', True)
        self.max_batch_size = self.config.get('max_batch_size', 8)
        self.batch_timeout = self.config.get('batch_timeout_ms', 50) / 1000.0

        self.executor = executor
        self.queue = asyncio.Queue(maxsize=config.get('limits', {}).get('max_queue_size', 100))
        self.running = False

        logger.info(
            f"BatchProcessor initialized: enabled={self.enabled}, "
            f"max_batch_size={self.max_batch_size}, timeout={self.batch_timeout}s"
        )

    async def add_request(self, request: InferenceRequest) -> Any:
        """Add request to queue and wait for result"""
        try:
            await asyncio.wait_for(
                self.queue.put(request),
                timeout=5.0
            )

            # Wait for result
            timeout = config.get('server', {}).get('limits', {}).get('request_timeout', 300)
            result = await asyncio.wait_for(request.future, timeout=timeout)
            return result

        except asyncio.TimeoutError:
            logger.error(f"Request {request.request_id} timed out")
            raise HTTPException(status_code=504, detail="Request timeout")
        except asyncio.QueueFull:
            logger.error(f"Request queue full, rejecting request {request.request_id}")
            raise HTTPException(status_code=503, detail="Server overloaded")

    async def process_loop(self):
        """Main processing loop"""
        self.running = True
        logger.info("Batch processing loop started")

        while self.running:
            try:
                batch = await self._collect_batch()

                if not batch:
                    await asyncio.sleep(0.01)
                    continue

                # Process batch in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self._process_batch,
                    batch
                )

            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests into a batch"""
        batch = []
        deadline = time.time() + self.batch_timeout

        # Get first request (blocking with timeout)
        try:
            first_request = await asyncio.wait_for(
                self.queue.get(),
                timeout=self.batch_timeout
            )
            batch.append(first_request)
        except asyncio.TimeoutError:
            return batch

        # Collect additional requests until batch full or timeout
        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                request = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=max(0.001, deadline - time.time())
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        return batch

    def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests (runs in thread pool)"""
        try:
            logger.debug(f"Processing batch of {len(batch)} requests")

            # Separate streaming and non-streaming requests
            stream_requests = [r for r in batch if r.stream]
            non_stream_requests = [r for r in batch if not r.stream]

            # Process non-streaming in batch
            if non_stream_requests:
                self._process_non_stream_batch(non_stream_requests)

            # Process streaming individually (batching streaming is complex)
            for request in stream_requests:
                self._process_stream_request(request)

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

    def _process_non_stream_batch(self, batch: List[InferenceRequest]):
        """Process non-streaming requests in batch"""
        if len(batch) == 1:
            # Single request - process directly
            request = batch[0]
            try:
                result = self._generate_single(request)
                request.future.set_result(result)
            except Exception as e:
                request.future.set_exception(e)
            return

        # Batch processing
        try:
            # Prepare batch inputs
            prompts = []
            for req in batch:
                prompt = tokenizer.apply_chat_template(
                    req.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=req.tools or None,
                    open_thinking=req.open_thinking
                )
                prompts.append(prompt)

            # Tokenize batch
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            # Generate
            gen_params = batch[0].gen_params  # Use first request's params
            max_tokens = gen_params.get('max_tokens', 8192)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=gen_params.get('do_sample', True),
                    temperature=gen_params.get('temperature', 0.7),
                    top_p=gen_params.get('top_p', 0.92),
                    top_k=gen_params.get('top_k', 50),
                    repetition_penalty=gen_params.get('repetition_penalty', 1.0),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode outputs
            for i, request in enumerate(batch):
                try:
                    output_ids = outputs[i][inputs.input_ids[i].shape[0]:]
                    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

                    content, reasoning_content, tool_calls = parse_response(generated_text)

                    result = {
                        "content": content,
                        "reasoning_content": reasoning_content,
                        "tool_calls": tool_calls
                    }

                    request.future.set_result(result)
                    memory_monitor.track_request(request.request_id)

                except Exception as e:
                    logger.error(f"Error processing request {request.request_id}: {e}")
                    request.future.set_exception(e)

        except Exception as e:
            logger.error(f"Error in batch generation: {e}", exc_info=True)
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

    def _process_stream_request(self, request: InferenceRequest):
        """Process streaming request individually"""
        try:
            # For streaming, return a generator
            generator = self._generate_stream(request)
            request.future.set_result(generator)
        except Exception as e:
            request.future.set_exception(e)

    def _generate_single(self, request: InferenceRequest) -> Dict[str, Any]:
        """Generate response for single request"""
        prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=request.tools or None,
            open_thinking=request.open_thinking
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        max_tokens = request.gen_params.get('max_tokens', 8192)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=request.gen_params.get('do_sample', True),
                temperature=request.gen_params.get('temperature', 0.7),
                top_p=request.gen_params.get('top_p', 0.92),
                top_k=request.gen_params.get('top_k', 50),
                repetition_penalty=request.gen_params.get('repetition_penalty', 1.0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        output_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        content, reasoning_content, tool_calls = parse_response(generated_text)

        memory_monitor.track_request(request.request_id)

        return {
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls
        }

    def _generate_stream(self, request: InferenceRequest):
        """Generate streaming response (generator function)"""
        prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=request.tools or None,
            open_thinking=request.open_thinking
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        max_tokens = request.gen_params.get('max_tokens', 8192)

        # Use model.generate with streamer
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'max_new_tokens': max_tokens,
            'do_sample': request.gen_params.get('do_sample', True),
            'temperature': request.gen_params.get('temperature', 0.7),
            'top_p': request.gen_params.get('top_p', 0.92),
            'top_k': request.gen_params.get('top_k', 50),
            'repetition_penalty': request.gen_params.get('repetition_penalty', 1.0),
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'streamer': streamer,
        }

        # Start generation in separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens
        full_text = ""
        for text in streamer:
            full_text += text
            yield text

        thread.join()
        memory_monitor.track_request(request.request_id)

    async def stop(self):
        """Stop processing loop"""
        self.running = False
        logger.info("Batch processor stopped")


# ============================================================================
# Helper Functions
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def parse_response(text: str):
    """Parse model response to extract reasoning content and tool calls"""
    # 提取所有 thinking 块（支持多个，合并为一个 reasoning_content）
    THINK_PATTERN = re.compile(
        r'<(thinking|think)>(.*?)</(thinking|think)>',
        re.DOTALL
    )
    reasoning_parts = []
    def _collect_thinking(m: re.Match) -> str:
        reasoning_parts.append(m.group(2).strip())
        return ''   # 替换为空字符串，从 text 中移除

    text = THINK_PATTERN.sub(_collect_thinking, text)
    reasoning_content = '\n'.join(reasoning_parts) if reasoning_parts else None

    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({
                "id": generate_tool_call_id(),
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)
                }
            })
        except Exception as e:
            logger.warning(f"Failed to parse tool call: {e}")

    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)

    return text.strip(), reasoning_content, tool_calls or None


def init_model(
    model_config: Dict[str, Any],
    device_config: Union[Dict[str, Any], ModelDeviceConfig]
):
    """Initialize model and tokenizer based on configuration"""
    global model, tokenizer

    if isinstance(device_config, ModelDeviceConfig):
        device_config = {
            'device': device_config.device,
            'device_map': device_config.device_map,
            'dtype': device_config.dtype,
            'gpus': device_config.gpus
        }

    logger.info(f"Loading model from {model_config['name_or_path']}")

    # Load tokenizer
    tokenizer_kwargs = model_config.get('tokenizer', {})
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name_or_path'],
        trust_remote_code=model_config.get('trust_remote_code', True),
        **tokenizer_kwargs
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model loading kwargs
    model_kwargs = model_config.get('model_kwargs', {})
    model_kwargs['trust_remote_code'] = model_config.get('trust_remote_code', True)
    model_kwargs['revision'] = model_config.get('model_revision', "main")

    # Set dtype
    torch_dtype_str = device_config.get('dtype') or model_config.get('torch_dtype', 'auto')
    if torch_dtype_str != 'auto':
        model_kwargs['torch_dtype'] = getattr(torch, torch_dtype_str)

    # Device mapping
    if device_config.get('device_map'):
        model_kwargs['device_map'] = device_config['device_map']

    # Multi-GPU support
    if device_config.get('gpus') and not model_kwargs.get('device_map'):
        gpu_ids = ','.join(map(str, device_config['gpus']))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        model_kwargs['device_map'] = 'auto'

    # Quantization
    quant_config = model_config.get('quantization', {})
    load_in_8bit = quant_config.get('load_in_8bit', False)
    load_in_4bit = quant_config.get('load_in_4bit', False)

    is_quantized = load_in_8bit or load_in_4bit

    if is_quantized:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=quant_config.get('load_in_8bit', False),
            llm_int8_threshold=quant_config.get('llm_int8_threshold', 6.0),
            llm_int8_skip_modules=quant_config.get('llm_int8_skip_modules', None),
            llm_int8_enable_fp32_cpu_offload=quant_config.get('llm_int8_enable_fp32_cpu_offload', False),
            llm_int8_has_fp16_weight=quant_config.get('llm_int8_has_fp16_weight', False),
            load_in_4bit=quant_config.get('load_in_4bit', False),
            bnb_4bit_compute_dtype=getattr(
                torch, quant_config.get('bnb_4bit_compute_dtype', 'bfloat16')
            ),
            bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
            bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4')
        )

    if 'use_cache' in model_config:
        model_kwargs['use_cache'] = model_config['use_cache']

    if model_config.get('attn_implementation'):
        model_kwargs['attn_implementation'] = model_config['attn_implementation']

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name_or_path'],
        **model_kwargs
    )

    # Apply optimizations
    if model_config.get('apply_liger_kernel', False):
        try:
            from transformers.integrations.liger import apply_liger_kernel
            kernel_config = model_config.get('kernel_config', {})
            apply_liger_kernel(model, kernel_config)
            logger.info("Applied Liger Kernel optimizations")
        except ImportError:
            logger.warning("liger_kernel not installed, skipping optimization")

    if not device_config.get('device_map') and not is_quantized:
        device = device_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded successfully. Total parameters: {total_params / 1e6:.2f}M")

    return model, tokenizer


# ============================================================================
# Background Tasks
# ============================================================================

async def memory_cleanup_task():
    """Background task for periodic memory cleanup"""
    logger.info("Memory cleanup task started")

    while True:
        try:
            await asyncio.sleep(memory_monitor.cleanup_interval)
            memory_monitor.cleanup_cache()

            stats = memory_monitor.get_memory_stats()
            if stats:
                logger.debug(f"Memory stats: {stats}")

        except Exception as e:
            logger.error(f"Error in memory cleanup task: {e}", exc_info=True)


# ============================================================================
# FastAPI Application
# ============================================================================

def normalize_messages_for_template(messages: List[dict]) -> List[dict]:
    """
    将 Pydantic model_dump 产出的 messages 规范化，
    确保 tool_calls[].function.arguments 始终为 str（JSON 字符串），
    同时兼容部分模型 chat_template 期望 dict 的情况由 tokenizer 自行处理。
    此函数统一输出 str，与 OpenAI 规范一致。
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)  # shallow copy，避免修改原始数据
        if msg.get("tool_calls"):
            new_tool_calls = []
            for tc in msg["tool_calls"]:
                tc = dict(tc)
                func = dict(tc.get("function", {}))
                args = func.get("arguments", {})
                if isinstance(args, dict):
                    # dict -> JSON str
                    func["arguments"] = json.dumps(args, ensure_ascii=False)
                elif isinstance(args, str):
                    # 已经是 str，验证是否合法 JSON
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"tool_call arguments is invalid JSON str: {args[:100]}, "
                            f"wrapping as empty object"
                        )
                        func["arguments"] = "{}"
                else:
                    func["arguments"] = json.dumps(args, ensure_ascii=False)
                tc["function"] = func
                new_tool_calls.append(tc)
            msg["tool_calls"] = new_tool_calls
        normalized.append(msg)
    return normalized


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global inference_executor, batch_processor, memory_monitor

    # Startup
    logger.info("Starting up server...")

    # Initialize thread pool for model inference
    inference_executor = ThreadPoolExecutor(max_workers=1)  # Single worker for thread safety

    # Initialize memory monitor
    memory_monitor = MemoryMonitor(config)

    # Initialize batch processor
    batch_processor = BatchProcessor(config.get('server', {}), inference_executor)

    # Start background tasks
    batch_task = asyncio.create_task(batch_processor.process_loop())
    cleanup_task = asyncio.create_task(memory_cleanup_task())

    logger.info("Server startup complete")

    yield

    # Shutdown
    logger.info("Shutting down server...")

    await batch_processor.stop()
    batch_task.cancel()
    cleanup_task.cancel()

    inference_executor.shutdown(wait=True)

    logger.info("Server shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def memory_check_middleware(request: Request, call_next):
    """Check memory before processing request"""
    if request.url.path.startswith("/v1/chat/completions"):
        if not memory_monitor.check_memory_available():
            stats = memory_monitor.get_memory_stats()
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={"detail": f"GPU memory exhausted: {stats}"}
            )

    response = await call_next(request)
    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    try:
        request_id = f"req_{int(time.time() * 1000)}"
        gen_params = request.get_generation_params()

        # Create inference request
        inference_request = InferenceRequest(
            request_id=request_id,
            messages=normalize_messages_for_template(
                [msg.model_dump(exclude_none=True) for msg in request.messages]
            ),
            gen_params=gen_params,
            tools=request.tools,
            open_thinking=request.get_open_thinking(),
            stream=request.stream
        )

        # Add to batch processor
        result = await batch_processor.add_request(inference_request)

        # Handle streaming response
        if request.stream:
            def emit_tool_call_stream_chunks(raw_buffer: str, tc_index: int):
                """
                将完整的 <tool_call>...</tool_call> 字符串按 OpenAI 流式格式拆分。
                raw_buffer 必须包含且仅包含一个完整 tool_call 块。
                """
                m = re.search(r'<tool_call>(.*?)</tool_call>', raw_buffer, re.DOTALL)
                if not m:
                    logger.warning(f"emit_tool_call_stream_chunks: no tool_call found in buffer")
                    return
                try:
                    call = json.loads(m.group(1).strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool_call JSON in stream: {e}, raw={m.group(1)[:200]}")
                    return

                call_id = generate_tool_call_id()
                args_str = json.dumps(call.get("arguments", {}), ensure_ascii=False)
                func_name = call.get("name", "")

                # Chunk 1: id + name + 空 arguments
                yield (
                    f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': tc_index, 'id': call_id, 'type': 'function', 'function': {'name': func_name, 'arguments': ''}}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                )
                # Chunk 2~N: arguments 分块
                chunk_size = 16
                for start in range(0, len(args_str), chunk_size):
                    chunk = args_str[start: start + chunk_size]
                    yield (
                        f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': tc_index, 'function': {'arguments': chunk}}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                    )

            async def stream_generator():
                """Async generator for streaming response"""
                full_text = ""
                open_thinking = request.get_open_thinking()

                phase = "pending_think"
                think_open_tag = None
                thinking_start = 0
                MAX_THINK_WAIT = 50

                content_sent_len = 0
                tool_call_search_start = 0
                tool_call_index = 0
                TC_OPEN, TC_OPEN_LEN = '<tool_call>', len('<tool_call>')

                THINK_OPEN = ('<think>', '<thinking>')
                CLOSE = {'<think>': '</think>', '<thinking>': '</thinking>'}

                for token in result:
                    full_text += token

                    if phase == "pending_think":
                        hit = [t for t in THINK_OPEN if t in full_text]
                        if hit:
                            think_open_tag = min(hit, key=lambda t: full_text.find(t))
                            thinking_start = full_text.find(think_open_tag) + len(think_open_tag)
                            phase = "thinking"
                        elif len(full_text.strip()) > MAX_THINK_WAIT:
                            phase = "content"          # 确认无 think，正常进 content
                            content_sent_len = tool_call_search_start = 0
                        else:
                            continue

                    if phase == "thinking":
                        close = CLOSE[think_open_tag]
                        if close in full_text:
                            end = full_text.index(close)
                            reasoning = full_text[thinking_start:end].strip()
                            if reasoning and open_thinking:   # 是否下发由 open_thinking 决定
                                yield (
                                    f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': reasoning}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                                )
                            content_sent_len = tool_call_search_start = end + len(close)
                            phase = "content"
                        else:
                            continue

                    # ════════════════════════════════════════════════════════
                    # Phase: content
                    #   安全地发送 content，同时监听 <tool_call> 起始
                    # ════════════════════════════════════════════════════════
                    if phase == "content":
                        # 在未发送区域内搜索 <tool_call>
                        tc_pos = full_text.find(TC_OPEN, tool_call_search_start)

                        if tc_pos == -1:
                            # 没有 <tool_call>
                            # 保留末尾 TC_OPEN_LEN-1 字节作为"不安全缓冲"
                            # 防止 <tool_call> 被 token 边界切断后前半部分误发
                            safe_end = max(content_sent_len, len(full_text) - (TC_OPEN_LEN - 1))
                            if safe_end > content_sent_len:
                                to_send = full_text[content_sent_len: safe_end]
                                yield (
                                    f"data: {json.dumps({'choices': [{'delta': {'content': to_send}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                                )
                                content_sent_len = safe_end
                            # tool_call_search_start 推进到 safe_end，避免重复搜索已确认无 TC_OPEN 的区域
                            tool_call_search_start = content_sent_len
                            continue

                        # 找到 <tool_call>
                        # 先发送 tc_pos 之前未发送的 content
                        if tc_pos > content_sent_len:
                            to_send = full_text[content_sent_len: tc_pos]
                            yield (
                                f"data: {json.dumps({'choices': [{'delta': {'content': to_send}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                            )
                        content_sent_len = tc_pos          # 位置停在 <tool_call> 起始
                        tool_call_search_start = tc_pos
                        phase = "tool_call"
                        # ✅ 不 continue：让本次循环继续走 tool_call 分支
                        #    处理当前 token 带来的 tool_call 内容

                    # ════════════════════════════════════════════════════════
                    # Phase: tool_call
                    #   从 full_text[tool_call_search_start:] 切片，不独立维护 buf
                    #   等待 </tool_call> 到达后一次性处理
                    # ════════════════════════════════════════════════════════
                    if phase == "tool_call":
                        # 从 full_text 切片，永远不重复追加
                        tc_slice = full_text[tool_call_search_start:]

                        if '</tool_call>' not in tc_slice:
                            continue   # tool_call 尚未结束，继续等待

                        # 提取完整的单个 tool_call 块
                        tc_end_tag = '</tool_call>'
                        tc_end_pos = tc_slice.index(tc_end_tag) + len(tc_end_tag)
                        single_tc_raw = tc_slice[:tc_end_pos]

                        # 发送流式 chunks
                        for chunk in emit_tool_call_stream_chunks(single_tc_raw, tool_call_index):
                            yield chunk
                        tool_call_index += 1

                        # 推进 content_sent_len 和 search_start 到该 tool_call 结束位置
                        abs_tc_end = tool_call_search_start + tc_end_pos
                        content_sent_len = abs_tc_end
                        tool_call_search_start = abs_tc_end

                        # 检查紧跟其后是否还有 <tool_call>
                        after_tc = full_text[abs_tc_end:]
                        if TC_OPEN in after_tc:
                            # 定位到下一个 tool_call，保持 phase = "tool_call"
                            next_tc_rel = after_tc.index(TC_OPEN)
                            tool_call_search_start = abs_tc_end + next_tc_rel
                        else:
                            # 没有更多 tool_call，退回 content phase
                            phase = "content"
                            tool_call_search_start = abs_tc_end
                        continue

                # ── 流结束后：冲刷剩余 content ────────────────────────────────
                if phase == "content" and content_sent_len < len(full_text):
                    remaining = full_text[content_sent_len:]
                    if remaining.strip():
                        yield (
                            f"data: {json.dumps({'choices': [{'delta': {'content': remaining}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                        )

                has_tool_calls = tool_call_index > 0
                finish_reason = "tool_calls" if has_tool_calls else "stop"
                yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': finish_reason}]}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )

        # Non-streaming response
        else:
            message = {"role": "assistant", "content": result["content"]}
            if result["reasoning_content"]:
                message["reasoning_content"] = result["reasoning_content"]
            if result["tool_calls"]:
                message["tool_calls"] = result["tool_calls"]

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if result["tool_calls"] else "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": config.get('model', {}).get('name', 'default'),
                "object": "model",
                "created": int(time.time()),
                "owned_by": config.get('title', 'default')
            }
        ]
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    stats = memory_monitor.get_memory_stats() if memory_monitor else {}

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "memory": stats,
        "queue_size": batch_processor.queue.qsize() if batch_processor else 0
    }


@app.get("/metrics")
def metrics():
    """Prometheus-style metrics endpoint"""
    stats = memory_monitor.get_memory_stats() if memory_monitor else {}

    metrics_text = f"""# HELP gpu_memory_allocated_gb GPU memory allocated in GB
# TYPE gpu_memory_allocated_gb gauge
gpu_memory_allocated_gb {stats.get('allocated_gb', 0)}

# HELP gpu_memory_total_gb Total GPU memory in GB
# TYPE gpu_memory_total_gb gauge
gpu_memory_total_gb {stats.get('total_gb', 0)}

# HELP gpu_memory_ratio GPU memory utilization ratio
# TYPE gpu_memory_ratio gauge
gpu_memory_ratio {stats.get('ratio', 0)}

# HELP request_queue_size Current request queue size
# TYPE request_queue_size gauge
request_queue_size {batch_processor.queue.qsize() if batch_processor else 0}
"""

    return metrics_text


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLMs API Server - Production Ready"
    )

    parser.add_argument(
        '--config',
        default='config/production.yaml',
        type=str,
        help="Path to YAML configuration file"
    )

    parser.add_argument('--device', type=str, help="Override device")
    parser.add_argument('--dtype', type=str, help="Override dtype")
    parser.add_argument('--gpus', type=str, help="Comma-separated GPU IDs")
    parser.add_argument('--host', type=str, help="Override server host")
    parser.add_argument('--port', type=int, help="Override server port")
    parser.add_argument('--log-level', type=str, help="Override log level")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    log_level = args.log_level or config.get('server', {}).get('log_level', 'info')
    setup_logging(config.get('title', 'default'), log_level)

    # Override config with CLI args
    model_config = config.get('model', {})
    device_config = config.get('device', {})
    server_config = config.get('server', {})

    if args.device:
        device_config['device'] = args.device
    if args.dtype:
        device_config['dtype'] = args.dtype
    if args.gpus:
        device_config['gpus'] = [int(g) for g in args.gpus.split(',')]

    host = args.host or server_config.get('host', '0.0.0.0')
    port = args.port or server_config.get('port', 8998)

    # Initialize model
    logger.info("Initializing model...")
    model, tokenizer = init_model(model_config, device_config)

    # Start server
    uvicorn_config = {
        'app': app,
        'host': host,
        'port': port,
        'log_level': log_level.lower(),
        'timeout_keep_alive': server_config.get('timeout_keep_alive', 5),
    }

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(**uvicorn_config)

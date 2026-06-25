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
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
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

def setup_logging(log_level: str = "INFO"):
    """Setup structured logging with configurable level"""
    global logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger("minimind")
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


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: list
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


def parse_response(text: str):
    """Parse model response to extract reasoning content and tool calls"""
    reasoning_content = None
    think_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    elif ' thinking' in text and ' response' in text:
        parts = text.split(' response', 1)
        if len(parts) == 2:
            reasoning_content = parts[0].replace(' thinking', '').strip()
            text = parts[1].strip()

    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({
                "id": f"call_{int(time.time())}_{i}",
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
    quant_config = model_config.get('quantization', None)
    if quant_config:
        if quant_config.get('load_in_8bit', False):
            model_kwargs['load_in_8bit'] = True
        elif quant_config.get('load_in_4bit', False):
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, quant_config.get('bnb_4bit_compute_dtype', 'float16')
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

    is_quantized = (
            model_kwargs.get('load_in_8bit', False) or
            'quantization_config' in model_kwargs
    )

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
            return HTTPException(
                status_code=503,
                detail=f"GPU memory exhausted: {stats}"
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
            messages=request.messages,
            gen_params=gen_params,
            tools=request.tools,
            open_thinking=request.get_open_thinking(),
            stream=request.stream
        )

        # Add to batch processor
        result = await batch_processor.add_request(inference_request)

        # Handle streaming response
        if request.stream:
            async def stream_generator():
                """Async generator for streaming response"""
                full_text = ""
                thinking_ended = not request.get_open_thinking()

                for text in result:  # result is a generator
                    full_text += text

                    # Parse thinking vs response
                    if not thinking_ended:
                        if '</thinking>' in full_text or ' response' in full_text:
                            thinking_ended = True
                            parts = re.split(r'</thinking>| response', full_text, 1)
                            reasoning = parts[0].replace('<thinking>', '').strip()
                            if reasoning:
                                yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': reasoning}}]}, ensure_ascii=False)}\n\n"
                            if len(parts) > 1:
                                content = parts[1].strip()
                                if content:
                                    yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': text}}]}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]}, ensure_ascii=False)}\n\n"

                # Final message
                _, _, tool_calls = parse_response(full_text)
                if tool_calls:
                    yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': tool_calls}}]}, ensure_ascii=False)}\n\n"

                yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls' if tool_calls else 'stop'}]}, ensure_ascii=False)}\n\n"
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
                "owned_by": "minimind"
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
        description="MiniMind Server - Production Ready"
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
    setup_logging(log_level)

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

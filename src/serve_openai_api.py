"""
MiniMind Server with YAML-based Configuration
Supports streaming, sequential queueing, KV cache, multi-device, and advanced generation strategies
Refactored to be synchronous and thread-safe.
"""

import argparse
import json
import re
import os
import sys
import yaml
import time
import queue
import threading
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import torch
import warnings
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
    GenerationConfig
)

warnings.filterwarnings('ignore')

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()

# 全局模型、配置及线程锁
model = None
tokenizer = None
config = None
inference_lock = threading.Lock()  # 用于排队处理并发请求的互斥锁


@dataclass
class ModelDeviceConfig:
    """Device configuration for model loading and inference"""
    device: str = "cuda"
    dtype: str = "auto"
    device_map: Optional[str] = None
    gpus: Optional[List[int]] = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_model(
    model_config: Dict[str, Any],
    device_config: Union[Dict[str, Any], ModelDeviceConfig]
):
    """
    Initialize model and tokenizer based on configuration
    """
    global model, tokenizer
    
    if isinstance(device_config, ModelDeviceConfig):
        device_config = {
            'device': device_config.device,
            'device_map': device_config.device_map,
            'dtype': device_config.dtype,
            'gpus': device_config.gpus
        }
    
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
    
    # 1. 优化：在加载时直接指定 dtype
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
    
    # Quantization config
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
    
    if model_config.get('use_flash_attention_2', False):
        model_kwargs['attn_implementation'] = 'flash_attention_2'
    
    # Load model (加载时已应用 dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name_or_path'],
        **model_kwargs
    )
    
    # Apply Liger Kernel optimizations
    if model_config.get('apply_liger_kernel', False):
        try:
            from transformers.integrations.liger import apply_liger_kernel
            kernel_config = model_config.get('kernel_config', {})
            apply_liger_kernel(model, kernel_config)
            print("Applied Liger Kernel optimizations")
        except ImportError:
            print("Warning: liger_kernel not installed, skipping optimization")

    is_quantized = (
        model_kwargs.get('load_in_8bit', False) or 
        'quantization_config' in model_kwargs
    )
    
    if not device_config.get('device_map') and not is_quantized:
        device = device_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully. Total parameters: {total_params / 1e6:.2f}M")
    
    return model, tokenizer


class GenerationParams(BaseModel):
    """Generation parameters that can be overridden via OAI API"""
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
    stream: bool = True
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


class SyncQueueStreamer(TextStreamer):
    """用于同步生成线程与主响应线程交互的同步 Streamer"""
    def __init__(self, tokenizer, q: queue.Queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = q

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


class SimpleCollectorStreamer(TextStreamer):
    """用于非流式单次生成收集文本的同步 Streamer"""
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.tokens = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.tokens.append(text)


def parse_response(text: str):
    """Parse model response to extract reasoning content and tool calls"""
    reasoning_content = None
    think_match = re.search(r' thinking(.*?) response', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r' thinking.*? response\s*', '', text, flags=re.DOTALL)
    elif ' response' in text:
        parts = text.split(' response', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    
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
        except Exception:
            pass
    
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    
    return text.strip(), reasoning_content, tool_calls or None


def generate_stream_response(
    messages: List[dict],
    gen_params: Dict[str, Any],
    tools: Optional[List] = None,
    open_thinking: bool = False,
    eos_token_ids: Optional[List[int]] = None
):
    """
    同步流式生成器：
    使用全局锁 inference_lock 确保并发请求时，其他请求在进入此临界区前会进行排队。
    """
    with inference_lock:
        q = queue.Queue()
        streamer = SyncQueueStreamer(tokenizer, q)
        
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools or None,
            open_thinking=open_thinking
        )
        
        inputs = tokenizer(new_prompt, return_tensors="pt").to(model.device)
        max_tokens = gen_params.get('max_tokens', 8192)

        # 构建生成参数
        generate_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'pad_token_id': tokenizer.pad_token_id,
            'streamer': streamer,
            'max_new_tokens': max_tokens,
            'do_sample': gen_params.get('do_sample', True),
            'temperature': gen_params.get('temperature', 0.7),
            'top_p': gen_params.get('top_p', 0.92),
            'top_k': gen_params.get('top_k', 50),
            'repetition_penalty': gen_params.get('repetition_penalty', 1.0),
            'num_beams': gen_params.get('num_beams', 1),
            'no_repeat_ngram_size': gen_params.get('no_repeat_ngram_size', 0),
        }
        
        if eos_token_ids:
            generate_kwargs['eos_token_id'] = eos_token_ids
        else:
            generate_kwargs['eos_token_id'] = tokenizer.eos_token_id
            
        if gen_params.get('num_beams', 1) > 1:
            generate_kwargs['early_stopping'] = gen_params.get('early_stopping', False)
            generate_kwargs['do_sample'] = False
        
        gen_config_path = config.get('generation', {}).get('generation_config_path')
        if gen_config_path:
            external_config = GenerationConfig.from_pretrained(gen_config_path)
            for key, value in external_config.to_dict().items():
                if value is not None and key not in ['input_ids', 'attention_mask']:
                    generate_kwargs[key] = value

        # 在子线程启动推理，使主线程可以实时向客户端 yield 数据
        def _run_generation():
            try:
                model.generate(**generate_kwargs)
            except Exception as e:
                q.put(e)

        generation_thread = threading.Thread(target=_run_generation)
        generation_thread.start()

        full_text = ""
        emitted = 0
        thinking_ended = not bool(open_thinking)

        try:
            while True:
                text = q.get()
                if text is None:
                    break
                if isinstance(text, Exception):
                    raise text
                
                full_text += text

                # 实时流输出解析逻辑
                if not thinking_ended:
                    pos = full_text.find(' response')
                    if pos >= 0:
                        thinking_ended = True
                        new_r = full_text[emitted:pos]
                        if new_r:
                            yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': new_r}}]}, ensure_ascii=False)}\n\n"
                        emitted = pos + len(' response')
                        after = full_text[emitted:].lstrip('\n')
                        emitted = len(full_text) - len(after)
                        if after:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': after}}]}, ensure_ascii=False)}\n\n"
                            emitted = len(full_text)
                    else:
                        new_r = full_text[emitted:]
                        if new_r:
                            yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': new_r}}]}, ensure_ascii=False)}\n\n"
                            emitted = len(full_text)
                else:
                    new_c = full_text[emitted:]
                    if new_c:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': new_c}}]}, ensure_ascii=False)}\n\n"
                        emitted = len(full_text)
        finally:
            generation_thread.join()

        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': tool_calls}}]}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls' if tool_calls else 'stop'}]}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    """
    FastAPI 同步阻塞路由（使用 def 而不是 async def）。
    当高并发请求到达时，FastAPI 会在内置线程池中分配线程执行此函数。
    由于函数内部涉及互斥锁或生成器里的锁，请求会自动在此接口进行安全排队。
    """
    try:
        gen_params = request.get_generation_params()
        eos_token_ids = config.get('generation', {}).get('eos_token_id', None)
        
        if request.stream:
            return StreamingResponse(
                generate_stream_response(
                    messages=request.messages,
                    gen_params=gen_params,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking(),
                    eos_token_ids=eos_token_ids
                ),
                media_type="text/event-stream"
            )
        else:
            # 非流式生成：获取全局锁，同步生成并等待结束
            with inference_lock:
                new_prompt = tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=request.tools or None,
                    open_thinking=request.get_open_thinking()
                )
                inputs = tokenizer(new_prompt, return_tensors="pt").to(model.device)
                max_tokens = gen_params.get('max_tokens', 8192)

                streamer = SimpleCollectorStreamer(tokenizer)
                
                generate_kwargs = {
                    'input_ids': inputs.input_ids,
                    'attention_mask': inputs.attention_mask,
                    'pad_token_id': tokenizer.pad_token_id,
                    'streamer': streamer,
                    'max_new_tokens': max_tokens,
                    'do_sample': gen_params.get('do_sample', True),
                    'temperature': gen_params.get('temperature', 0.7),
                    'top_p': gen_params.get('top_p', 0.92),
                    'top_k': gen_params.get('top_k', 50),
                    'repetition_penalty': gen_params.get('repetition_penalty', 1.0),
                    'num_beams': gen_params.get('num_beams', 1),
                    'no_repeat_ngram_size': gen_params.get('no_repeat_ngram_size', 0),
                }
                
                if eos_token_ids:
                    generate_kwargs['eos_token_id'] = eos_token_ids
                else:
                    generate_kwargs['eos_token_id'] = tokenizer.eos_token_id

                if gen_params.get('num_beams', 1) > 1:
                    generate_kwargs['early_stopping'] = gen_params.get('early_stopping', False)
                    generate_kwargs['do_sample'] = False

                model.generate(**generate_kwargs)
                full_text = "".join(streamer.tokens)

            content, reasoning_content, tool_calls = parse_response(full_text)
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except Exception as e:
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
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MiniMind Server with YAML Configuration (Synchronous Mode)"
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        default='config/default.yaml',
        type=str,
        help="Path to YAML configuration file (default: config/default.yaml)"
    )
    
    # Device override arguments
    parser.add_argument(
        '--device',
        type=str,
        help="Override device for model inference (e.g., 'cuda', 'cpu')"
    )
    parser.add_argument(
        '--dtype',
        type=str,
        help="Override model dtype (e.g., 'float16', 'bfloat16', 'float32')"
    )
    parser.add_argument(
        '--gpus',
        type=str,
        help="Comma-separated GPU IDs to use (e.g., '0,1')"
    )
    
    # Server override arguments
    parser.add_argument(
        '--host',
        type=str,
        help="Override server host address"
    )
    parser.add_argument(
        '--port',
        type=int,
        help="Override server port"
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
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
    
    # 初始化加载模型
    model, tokenizer = init_model(model_config, device_config)
    
    uvicorn_config = {
        'app': app,
        'host': host,
        'port': port,
        'workers': server_config.get('workers', 1),
        'log_level': server_config.get('log_level', 'info'),
        'limit_concurrency': server_config.get('limit_concurrency', None),
        'timeout_keep_alive': server_config.get('timeout_keep_alive', 5),
    }
    
    print(f"Starting synchronous server on {host}:{port}")
    uvicorn.run(**uvicorn_config)

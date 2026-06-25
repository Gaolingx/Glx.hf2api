"""
MiniMind Server with YAML-based Configuration
Supports streaming, concurrency, KV cache, multi-device, and optimizations
"""

import argparse
import json
import re
import os
import sys
import yaml
import time
from typing import Optional, Dict, Any, List

import torch
import warnings
import uvicorn
from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig
)

warnings.filterwarnings('ignore')

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()

# Global variables for model and tokenizer
model = None
tokenizer = None
config = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_model(model_config: Dict[str, Any], device_config: Dict[str, Any]):
    """
    Initialize model and tokenizer based on configuration
    
    Args:
        model_config: Model loading configuration
        device_config: Device and optimization configuration
    """
    global model, tokenizer
    
    # Load tokenizer
    tokenizer_kwargs = model_config.get('tokenizer', {})
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name_or_path'],
        trust_remote_code=model_config.get('trust_remote_code', True),
        **tokenizer_kwargs
    )
    
    # Prepare model loading kwargs
    model_kwargs = model_config.get('model_kwargs', {})
    model_kwargs['trust_remote_code'] = model_config.get('trust_remote_code', True)
    
    # Device mapping
    if device_config.get('device_map'):
        model_kwargs['device_map'] = device_config['device_map']
    
    # Quantization config
    quant_config = model_config.get('quantization', None)
    if quant_config:
        if quant_config.get('load_in_8bit', False):
            model_kwargs['load_in_8bit'] = True
        elif quant_config.get('load_in_4bit', False):
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'float16')),
                bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
                bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4')
            )
    
    # KV cache settings
    if 'use_cache' in model_config:
        model_kwargs['use_cache'] = model_config['use_cache']
    
    # Attention implementation
    if model_config.get('attn_implementation'):
        model_kwargs['attn_implementation'] = model_config['attn_implementation']
    
    # Flash Attention 2
    if model_config.get('use_flash_attention_2', False):
        model_kwargs['attn_implementation'] = 'flash_attention_2'
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name_or_path'],
        **model_kwargs
    )
    
    # Apply dtype
    torch_dtype = model_config.get('torch_dtype', 'auto')
    if torch_dtype != 'auto':
        dtype = getattr(torch, torch_dtype)
        model = model.to(dtype)
    
    # Apply Liger Kernel optimizations
    if model_config.get('apply_liger_kernel', False):
        try:
            from transformers.integrations.liger import apply_liger_kernel
            kernel_config = model_config.get('kernel_config', {})
            apply_liger_kernel(model, kernel_config)
            print("Applied Liger Kernel optimizations")
        except ImportError:
            print("Warning: liger_kernel not installed, skipping optimization")
    
    # Move to device if not using device_map
    if not device_config.get('device_map') and not model_kwargs.get('load_in_8bit') and not model_kwargs.get('load_in_4bit'):
        device = device_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
    model.eval()
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded successfully")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    return model, tokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: Optional[int] = None
    stream: bool = True
    tools: list = []
    open_thinking: bool = False
    chat_template_kwargs: Optional[dict] = None

    def get_open_thinking(self) -> bool:
        """Support multiple ways to enable thinking mode"""
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False
    
    def get_max_tokens(self) -> int:
        """Get max tokens from request or config"""
        if self.max_tokens is not None:
            return self.max_tokens
        return config.get('generation', {}).get('max_tokens', 8192)


class CustomStreamer(TextStreamer):
    """Custom streamer that outputs to a queue for async streaming"""
    
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


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
    temperature: float,
    top_p: float,
    max_tokens: int,
    tools: Optional[List] = None,
    open_thinking: bool = False
):
    """
    Generate streaming response for chat completion
    
    Args:
        messages: Chat messages
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        tools: Available tools for function calling
        open_thinking: Whether to enable thinking mode
    """
    try:
        # Apply chat template
        gen_config = config.get('generation', {})
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools or None,
            open_thinking=open_thinking
        )[-max_tokens:]
        
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(model.device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            generate_kwargs = {
                'input_ids': inputs.input_ids,
                'max_new_tokens': max_tokens,
                'do_sample': True,
                'temperature': temperature,
                'top_p': top_p,
                'attention_mask': inputs.attention_mask,
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'streamer': streamer
            }
            
            # Add generation config overrides
            if gen_config.get('top_k'):
                generate_kwargs['top_k'] = gen_config['top_k']
            if gen_config.get('repetition_penalty'):
                generate_kwargs['repetition_penalty'] = gen_config['repetition_penalty']
            
            model.generate(**generate_kwargs)

        Thread(target=_generate).start()

        full_text = ""
        emitted = 0
        thinking_ended = not bool(open_thinking)

        while True:
            text = queue.get()
            if text is None:
                break
            full_text += text

            if not thinking_ended:
                pos = full_text.find(' response')
                if pos >= 0:
                    thinking_ended = True
                    new_r = full_text[emitted:pos]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                    emitted = pos + len(' response')
                    after = full_text[emitted:].lstrip('\n')
                    emitted = len(full_text) - len(after)
                    if after:
                        yield json.dumps({"choices": [{"delta": {"content": after}}]}, ensure_ascii=False)
                        emitted = len(full_text)
                else:
                    new_r = full_text[emitted:]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                        emitted = len(full_text)
            else:
                new_c = full_text[emitted:]
                if new_c:
                    yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)
                    emitted = len(full_text)

        # Parse and emit tool calls if present
        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        yield json.dumps({
            "choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]
        }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    try:
        max_tokens = request.get_max_tokens()
        
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )[-max_tokens:]
            
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(model.device)
            
            generate_kwargs = {
                'input_ids': inputs["input_ids"],
                'max_new_tokens': max_tokens,
                'do_sample': True,
                'attention_mask': inputs["attention_mask"],
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'top_p': request.top_p,
                'temperature': request.temperature
            }
            
            with torch.no_grad():
                generated_ids = model.generate(**generate_kwargs)
                answer = tokenizer.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            
            content, reasoning_content, tool_calls = parse_response(answer)
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
async def list_models():
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
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Server with YAML Configuration")
    parser.add_argument(
        '--config',
        default='config/default.yaml',
        type=str,
        help="Path to YAML configuration file (default: config/default.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract config sections
    model_config = config.get('model', {})
    device_config = config.get('device', {})
    server_config = config.get('server', {})
    
    # Initialize model
    model, tokenizer = init_model(model_config, device_config)
    
    # Configure uvicorn server
    uvicorn_config = {
        'app': app,
        'host': server_config.get('host', '0.0.0.0'),
        'port': server_config.get('port', 8998),
        'workers': server_config.get('workers', 1),
        'log_level': server_config.get('log_level', 'info'),
        'limit_concurrency': server_config.get('limit_concurrency', None),
        'timeout_keep_alive': server_config.get('timeout_keep_alive', 5),
    }
    
    # Start server
    print(f"Starting server on {uvicorn_config['host']}:{uvicorn_config['port']}")
    uvicorn.run(**uvicorn_config)

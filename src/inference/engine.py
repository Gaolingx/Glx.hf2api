"""
InferenceEngine：封装模型加载与推理，与 asyncio / FastAPI 完全解耦。
所有公开方法均为同步，在 ThreadPoolExecutor 中调用。
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from ..configs import DeviceConfig, ModelConfig
from .types import GenerationConfig, GenerationResult, StreamGenerator

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    负责：
      - 模型 / tokenizer 加载
      - 单请求非流式推理
      - 单请求流式推理（返回 generator）
    不负责：队列调度、批处理、asyncio 事件循环。
    """

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------ #
    # 加载                                                                 #
    # ------------------------------------------------------------------ #

    def load(self, model_cfg: ModelConfig, device_cfg: DeviceConfig) -> None:
        """加载 tokenizer 与模型，幂等（重复调用无副作用）。"""
        if self._loaded:
            logger.warning("InferenceEngine.load() called again; skipping.")
            return

        self._configure_cuda_precision(device_cfg)
        self.tokenizer = self._load_tokenizer(model_cfg)
        self.model = self._load_model(model_cfg, device_cfg)
        self._loaded = True

        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Engine ready. Parameters: {total / 1e6:.1f}M")

    def _load_tokenizer(self, model_cfg: ModelConfig):
        POST_LOAD_FIELDS = {"chat_template", "eos_token", "pad_token"}
        tok_cfg = model_cfg.tokenizer
        tok_kwargs = {
            k: v
            for k, v in asdict(tok_cfg).items()
            if k not in POST_LOAD_FIELDS and v is not None
        }

        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.name_or_path,
            trust_remote_code=model_cfg.trust_remote_code,
            **tok_kwargs,
        )

        if tok_cfg.chat_template:
            tokenizer.chat_template = tok_cfg.chat_template
        if tok_cfg.eos_token:
            tokenizer.eos_token = tok_cfg.eos_token
        if tok_cfg.pad_token:
            tokenizer.pad_token = tok_cfg.pad_token
        elif tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Tokenizer loaded. vocab_size={len(tokenizer)}")
        return tokenizer

    def _load_model(self, model_cfg: ModelConfig, device_cfg: DeviceConfig):
        model_kwargs: Dict[str, Any] = dict(model_cfg.model_kwargs)
        model_kwargs["trust_remote_code"] = model_cfg.trust_remote_code
        model_kwargs["revision"] = model_cfg.model_revision

        # dtype
        dtype_str = device_cfg.dtype or model_cfg.torch_dtype
        model_kwargs["torch_dtype"] = (
            getattr(torch, dtype_str) if dtype_str and dtype_str != "auto" else "auto"
        )

        if model_cfg.attn_implementation:
            model_kwargs["attn_implementation"] = model_cfg.attn_implementation
        if model_cfg.use_cache is not None:
            model_kwargs["use_cache"] = model_cfg.use_cache

        # device_map / multi-GPU
        if device_cfg.gpus and not device_cfg.device_map:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_cfg.gpus))
            model_kwargs["device_map"] = "auto"
            logger.info(f"Multi-GPU: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        elif device_cfg.device_map:
            model_kwargs["device_map"] = device_cfg.device_map

        # 量化
        quant_cfg = model_cfg.quantization
        if quant_cfg.enabled:
            if not model_kwargs.get("device_map"):
                model_kwargs["device_map"] = "auto"
                logger.warning("Quantization requires device_map; auto-set to 'auto'.")
            model_kwargs["quantization_config"] = self._build_bnb_config(quant_cfg)

        logger.info(
            "Loading model with kwargs: %s",
            {k: v for k, v in model_kwargs.items() if k != "quantization_config"},
        )
        model = AutoModelForCausalLM.from_pretrained(model_cfg.name_or_path, **model_kwargs)

        # 没有 device_map 且未量化：手动 .to(device)
        if not model_kwargs.get("device_map") and not quant_cfg.enabled:
            target = self._resolve_device(device_cfg)
            model = model.to(target)
            logger.info(f"Model moved to {target}")

        model.eval()

        # Liger Kernel（eval 之后，compile 之前）
        if model_cfg.apply_liger_kernel:
            self._apply_liger(model, model_cfg)

        return model

    # ------------------------------------------------------------------ #
    # 推理：非流式                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        messages: List[Dict[str, Any]],
        gen_config: GenerationConfig,
        tools: Optional[List[Any]] = None,
        open_thinking: bool = False,
    ) -> GenerationResult:
        """同步非流式生成，在工作线程中调用。"""
        prompt = self._apply_chat_template(messages, tools, open_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        hf_kwargs = self._build_hf_kwargs(inputs, gen_config)

        with torch.no_grad():
            output_ids = self.model.generate(**hf_kwargs)

        new_ids = output_ids[0][input_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        from ..utils.parser import parse_response
        content, reasoning, tool_calls = parse_response(text)
        return GenerationResult(
            content=content,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------ #
    # 推理：流式                                                            #
    # ------------------------------------------------------------------ #

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        gen_config: GenerationConfig,
        tools: Optional[List[Any]] = None,
        open_thinking: bool = False,
    ) -> StreamGenerator:
        """
        返回一个同步 generator，每次 yield 一个 token 字符串。
        调用方负责在合适线程中消费（见 RequestQueue）。
        generation 线程的异常通过 generator 的 StopIteration 之后 raise 传出。
        """
        from threading import Thread

        prompt = self._apply_chat_template(messages, tools, open_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        hf_kwargs = self._build_hf_kwargs(inputs, gen_config, streamer=streamer)

        error_holder: Dict[str, Any] = {"exc": None}

        def _run():
            try:
                with torch.no_grad():
                    self.model.generate(**hf_kwargs)
            except Exception as exc:
                error_holder["exc"] = exc
                logger.error("Generation thread error: %s", exc, exc_info=True)
            finally:
                # 确保 streamer 一定结束，避免消费方永久阻塞
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = Thread(target=_run, daemon=True)
        thread.start()

        try:
            yield from streamer
        finally:
            thread.join(timeout=30.0)
            if not thread.is_alive() is False:
                logger.warning("Generation thread did not finish within timeout.")
            if error_holder["exc"] is not None:
                raise error_holder["exc"]

    # ------------------------------------------------------------------ #
    # 内部工具                                                              #
    # ------------------------------------------------------------------ #

    def _apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]],
        open_thinking: bool,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools or None,
            open_thinking=open_thinking,
        )

    def _build_hf_kwargs(
        self,
        inputs,
        gen_config: GenerationConfig,
        streamer: Optional[TextIteratorStreamer] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **gen_config.to_hf_kwargs(),
        }
        if streamer is not None:
            kwargs["streamer"] = streamer
        return kwargs

    @staticmethod
    def _build_bnb_config(quant_cfg) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_8bit=quant_cfg.load_in_8bit,
            llm_int8_threshold=quant_cfg.llm_int8_threshold,
            llm_int8_skip_modules=quant_cfg.llm_int8_skip_modules,
            llm_int8_enable_fp32_cpu_offload=quant_cfg.llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=quant_cfg.llm_int8_has_fp16_weight,
            load_in_4bit=quant_cfg.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        )

    @staticmethod
    def _resolve_device(device_cfg: DeviceConfig) -> str:
        if torch.cuda.is_available():
            return device_cfg.device or "cuda"
        if device_cfg.device and device_cfg.device != "cpu":
            logger.warning(
                f"CUDA not available, falling back to CPU (requested: {device_cfg.device})"
            )
        return "cpu"

    @staticmethod
    def _configure_cuda_precision(device_cfg: DeviceConfig) -> None:
        if device_cfg.device not in {None, "auto", "cuda"}:
            return
        if not torch.cuda.is_available():
            return
        torch.backends.cudnn.allow_tf32 = device_cfg.tf32
        torch.set_float32_matmul_precision("high" if device_cfg.tf32 else "highest")

    @staticmethod
    def _apply_liger(model, model_cfg: ModelConfig) -> None:
        try:
            from transformers.integrations.liger import apply_liger_kernel
            apply_liger_kernel(model, model_cfg.kernel_config)
            logger.info("Liger Kernel applied.")
        except ImportError:
            logger.warning("liger_kernel not installed; skipping.")

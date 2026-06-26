from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import yaml

import dacite


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class TokenizerConfig:
    """Tokenizer loading kwargs"""
    chat_template: Optional[str] = None
    eos_token: Optional[str] = None
    pad_token: Optional[str] = None
    padding_side: str = "left"
    use_fast: bool = True


@dataclass
class QuantizationConfig:
    """BitsAndBytes quantization configuration"""
    load_in_8bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[List[str]] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    @property
    def enabled(self) -> bool:
        return self.load_in_8bit or self.load_in_4bit


@dataclass
class ModelConfig:
    """Model loading configuration"""
    name_or_path: str = ""
    name: str = "default"
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    model_revision: str = "main"
    use_cache: Optional[bool] = None
    attn_implementation: Optional[str] = None
    apply_liger_kernel: bool = False
    kernel_config: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass
class DeviceConfig:
    """Device configuration for model loading and inference"""
    device: str = "cuda"
    dtype: str = "auto"
    tf32: bool = True
    device_map: Optional[str] = None
    gpus: Optional[List[int]] = None


@dataclass
class GenerationConfig:
    """Default generation parameters"""
    temperature: float = 0.7
    top_p: float = 0.92
    top_k: int = 50
    max_tokens: int = 8192
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    early_stopping: bool = False
    no_repeat_ngram_size: int = 0


@dataclass
class MemoryConfig:
    """Memory monitor configuration"""
    max_gpu_memory_ratio: float = 0.8
    kv_cache_timeout_minutes: int = 30
    cleanup_interval: int = 60


@dataclass
class BatchingConfig:
    """Batch processor configuration"""
    enabled: bool = True
    max_batch_size: int = 8
    batch_timeout_ms: int = 50


@dataclass
class LimitsConfig:
    """Server request limits"""
    max_queue_size: int = 100
    request_timeout: int = 300


@dataclass
class ServerConfig:
    """HTTP server configuration"""
    host: str = "0.0.0.0"
    port: int = 8998
    log_level: str = "INFO"
    timeout_keep_alive: int = 5
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)


@dataclass
class AppConfig:
    """Root application configuration"""
    title: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# dacite 加载配置时允许的类型转换规则
_DACITE_CONFIG = dacite.Config(
    # 允许 YAML 中整数字段传入 float（如 timeout: 5 -> float），反之亦然
    cast=[int, float, bool],
    # 对于嵌套 dataclass，dacite 自动递归；对于非 dataclass dict 直接透传
    strict=False,
)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(config_path: str) -> AppConfig:
    """Load and validate configuration from YAML file, returning typed AppConfig"""
    # ── 将 model.tokenizer（若存在）规范化为 dict，dacite 才能映射到 TokenizerConfig ──
    # YAML 中 tokenizer 可能是任意 kwargs dict，字段名与 TokenizerConfig 不完全一致时
    # 多余字段在 strict=False 下会被忽略，缺失字段使用默认值。
    app_cfg: AppConfig = dacite.from_dict(
        data_class=AppConfig,
        data=load_yaml_config(config_path),
        config=_DACITE_CONFIG,
    )
    return app_cfg

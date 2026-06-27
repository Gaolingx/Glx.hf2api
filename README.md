# Glx.hf2api

一个基于 FastAPI + HuggingFace Transformers 构建的轻量级 LLM 推理服务，提供兼容 OpenAI 接口规范的 REST API，支持流式输出、工具调用、思维链推理、在线量化等特性。

---

## 目录

- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 文档](#api-文档)
- [设计亮点](#设计亮点)
- [监控与运维](#监控与运维)

---

## 功能特性

| 特性 | 说明 |
|------|------|
| **OpenAI 兼容接口** | `/v1/chat/completions`、`/v1/models`，可直接替换 OpenAI SDK 的 `base_url` |
| **流式输出（SSE）** | 基于 `TextIteratorStreamer` 的 token 级流式推送，支持 `stream: true` |
| **工具调用（Tool Call）** | 解析模型输出的 `<tool_call>` 标签，转换为 OpenAI 规范的 `tool_calls` 结构 |
| **思维链推理** | 支持 `<thinking>/<think>` 标签，`open_thinking: true` 启用，响应中携带 `reasoning_content` |
| **在线量化（BitsAndBytes）** | 启动时自动量化，支持 INT8 / NF4 等格式，无需预处理模型权重 |
| **多 GPU 推理** | `device_map: auto` 自动分配层到多卡，或通过 `gpus` 指定设备列表 |
| **GPU 内存监控** | 实时监控显存占用率，超阈值时拒绝新请求（503），后台定期清理缓存 |
| **请求队列** | `asyncio.Queue` + 单线程 Executor 保证 GPU 串行访问，支持超时与背压 |
| **Liger Kernel** | 可选集成 Liger Kernel 加速 Transformer 算子（Fused RMSNorm、SwiGLU 等） |

---

## 项目结构

```
.
├── config/
│   └── production.yaml        # 服务配置文件
├── src/
│   ├── __init__.py
│   ├── configs.py             # 配置数据类（ModelConfig / DeviceConfig / ServerConfig 等）
│   ├── server.py              # 主入口：FastAPI app、路由、lifespan
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── engine.py          # InferenceEngine：模型加载与推理核心
│   │   ├── queue.py           # RequestQueue：异步队列与线程调度
│   │   ├── streamer.py        # StreamHandler：流式 token → SSE 事件流
│   │   └── types.py           # 内部数据类型（GenerationConfig / InferenceRequest 等）
│   ├── api/
│   │   ├── __init__.py
│   │   ├── models.py          # Pydantic 请求/响应模型（ChatRequest / Message 等）
│   │   └── middleware.py      # FastAPI 中间件（内存检查）
│   └── utils/
│       ├── __init__.py
│       ├── logging.py         # 日志配置
│       ├── memory.py          # GPU 内存监控与缓存清理
│       └── parser.py          # 模型输出解析（thinking / tool_call 提取）
└── requirements.txt
```

### 模块职责速览

```
server.py          ← HTTP 入口，组装依赖，不含推理逻辑
    │
    ├── RequestQueue   ← asyncio 侧：入队、超时、Future 管理
    │       │
    │       └── InferenceEngine   ← 纯同步：模型加载、generate、generate_stream
    │
    ├── StreamHandler  ← 将 token generator 转换为 SSE 字符串流
    │
    └── MemoryMonitor  ← 显存检查 + 后台清理任务
```

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8 / 12.x（CPU 也可运行，但不推荐生产使用）
- 推荐：NVIDIA GPU，显存 ≥ 16GB（视模型而定）

### 安装

```bash
# 克隆项目
git clone https://github.com/your-org/llms-api-server.git
cd llms-api-server

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

**`requirements.txt` 最小依赖：**

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0
torch>=2.2.0
transformers>=4.40.0
accelerate>=0.30.0
bitsandbytes>=0.43.0            # 在线量化（可选）
liger-kernel>=0.3.0             # Liger Kernel 加速（可选）
```

### 配置

复制并编辑配置文件：

```bash
cp config/production.yaml config/local.yaml
```

最小配置示例：

```yaml
title: "My LLM Server"

model:
  name: "Qwen3-8B"
  name_or_path: "/path/to/your/model"   # 本地路径或 HuggingFace Hub ID
  trust_remote_code: true

device:
  device: "cuda"
  dtype: "bfloat16"

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"

generation:
  temperature: 0.7
  top_p: 0.92
  max_tokens: 8192
```

### 启动服务

```bash
# 使用配置文件启动
python -m src.server --config config/local.yaml

# 命令行参数覆盖（优先级最高）
python -m src.server \
    --config config/local.yaml \
    --device cuda \
    --dtype bfloat16 \
    --port 8080 \
    --log-level DEBUG

# 多卡推理（指定 GPU 0 和 1）
python -m src.server --config config/local.yaml --gpus 0,1
```

### 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 非流式对话
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "你好"}]
  }'

# 流式对话
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'
```

---

## 配置说明

### 完整配置结构

```yaml
title: "LLM Server"

model:
  name: "Qwen3-8B"                  # 对外显示的模型名
  name_or_path: "/models/Qwen3-8B"  # 本地路径 或 HuggingFace repo id
  trust_remote_code: true
  torch_dtype: "bfloat16"           # 默认 dtype（可被 device.dtype 覆盖）
  attn_implementation: "flash_attention_2"  # 注意力实现（可选）
  use_cache: true
  apply_liger_kernel: false         # 是否启用 Liger Kernel

  tokenizer:
    padding_side: "left"
    chat_template: null             # 自定义 chat template（可选）
    eos_token: null
    pad_token: null

  quantization:
    enabled: false                  # 是否启用在线量化
    load_in_8bit: false
    load_in_4bit: false
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true

device:
  device: "cuda"                    # cuda / cpu / cuda:0
  dtype: "bfloat16"                 # float32 / float16 / bfloat16
  device_map: null                  # auto / balanced / sequential（与 gpus 二选一）
  gpus: null                        # [0, 1] 多卡列表
  tf32: true                        # 启用 TF32 加速（Ampere+）

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"
  timeout_keep_alive: 30
  limits:
    max_queue_size: 100             # 队列最大深度
    request_timeout: 300            # 单请求超时（秒）

generation:                         # 服务端默认生成参数
  temperature: 0.7
  top_p: 0.92
  top_k: 50
  max_tokens: 8192
  repetition_penalty: 1.0
  num_beams: 1
  do_sample: true
  no_repeat_ngram_size: 0

memory:
  max_gpu_memory_ratio: 0.90        # 显存占用上限（超过则拒绝请求）
  kv_cache_timeout_minutes: 30      # KV cache 记录过期时间
  cleanup_interval: 300             # 后台清理间隔（秒）
```

### 在线量化配置示例

无需预量化权重，启动时自动量化，适合显存受限场景：

```yaml
# INT8 量化（约节省 50% 显存）
model:
  quantization:
    enabled: true
    load_in_8bit: true

# NF4 量化（约节省 75% 显存，精度损失更小）
model:
  quantization:
    enabled: true
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true  # 双重量化进一步压缩
```

> **注意**：量化会自动设置 `device_map: auto`，无需手动配置。

---

## API 文档

### `POST /v1/chat/completions`

兼容 OpenAI Chat Completions 规范，额外支持 `reasoning_content` 和 `open_thinking`。

**请求体：**

```jsonc
{
  "model": "Qwen3-8B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "今天天气怎么样？"}
  ],

  // 生成参数（可选，覆盖服务端默认值）
  "temperature": 0.7,
  "top_p": 0.92,
  "top_k": 50,
  "max_tokens": 2048,
  "repetition_penalty": 1.0,
  "num_beams": 1,
  "do_sample": true,

  "stream": false,             // 是否流式输出
  "tools": [],                 // 工具列表（OpenAI 格式）
  "open_thinking": false       // 是否启用思维链
}
```

**非流式响应：**

```jsonc
{
  "id": "chatcmpl-req_abc123",
  "object": "chat.completion",
  "created": 1749000000,
  "model": "Qwen3-8B",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "您好！很高兴为您服务。",
      "reasoning_content": "用户在问天气..."  // open_thinking=true 时存在
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 32,
    "completion_tokens": 18,
    "total_tokens": 50
  }
}
```

**流式响应（SSE）：**

```
data: {"choices":[{"delta":{"reasoning_content":"用户在问..."},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"您好"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"！"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**工具调用响应：**

```jsonc
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"北京\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### `GET /v1/models`

返回已加载模型列表。

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "memory": {
    "total_gb": 80.0,
    "allocated_gb": 18.3,
    "cached_gb": 20.1,
    "ratio": 0.229
  },
  "queue_size": 0
}
```

### `GET /metrics`

Prometheus 格式指标，可直接接入 Grafana：

```
# HELP gpu_memory_allocated_gb GPU memory allocated
gpu_memory_allocated_gb 18.3

# HELP gpu_memory_total_gb Total GPU memory
gpu_memory_total_gb 80.0

# HELP gpu_memory_ratio GPU memory utilization
gpu_memory_ratio 0.229

# HELP request_queue_size Current queue depth
request_queue_size 2
```

### 与 OpenAI SDK 集成

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",           # 本地服务无需鉴权
)

# 非流式
response = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[{"role": "user", "content": "写一篇文章"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## 设计亮点

### 1. asyncio 与工作线程严格隔离

GPU 推理在 `ThreadPoolExecutor`（`max_workers=1`）中同步执行，asyncio 事件循环不阻塞。`asyncio.Future` 仅在事件循环线程中创建和 resolve，工作线程通过 `call_soon_threadsafe` 回写结果，避免竞态条件：

```python
# 工作线程（安全）
loop.call_soon_threadsafe(future.set_result, result)

# 事件循环侧
result = await asyncio.wait_for(request.future, timeout=300)
```

### 2. 串行队列取代伪批处理

单 GPU 场景下，原始批处理器会静默丢弃非首个请求的参数。本项目采用 `asyncio.Queue` + 单线程 Executor 的串行调度方案，每个请求独立参数，语义完全正确，GPU 互斥访问由线程池保证。

### 3. 流式状态机：三阶段解析

`StreamHandler` 内置 `thinking → content → tool_call` 状态机，在 token 流式到达时实时解析，支持多个 tool_call 连续输出，并使用安全缓冲区防止标签在 token 边界被截断：

```
thinking 阶段  → 等待 </thinking> 标签，收集 reasoning_content
content  阶段  → 实时转发，末尾保留 (len("<tool_call>") - 1) 字节缓冲
tool_call 阶段 → 等待 </tool_call> 闭合，解析 JSON，分块发送 arguments
```

### 4. 生成参数三层优先级

```
服务端配置默认值（config.yaml）
        ↓  被覆盖
请求直接字段（temperature / top_p / ...）
```

`build_generation_config()` 一处合并，只有请求中显式设置的字段（非 `None`）才会覆盖默认值，消除了原有的三层混乱参数结构。

### 5. beam search 与采样参数互斥处理

`GenerationConfig.to_hf_kwargs()` 内部统一处理参数互斥，调用方无需关心：

```python
if self.num_beams > 1:
    # beam search 模式强制关闭采样，temperature/top_p 无效
    kwargs["do_sample"] = False
    kwargs["early_stopping"] = True
else:
    # 仅在采样模式下传入 temperature / top_p / top_k
    if self.do_sample:
        kwargs["temperature"] = self.temperature
        ...
```

### 6. 在线量化（零预处理）

通过 `BitsAndBytesConfig` 在模型加载阶段自动量化，支持 INT8 和 NF4 格式，无需预先量化权重文件，适合快速部署资源受限场景：

```
原始权重（FP16/BF16）
       ↓  model load 阶段自动量化
INT8 / NF4 量化权重（常驻 GPU）
       ↓  forward 时自动反量化为 compute_dtype
BF16 计算
```

---

## 监控与运维

### 内存管理

服务内置两层显存保护：

1. **请求前检查**（`MemoryCheckMiddleware`）：显存占用超过 `max_gpu_memory_ratio` 时，以 HTTP 503 拒绝新请求
2. **后台清理**（`MemoryMonitor.run_cleanup_loop`）：每隔 `cleanup_interval` 秒调用 `torch.cuda.empty_cache()` 释放碎片缓存

### 优雅停机

`Ctrl+C` 或 `SIGTERM` 触发 lifespan 清理流程：

```
1. 停止接收新请求
2. 取消 dispatch_loop
3. 等待当前推理任务完成（ThreadPoolExecutor.shutdown(wait=True)）
4. 停止内存清理任务
```

### 日志级别

```bash
# 开发调试
python -m src.server --log-level DEBUG

# 生产环境（仅 WARNING+）
python -m src.server --log-level WARNING
```

`INFO` 级别以上会自动抑制 `transformers` 和 `torch` 的冗余日志。

---

## 常见问题

**Q: 启动时报 `CUDA out of memory`？**

尝试以下方案（按显存节省从少到多排列）：
1. 降低 `dtype` 为 `float16`
2. 启用 `quantization.load_in_8bit: true`
3. 启用 `quantization.load_in_4bit: true`（推荐 NF4）
4. 多卡分布：`gpus: [0, 1]` 或 `device_map: auto`

**Q: 如何支持需要自定义 `chat_template` 的模型？**

在配置文件中直接填写 Jinja2 模板字符串：

```yaml
model:
  tokenizer:
    chat_template: "{% for message in messages %}..."
```

**Q: 流式输出卡住不动？**

检查 `generation_thread` 是否因异常退出，`TextIteratorStreamer` 的 `end()` 调用保证了即使生成线程崩溃也会解除消费方阻塞，错误信息会通过 SSE error chunk 返回。

**Q: `tool_calls` 的 `arguments` 是字符串还是对象？**

与 OpenAI 规范一致，`arguments` 始终为 **JSON 字符串**，客户端需要 `json.loads()` 后使用。

---

## License

MIT

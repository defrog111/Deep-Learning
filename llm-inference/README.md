# LLM Inference

一个从本地推理演进到 Cloud LLM inference 与 coding-agent control plane 的学习项目，当前支持：

- Hugging Face Transformers 本地模型
- 自动选择 CPU、Apple Silicon MPS 或 CUDA（由 Accelerate 处理）
- 单次命令行推理
- OpenAI 风格的 `POST /v1/chat/completions` 接口
- 模型延迟加载：健康检查不会触发模型下载
- Task-complexity model tier routing
- KV-token-aware TTFT 预测与 worker 选择
- Coding-session intent drift 与 context reset
- Worker heartbeat、健康 TTL 和 explainable route decision

## 学习路线

建议先完成 inference 基础，再阅读 Router 和 agent control plane：

- [第一阶段：Prefill、Decode 与 KV Cache](docs/01_INFERENCE_BASICS.md)
- [Cloud LLM Inference 与 Coding Agent 完整流程](docs/CLOUD_LLM_INFERENCE_FLOW.md)
- [Intelligent Router 公式与逐函数说明](docs/ROUTER_DESIGN.md)
- 实验命令：`llm-prefix-cache-lab`

实验工具会向启用了 Automatic Prefix Caching 的 vLLM server 连续发送两个共享长前缀的流式请求，并比较 TTFT、近似 TPOT、prompt tokens 和 prefix-cache metrics。

## 快速开始

建议使用 Python 3.10+；项目最低支持 Python 3.9。

```bash
cd llm-inference
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

首次推理会从 Hugging Face 下载默认模型 `Qwen/Qwen2.5-0.5B-Instruct`。

### 命令行

```bash
llm-infer "用三句话解释 Transformer 的 KV cache"
```

指定其他模型或设备：

```bash
llm-infer "Hello" --model Qwen/Qwen2.5-1.5B-Instruct --device mps
```

### HTTP 服务

```bash
llm-serve
```

另开终端调用：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "什么是 speculative decoding？"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

健康检查与交互式接口文档：

- `GET http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

Control-plane API：

- `PUT /v1/workers/{worker_id}/heartbeat`
- `GET /v1/workers`
- `POST /v1/routing/decisions`

## 配置

环境变量示例见 `.env.example`。常用变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face 模型名或本地路径 |
| `LLM_DEVICE` | `auto` | `auto`、`cpu`、`mps` 或 `cuda` |
| `LLM_DTYPE` | `auto` | `auto`、`float16`、`bfloat16` 或 `float32` |
| `LLM_HOST` | `127.0.0.1` | 服务监听地址 |
| `LLM_PORT` | `8000` | 服务端口 |

## 测试

测试不会下载模型：

```bash
pytest
ruff check .
```

## vLLM Prefix Cache 实验

在 GPU node 上启动服务：

```bash
vllm serve Qwen/Qwen3-8B \
  --enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000
```

然后在 client 端运行：

```bash
llm-prefix-cache-lab \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B
```

## Intelligent Router dry-run

先注册一个模拟 vLLM worker：

```bash
curl -X PUT http://127.0.0.1:8000/v1/workers/gpu-1/heartbeat \
  -H 'Content-Type: application/json' \
  -d '{
    "worker_id": "gpu-1",
    "endpoint": "http://gpu-1:8000",
    "model_id": "kimi-reasoning",
    "model_tier": "reasoning",
    "max_context_tokens": 131072,
    "max_concurrency": 16,
    "estimated_queue_ms": 40,
    "prefill_tokens_per_second": 20000,
    "first_token_decode_ms": 12,
    "network_rtt_ms": 2,
    "kv_cache_usage_percent": 0.45,
    "cache_entries": []
  }'
```

请求一次可解释决策：

```bash
curl http://127.0.0.1:8000/v1/routing/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{
      "role": "user",
      "content": "分析架构、修复异常、实现代码并运行测试"
    }],
    "prompt_tokens": 10000,
    "reusable_prefix_tokens": 9500
  }'
```

响应包含最低模型 tier、intent drift、context action，以及每个 worker 的 matched KV tokens、预计 prefill、TTFT 和选择原因。

## 目录结构

```text
src/llm_inference/
├── api.py       # FastAPI 路由
├── backend.py   # Transformers 推理后端
├── cli.py       # 命令行入口
├── config.py    # 环境配置
├── routing.py   # complexity、intent、KV-aware control plane
├── schemas.py   # OpenAI 风格数据结构
└── server.py    # HTTP 服务入口
```

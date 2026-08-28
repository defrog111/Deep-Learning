# LLM Inference

一个小而清晰的本地 LLM 推理项目，当前支持：

- Hugging Face Transformers 本地模型
- 自动选择 CPU、Apple Silicon MPS 或 CUDA（由 Accelerate 处理）
- 单次命令行推理
- OpenAI 风格的 `POST /v1/chat/completions` 接口
- 模型延迟加载：健康检查不会触发模型下载

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

## 目录结构

```text
src/llm_inference/
├── api.py       # FastAPI 路由
├── backend.py   # Transformers 推理后端
├── cli.py       # 命令行入口
├── config.py    # 环境配置
├── schemas.py   # OpenAI 风格数据结构
└── server.py    # HTTP 服务入口
```

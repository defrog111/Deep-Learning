# 第一阶段：LLM Inference、Prefill、Decode 与 KV Cache

这一阶段只研究单个 vLLM 实例内部的一次请求。先不讨论多模型 Router、Kubernetes 或 Agent orchestration。

## 1. 一个 request 进入 vLLM 后发生什么

以 OpenAI-compatible Chat API 为例，请求大致经历：

```text
HTTP request
  ↓
校验参数 + 应用 chat template + tokenize
  ↓
Admission / scheduler queue
  ↓
Prefill（也叫 prompt/context phase）
  ↓
生成当前请求的 KV Cache
  ↓
Decode token 1
  ↓
Decode token 2 → token 3 → ...
  ↓
SSE streaming / final response
```

vLLM 和简单 `transformers.generate()` 最大的 serving 差异之一，是它不会等一个静态 batch 全部完成再接下一批请求。调度器会持续把新请求、prefill token 和正在 decode 的请求组合起来执行，这就是 continuous batching 的基本思路。

用户看到的 TTFT 更完整地写是：

```text
TTFT ≈ network + API/tokenize + queue + prefill + first decode step + stream flush
```

在本机压测时 network 很小，可以近似关注：

```text
TTFT ≈ queue time + prefill time
```

但线上不能忘记 gateway、排队和网络，否则会误把所有慢请求都归因于 GPU。

## 2. Prefill 是什么

假设 prompt token 是：

```text
x1, x2, x3, ..., x10000
```

Prefill 会一次处理这段已知序列。每一层都计算：

```text
Q = X Wq
K = X Wk
V = X Wv
Attention(Q, K, V)
```

由于 10,000 个输入 token 已经全部已知，GPU 可以对大量 token 做并行矩阵计算。Prefill 通常具有较高计算密度，长 prompt 会增加 prefill 时间，并显著推高 TTFT。

Prefill 的输出不只是“下一个 token 的概率”，还包括每一层、每个历史 token 的 K/V。这些 K/V 会被保存，供后续 decode 使用。

为什么不缓存 Q？某个历史 token 的 Q 只用于该 token 当时的 attention；未来新 token 需要的是自己的新 Q，以及全部历史 token 的 K/V。

## 3. Decode 是什么

Prefill 后开始自回归生成：

```text
已有 10,000 tokens
  → 预测 token 10,001
  → 把它的 K/V 追加到 cache
  → 预测 token 10,002
  → 再追加 K/V
  → ...
```

第 `t` 步的输入 token 依赖第 `t-1` 步采样结果，所以单个序列无法一次并行生成未来 100 个 token。每次 decode 只新增一个 token 的 Q/K/V，但要读取很长的历史 KV cache。

因此通常可以这样形成直觉：

- Prefill：输入 token 多，偏 compute-bound，主要影响 TTFT。
- Decode：逐 token 串行，频繁读取权重和 KV，往往偏 memory-bandwidth-bound，主要影响 TPOT/ITL。

这不是绝对定律；batch size、模型结构、GPU、量化、context 长度和调度方式都会改变瓶颈。

## 4. KV Cache 为什么存在

Transformer 的 causal attention 保证未来 token 可以读取过去 token，而过去 token 的表示不需要因为未来 token 到来而重新计算。

如果没有 KV cache，生成第 10,002 个 token 时又要重新计算前 10,001 个 token 的 K/V；下一步还要再算一遍。生成越长，重复计算越严重。

KV cache 保存每一层的历史 K/V：

```text
Layer 0: K[1..t], V[1..t]
Layer 1: K[1..t], V[1..t]
...
Layer L: K[1..t], V[1..t]
```

下一步只需要：

1. 计算新 token 的 Q/K/V。
2. 用新 Q attention 到缓存的历史 K/V。
3. 把新 K/V append 到 cache。

代价是显存。近似公式：

```text
KV bytes ≈ 2 × layers × kv_heads × head_dim
             × sequence_length × batch × bytes_per_element
```

其中 `2` 是 K 和 V。GQA/MQA 减少 `kv_heads`，因此能明显降低 KV cache 占用。

## 5. 普通 KV Cache 与 Prefix Cache 的区别

这两个概念容易混淆：

- Request KV cache：服务当前请求的 decode，避免每步重算历史 token。
- Automatic Prefix Caching（APC）：请求结束后保留可复用前缀的 KV blocks，让后续请求跳过共享部分的 prefill。

例子：

```text
Request 1 = [system + repo + history 9,500 tokens] + [question A]
Request 2 = [system + repo + history 9,500 tokens] + [question B]
```

第二次请求若命中 APC：

```text
9,500-token shared prefix → reuse cached KV blocks
question B               → only prefill new suffix
answer                    → decode normally
```

所以 APC 的直接作用是减少重复 prefill：

- 通常降低 TTFT。
- 降低 prefill GPU 计算量，提高系统可用吞吐。
- 不会跳过 answer 的自回归 decode，因此通常不会显著降低 TPOT。

APC 不是 response cache。问题 B 仍然会真正运行模型并生成答案。

## 6. Prefix 命中的必要条件

“文本看起来一样”不一定命中。重要的是 tokenized prefix 一致，并且缓存仍然存在：

- 同一个模型、tokenizer 和兼容的 model/adapter 配置。
- Chat template、system prompt、工具定义、空格和消息顺序产生相同 token prefix。
- 请求到达保存该 cache 的实例，或系统配置了跨实例 KV transfer。
- cache block 没有因容量压力被淘汰。
- 可复用长度通常受 block/match boundary 影响。

实验时应该在隔离的 vLLM 实例上顺序发送请求，避免其他流量污染 counter 和排队时间。

## 7. 亲手实验

### 7.1 在 GPU node 安装并启动

先确认 GPU 和 vLLM 版本：

```bash
nvidia-smi
vllm --version
```

启动 Qwen3-8B：

```bash
vllm serve Qwen/Qwen3-8B \
  --enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000
```

如果单卡放不下，再根据节点 GPU 数和模型要求配置 tensor parallel；不要为了这个实验先引入多节点变量。

另开一个 GPU-node 终端观察利用率和显存：

```bash
nvidia-smi dmon -s pucm -d 1
```

这里的 GPU utilization 是辅助证据：采样周期、kernel 很短和其他请求都会让曲线难以解释，不能替代 vLLM 的 token/cache/latency metrics。

### 7.2 运行本仓库实验工具

安装项目后执行：

```bash
cd llm-inference
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

llm-prefix-cache-lab \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B \
  --prefix-characters 40000 \
  --max-tokens 64
```

工具执行：

1. 构造一份稳定的长代码仓库上下文。
2. 先发一个不含实验前缀的短请求，排除首次 kernel/JIT warm-up。
3. 发送 request 1，建立 prefix KV cache。
4. 用完全相同的长前缀发送 request 2，但问题不同。
5. 从流式响应计算客户端 TTFT、近似 TPOT 和 E2E latency。
6. 请求前后抓取 `/metrics`，计算 prefix query/hit token 差值。

`--prefix-characters` 不是精确 token 数。真正的 `prompt_tokens` 使用 vLLM 响应里的 usage。

### 7.3 直接查看 metrics

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E \
  'prefix_cache|prompt_tokens|time_to_first_token|request_prefill|kv_cache_usage'
```

重点关注：

- `vllm:prefix_cache_hits`
- `vllm:prefix_cache_queries`
- `vllm:prompt_tokens_cached`
- `vllm:prompt_tokens`
- `vllm:request_prefill_time_seconds`
- `vllm:request_queue_time_seconds`
- `vllm:time_to_first_token_seconds`
- `vllm:request_time_per_output_token_seconds`
- `vllm:kv_cache_usage_perc`

Histogram 不能通过读取一次 `_sum` 就得到某个请求的精确值；在无并发实验里可以观察前后差值，生产环境则应用 PromQL 按时间窗口计算分位数。

## 8. 预期结果和判断方法

理想结果：

```text
Request 1:
  prefix hit tokens ≈ 0
  TTFT             较高

Request 2:
  prefix hit tokens 接近共享 prefix token 数
  TTFT             明显降低
  TPOT             与 request 1 大致同量级
```

不要强求 request 2 命中精确的 9,500 tokens。block boundary、chat template 和问题分叉位置会影响可复用 token 数。

如果第二次没有更快，按顺序检查：

1. `prefix_cache_queries` 是否增长。
2. `prefix_cache_hits` 是否增长。
3. 两次请求的 system/repository prefix 是否逐 token 相同。
4. 是否误发到不同进程或不同 replica。
5. cache 是否因 GPU KV 空间压力被淘汰。
6. queue time 是否掩盖了 prefill 节省。
7. 第一次是否包含模型/JIT warm-up；必要时先发一个不参与统计的短请求。

## 9. 第一阶段通过标准

你应当能够不用背定义，自己解释：

1. 为什么长 prompt 主要推高 TTFT。
2. 为什么长 output 主要放大 decode 总时间。
3. 为什么 decode 必须保存历史 K/V，但不保存历史 Q。
4. 为什么 prefix caching 命中后 TTFT 下降，而 TPOT 通常基本不变。
5. 为什么 cache-aware routing 必须同时考虑 cache locality 和 queue time。
6. 如何从 `/metrics` 证明缓存真的命中，而不是凭一次 latency 猜测。

## 官方参考

- [vLLM Automatic Prefix Caching example](https://docs.vllm.ai/en/stable/examples/features/automatic_prefix_caching/)
- [vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/)
- [vLLM serve CLI](https://docs.vllm.ai/en/stable/cli/serve/)

# Cloud LLM Inference 与 Coding Agent 完整流程

本文描述一个类似 Claude Code/Codex 的 coding agent 在自建 vLLM cloud 上，从用户请求进入到工具执行、模型推理和最终验证的完整工程流程。

## 1. 系统边界

```text
                        CONTROL PLANE
  model registry | routing policy | worker registry | experiments
                         |       ↑
                         v       | heartbeat / KV events

Client → API Gateway → Agent Orchestrator → Intelligent Router
                                              |
                                              v
                    vLLM Replica / GPU Worker Pool
                    scheduler → prefill → KV → decode
                                              |
                                              v
                         SSE token stream → Client/Agent

                         DATA PLANE
```

### Control plane

- 模型目录：model ID、tier、context、quantization、adapter。
- Worker registry：健康、容量、queue、throughput、KV directory。
- Routing policy：complexity、intent、cache、SLO、成本。
- 策略发布：版本、shadow、canary、回滚。

### Data plane

- API gateway：auth、tenant quota、rate limit、request ID。
- Agent orchestrator：状态、工具、预算、停止条件。
- vLLM：tokenize、continuous batching、prefill、KV、decode、stream。
- Tool workers：repo、terminal、test、browser 等隔离执行环境。

## 2. 一次 coding-agent turn 的完整生命周期

### Step 1：Ingress

Gateway 接收：

- tenant/user/session/task ID；
- messages 和 tool schemas；
- deadline、priority、最大 token/step/cost；
- model override（若产品允许）；
- idempotency key 和 trace ID。

执行鉴权、配额、payload/context 上限、安全预检查。外部副作用不能依赖 HTTP retry 猜测是否执行成功，必须使用幂等 key 或持久化 tool-call 状态。

### Step 2：加载 Agent state

Orchestrator 加载：

- 当前目标和成功标准；
- repository snapshot/commit；
- 已完成步骤和 tool observations；
- 未解决 blocker；
- token/time/cost budget；
- approval policy。

不要把所有原始日志无限塞回 prompt。应区分 durable task state、可压缩对话历史和可重新获取的 tool output。

### Step 3：Intent drift

比较最新 user goal 与 active goal：

```text
低 drift → 延续 task/history/KV prefix
高 drift → 新 task state，只保留稳定 system/tool policy，重新检索 repo
```

这一步必须早于 KV lookup。否则 Router 可能因为旧历史有 cache，就继续携带已经无关的目标。

### Step 4：构造有效 context

Context builder 按稳定性排序：

1. system、安全和审批政策；
2. 精简工具定义；
3. active goal、constraints、success criteria；
4. repository map 和相关文件；
5. 已验证 tool observations；
6. 必要的对话/计划摘要；
7. 最新 user turn。

然后用目标模型 tokenizer 得到精确：

- `prompt_tokens`；
- 可复用 prefix token 数；
- block-aligned prefix fingerprint。

超过 context 时应按语义 compact/retrieve，不应简单从头截断 system policy。

### Step 5：Task complexity routing

Router 估计“完成这个 turn 所需的最低能力”：

- 简单解释、格式化、单文件小改动 → fast；
- 多文件实现、常规 debugging、少量工具 → balanced；
- 架构、复杂根因、长计划、多工具验证 → reasoning。

还可以同时选择 reasoning effort、max output、可见工具子集和是否允许并行工具调用。质量 tier 是约束，不只是与 latency 混成一个任意分数。

### Step 6：Worker admission 与 KV-aware routing

先过滤：

- heartbeat 过期/unhealthy；
- model/tokenizer/adapter 不匹配；
- context capacity 不足；
- tenant/region/data-policy 不允许；
- circuit breaker 已打开。

再对候选计算：

```text
new_prefill_tokens = prompt_tokens - matched_prefix_tokens

TTFT estimate = network + queue
              + new_prefill_tokens / prefill_throughput
              + first_decode
```

Cache-aware 不等于 cache-first。路由目标是满足质量约束后的最低 deadline/cost risk。

### Step 7：请求进入 vLLM

vLLM data plane 执行：

1. OpenAI-compatible API 校验和 chat template。
2. Tokenization/context admission。
3. Scheduler 把请求放入 waiting queue。
4. 查询 local/external prefix cache，确定 matched blocks。
5. 为未缓存 prompt 执行 prefill。
6. 分配/追加 paged KV blocks。
7. Continuous batching 调度 decode。
8. Sampling/stop/tool-call token 处理。
9. SSE 逐 token 返回并传播 cancel/disconnect。

完整 TTFT：

```text
TTFT = gateway + route + network + queue + cache lookup
     + prefill + first decode + stream flush
```

TPOT 主要由 decode batch、权重/KV 内存访问、context 长度、GPU 和调度影响。Prefix cache 跳过的是共享 prefill，不会跳过新答案的 decode。

### Step 8：解析模型输出

输出可能是：

- final answer；
- tool call；
- clarification request；
- structured plan；
- refusal/安全结果。

Orchestrator 必须验证 tool name、schema、参数、权限、预算和审批要求，不能把模型生成的 JSON 直接作为 shell 命令执行。

### Step 9：工具执行

Coding agent 常见工具：

- repository search/read；
- patch/write；
- terminal/build/test；
- browser/docs；
- git diff/status；
- issue/CI/log systems。

工具层要求 sandbox、cwd/target 范围、timeout、stdout 限制、secret isolation、network policy 和 side-effect classification。结果带 `call_id` 写入 durable state，重试时先查是否已完成。

### Step 10：Observation 回到下一轮 inference

Tool output 经裁剪/结构化后追加到 active context。每轮重新执行：

```text
intent/context check → complexity → KV-aware worker → inference
```

原因是工具输出可能让任务从简单升级为复杂，例如测试暴露跨模块 failure；也可能因 prompt prefix 增长而改变最佳 cache worker。

### Step 11：Verification 和终止

成功条件应是可验证结果，而不是模型说“完成了”：

- 目标文件存在且 diff 符合范围；
- tests/lint/build 通过；
- 没有未授权副作用；
- 最终回答包含证据和剩余风险。

停止条件：完成、需要用户审批、明确 blocked、超出 step/token/time/cost budget、重复无进展或安全策略终止。

## 3. 三类智能路由如何协作

```text
Task complexity
  决定最低模型质量 / reasoning / tool capability

Intent drift
  决定 active task state 和真正可复用的 context

KV awareness
  在合格 worker 中，用可加载 token 数预测 prefill/TTFT
```

错误做法：

- 只按 prompt 长度判断复杂度；
- 一律选择最大模型；
- 看到 cache hit 就忽略 queue；
- intent drift 后仍发送全部旧历史；
- 用全局 cache hit rate 推断某个 prefix 在哪个 replica；
- 把三个信号无约束地混成一个无法解释的分数。

## 4. 观测指标

### 用户/SLO

- request success、agent task success；
- TTFT p50/p95/p99；
- TPOT/ITL；
- end-to-end latency；
- deadline miss/cancel rate。

### vLLM

- running/waiting requests；
- queue/prefill/decode time；
- prompt/generation tokens；
- prefix cache query/hit/cached tokens；
- KV usage、eviction、preemption；
- GPU utilization、memory、MFU。

### Router

- required tier / selected tier / degraded rate；
- predicted vs actual TTFT；
- matched-prefix prediction error；
- cache-locality win/loss；
- overserve rate 和 cost per successful task；
- policy version 和 route reason。

### Agent

- task/test success；
- steps/tool calls/retries；
- invalid/redundant tool calls；
- approval、安全和 sandbox violations；
- intent false-reset / false-carry-over；
- token/cost per successful task。

## 5. 评测和上线

1. 建立真实 coding task 集：解释、单文件、多文件、debug、test、架构。
2. 对每个任务运行多个 model/tier/reasoning 配置，得到成功标签。
3. 离线 replay Router，比较 quality-constrained latency/cost。
4. Shadow：记录新决策但仍走旧路由。
5. Canary：少量租户/流量，检查 SLO、质量和安全 guardrail。
6. 按 policy version 自动回滚。

不能只优化 token、latency 或 cache hit。最终 objective 应类似：

```text
minimize latency + compute cost + deadline risk
subject to task success ≥ target
           safety violations = 0
           tenant/policy constraints satisfied
```

## 6. 常用工程组件

- vLLM/SGLang：模型 data plane。
- Kubernetes + GPU operator/device plugin：部署和 GPU 调度。
- Prometheus/Grafana：metrics 和 SLO。
- OpenTelemetry：跨 gateway/router/worker/tool trace。
- Redis/etcd/control-plane DB：短期 registry、lease、policy/state；按一致性需求选择。
- Object store/SQL/warehouse：trace、eval、训练数据和成本分析。
- Sandbox runtime：coding tool 的隔离执行。

不要一开始把所有组件都引入。当前仓库先用内存 registry 和 dry-run API验证决策，再连接真实 vLLM metrics/KV events。

## 7. 官方参考

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/examples/features/automatic_prefix_caching/)
- [vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/)
- [vLLM KV Connector contract](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/base/)
- [OpenAI model and coding-agent guidance](https://developers.openai.com/api/docs/guides/latest-model)

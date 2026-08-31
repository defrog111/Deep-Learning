# Intelligent Router：设计、公式与函数说明

这个 Router 是 Cloud LLM inference 的控制面原型。它不替代 vLLM scheduler，而是在请求进入某个 vLLM replica 之前决定：

1. 最低需要哪个模型 tier。
2. 当前会话历史是否仍与新目标相关。
3. 哪个 replica 能复用最多的有效 KV prefix。
4. cache locality 是否值得额外排队。

## 1. 路由决策顺序

```text
messages
  ↓
intent_drift(messages)
  ↓
reusable_prefix(messages, drift)
  ├─ reuse_history
  └─ reset_to_system
  ↓
prefix fingerprint + reusable prefix tokens
  ↓
analyze_task(messages)
  ↓
minimum model tier: fast / balanced / reasoning
  ↓
filter unhealthy / insufficient-context workers
  ↓
enforce minimum quality tier
  ↓
estimate matched KV tokens and TTFT per worker
  ↓
select minimum routing_cost_ms
```

顺序非常重要。Intent drift 必须发生在 cache lookup 之前。如果用户已经从“修 parser”切换到“设计数据库”，Router 应先决定只保留稳定 system contract，再对新的有效 prefix 计算 fingerprint。仅仅降低旧 cache 的权重，仍会把无关历史发送给模型。

## 2. Task complexity

`analyze_task()` 使用可解释规则提取：

- `code`：代码块、函数、异常、测试等。
- `reasoning`：架构、调试、根因、权衡等。
- `multi_step`：计划、实现、测试、部署等多阶段任务。
- `tool_use`：repository、terminal、file、browser、test suite 等。
- `long_conversation`：长会话。
- prompt 长度。

然后映射为最低模型 tier：

```text
complexity < 0.30  → fast
complexity < 0.62  → balanced
otherwise          → reasoning
```

这是生成数据的 baseline，不是最终 ML classifier。生产版本应学习：

```text
P(task succeeds | task features, model, reasoning config, tool policy)
```

目标不是预测一个抽象的“难度”，而是在满足成功率/SLO 的条件下选择成本最低的配置。

Router 把最低 tier 当作质量约束：只要存在满足 tier 的 worker，低 tier 节点即使完全空闲也不能抢赢。如果没有满足 tier 的节点，则选择最高可用 tier，并将 `degraded=true` 返回给 gateway，由策略决定排队、降级或拒绝。

## 3. Intent drift

`intent_drift()` 比较最近两个 user goal：

- 显式换题词，例如 `new task`、`instead`、`新任务`，直接给高 drift。
- 否则计算词项 cosine-like overlap。
- `run tests`、`fix it` 一类短 follow-up 被限制在低 drift，避免误切任务。

当前策略：

```text
drift < 0.75  → reuse_history
drift ≥ 0.75  → reset_to_system
```

生产版可加入：

- embedding similarity；
- repo、文件路径、语言和工具集合变化；
- active plan/issue ID 变化；
- 用户纠正或撤销；
- classifier 的 false-reset 与 false-carry-over eval。

Intent drift 的输出还应驱动 agent state：新 task ID、重新检索 repository、废弃旧计划和 tool observations。当前代码只返回 `context_action`，不会自动删除外部状态。

## 4. KV-cache awareness

每个 worker heartbeat 上报 `cache_entries`：

```json
{
  "prefix_fingerprint": "f31bd17d63c0f9e2",
  "cached_tokens": 9500
}
```

Router 只相信当前 heartbeat 中实际可用的最大 prefix：

```text
matched_prefix_tokens = min(
    advertised_cached_tokens,
    reusable_prefix_tokens,
    prompt_tokens
)
```

然后估算：

```text
new_prefill_tokens = prompt_tokens - matched_prefix_tokens

estimated_prefill_ms =
    new_prefill_tokens / prefill_tokens_per_second × 1000

estimated_ttft_ms =
    network_rtt_ms
    + estimated_queue_ms
    + estimated_prefill_ms
    + first_token_decode_ms
```

因此一个有 9,500-token cache 的 worker 不一定胜出。如果它要排队 1,200 ms，而冷节点重新 prefill 只要 1,000 ms，Router 应选择冷节点。

对于简单任务，使用更强、更贵的模型还会加入 over-serving penalty：

```text
routing_cost_ms =
    estimated_ttft_ms
    + tier_gap × overserve_penalty_ms_per_tier
```

这个 penalty 是 latency-equivalent policy cost，不代表真实美元。生产系统应把 GPU-second、模型定价、租户优先级和 deadline 转成统一 objective，并用离线 replay/shadow traffic 校准。

### Cache directory 从哪里来

Prometheus 的 aggregate hit rate 能用于监控，却不能回答“某个 prefix 当前在哪个 replica”。生产 Router 需要一个 cache directory：

- 订阅 vLLM/connector 的 KV cache store/remove events；
- 或由每个 worker 汇总本地 prefix block 状态；
- 记录 model、tokenizer、LoRA adapter、block layout 和 tenant/salt；
- heartbeat/TTL 到期立即停止使用该条目；
- 大规模时使用分片索引、radix tree、Bloom filter 或两级 lookup，不能无限广播 hash 列表。

Fingerprint 必须基于最终 chat template 之后的 token IDs。当前学习代码对消息 JSON 做 SHA-256，只用于控制面演示。

## 5. Worker heartbeat 字段

| 字段 | 作用 | 生产来源 |
| --- | --- | --- |
| `model_id/model_tier` | 模型兼容和质量层级 | deployment/model registry |
| `max_context_tokens` | admission control | model config |
| `running_requests` | 当前并发 | vLLM metrics |
| `queue_depth` | 排队请求数 | vLLM metrics |
| `estimated_queue_ms` | 请求预计等待时间 | queue model / recent histogram |
| `prefill_tokens_per_second` | 估算未缓存 token 成本 | online EWMA，按长度 bucket |
| `first_token_decode_ms` | 首个 decode step | online EWMA |
| `network_rtt_ms` | gateway 到 worker 网络成本 | active/passive probing |
| `kv_cache_usage_percent` | cache pressure/观测 | vLLM metrics |
| `cache_entries` | prefix → available tokens | KV events/cache directory |

Heartbeat TTL 默认 30 秒。过期 worker 从 `healthy_workers()` 中消失。

## 6. 每个函数的作用

### `estimate_tokens(messages)`

用字符数除以 4 得到无 tokenizer 时的廉价估算。它避免 control plane 加载大模型 tokenizer，但中文和代码误差可能很大。API 支持由 gateway 传入精确 `prompt_tokens` 和 `reusable_prefix_tokens`。

### `_hash_messages(messages)`

把消息 canonical JSON 做 SHA-256，得到实验用 prefix fingerprint。它是私有函数，因为生产实现必须替换为基于最终 token/block hash 的实现。

### `_leading_system_messages(messages)`

提取消息开头连续的 system contract。Intent drift 高时，旧 conversation/task history 被放弃，但安全规则、工具政策等稳定 system context 仍可保留。

### `reusable_prefix(messages, drift_score)`

先做 context policy：稳定任务返回最新 user turn 之前的完整历史；高漂移只返回 leading system messages，并输出 `reset_to_system`。

### `prefix_fingerprint(messages, drift_score=0)`

对真正准备复用的 prefix 生成 fingerprint。测试和 cache directory producer 必须调用相同规范，否则永远无法命中。

### `_terms(text)`

把英文标识符、路径和中文字符提取为集合，为 explainable intent baseline 提供输入。

### `intent_drift(messages)`

输出 0～1 的 session goal drift。它不是 dataset drift，也不是模型质量分数。

### `analyze_task(messages, prompt_tokens, reusable_prefix_tokens)`

一次完成 complexity features、最低 tier、intent drift、context action、prefix fingerprint 和 token 数量，生成 `TaskSignals`。把所有路由输入放在一个可记录对象中，方便离线 replay。

### `WorkerRegistry.update(heartbeat)`

原子替换一个 worker 的最新状态并记录本地更新时间。重复 heartbeat 是幂等更新。

### `WorkerRegistry.healthy_workers()`

只返回 `healthy=true` 且 heartbeat 未超过 TTL 的 worker，避免把请求发给失联 replica。

### `WorkerRegistry.all_workers()`

返回完整 registry，供管理/调试 API 使用；其中可能包含过期或 unhealthy worker，不能直接用于路由。

### `IntelligentRouter.route(...)`

执行完整控制面策略：分析任务、context admission、质量 tier 约束、降级规则、候选估算、排序和解释性 decision。

### `IntelligentRouter._estimate(worker, signals)`

把某个 worker 的 cache、queue、prefill throughput、network 和 tier 成本转换成 `CandidateScore`。该函数是纯计算，适合用历史 trace 离线 replay 和调参。

### `get_worker_registry()` / `get_router()`

FastAPI dependency provider。测试可用 `dependency_overrides` 注入隔离 registry，而不用修改全局状态。

## 7. API 与实验工具函数

### `health()`

只验证 API process 可响应，不触发模型加载，也不代表下游 GPU worker 健康。生产 readiness 应另外检查 registry、policy 和依赖。

### `worker_heartbeat(worker_id, heartbeat, registry)`

校验 path/body worker ID 一致后更新 registry。Worker 负责周期性上报；control plane 通过 TTL 摘除失联节点。

### `list_workers(registry)`

提供调试视图。它调用 `all_workers()`，因此不能把返回列表直接当作健康路由集合。

### `routing_decision(request, registry, router)`

读取健康 workers，调用 `IntelligentRouter.route()`，无可用路径时返回 HTTP 503。它是 dry-run control-plane API，目前不会把生成请求代理到被选中的 endpoint。

### `chat_completions(request, backend)`

现有的本地 Transformers data-plane 示例：校验不支持 streaming，调用 `backend.generate()`，再组装 OpenAI-compatible response。它与 distributed Router 尚未串成一个 remote gateway。

### `TransformersBackend._load()`

第一次生成时才加载 tokenizer/model，并用 lock 防止并发重复加载。这个后端适合教学或单进程验证，不具备 vLLM 的 continuous batching/paged KV serving。

### `TransformersBackend.generate()`

应用 chat template、tokenize、移动 tensor、执行 `model.generate()`，最后计算 prompt/completion token。全局 generation lock 会串行化请求，所以它不是 cloud serving 性能实现。

### `get_backend()`

构造并缓存本地 backend singleton，配置来自环境变量。

### `RequestMeasurement.tpot_seconds`

Prefix-cache lab 用 `(E2E - TTFT) / (completion_tokens - 1)` 估算 TPOT。它是 client-side approximation，不等于 vLLM server histogram 的精确 request decode 指标。

### `RequestMeasurement.cache_hit_rate`

用请求前后 `prefix_cache_hits / prefix_cache_queries` counter 差值计算 token-level hit rate。

### `parse_prometheus_metrics(text)`

解析 `/metrics` 文本，兼容 Prometheus Counter 的 `_total` 后缀，并按 label 汇总目标指标。

### `metric_delta(before, after)`

计算请求前后 counter 差值。在共享实例有并发流量时差值会被其他请求污染，所以实验应使用隔离 worker。

### `build_repository_prefix(target_characters)`

生成稳定、可重复的长代码上下文，确保两次实验请求共享大量 token prefix。真正 token 数以 vLLM usage 为准。

### `VLLMPrefixCacheExperiment.scrape_metrics()`

从 vLLM OpenAI-compatible server 的 `/metrics` 抓取 Router/实验需要的指标。

### `VLLMPrefixCacheExperiment.run_request()`

发送 SSE chat request，记录首个生成 token 时间、结束时间、usage，并把 metrics 前后差值封装成 `RequestMeasurement`。

### `print_report()` / `main()`

CLI orchestration：短请求 warm-up、cold-prefix request、shared-prefix request、打印 TTFT/TPOT/cache token 对比。

## 8. API

### 注册或更新 worker

```http
PUT /v1/workers/{worker_id}/heartbeat
```

### 查看 registry

```http
GET /v1/workers
```

### Dry-run 路由决策

```http
POST /v1/routing/decisions
```

返回所有候选的 matched tokens、new prefill tokens、queue/prefill/TTFT 估算、routing cost 和最终选择原因。

## 9. 已知边界

- 当前 endpoint 只做决策，尚未代理/stream 到远程 vLLM。
- Registry 是单进程内存状态，不是高可用服务。
- Fingerprint 不是 vLLM block hash。
- Queue 和 throughput 由 worker 上报，尚未实现 EWMA estimator。
- Intent 和 complexity 是规则 baseline，未使用训练数据。
- 没有 tenant quota、auth、rate limit、circuit breaker 或 retry budget。
- 没有 coding-agent tool executor、sandbox 和 durable task state。

这些边界是下一阶段的实现清单，而不是可以直接上线的声明。

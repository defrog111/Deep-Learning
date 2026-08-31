# Cloud LLM Inference 与 Coding Agent 学习路线

这份路线面向“懂 Transformer/LLM 基础，但刚进入云端推理岗位”的工程师。目标不是只会启动模型，而是能设计、度量、上线和解释一个多租户 coding-agent 推理系统。

## 1. 岗位能力地图

### 1.1 模型与推理基础

必须能从一次请求的生命周期解释性能：

1. Tokenization：文本变成 token。
2. Prefill：并行处理整个输入，产生首批 KV cache；主要影响 TTFT。
3. Decode：每步生成一个 token 并读写 KV cache；主要影响 TPOT/ITL。
4. Sampling：temperature、top-p、停止条件。

重点指标：

- TTFT（time to first token）：用户多久看到第一个 token。
- TPOT/ITL（time per output token/inter-token latency）：流式输出是否顺滑。
- End-to-end latency：请求总时间。
- Throughput：每秒完成的 token 或请求。
- Goodput：满足 SLO 的有效吞吐，而不只是实验室最高吞吐。
- KV-cache hit rate：重复前缀有多少无需重新 prefill。

KV cache 的近似空间复杂度：

```text
bytes ~= 2 * layers * kv_heads * head_dim * sequence_length * batch * bytes_per_element
```

前面的 `2` 代表 K 和 V。GQA/MQA 通过减少 `kv_heads` 显著降低缓存大小。你应该能用这个公式估算一台 GPU 能容纳多少活跃会话。

### 1.2 Serving 系统

建议依次掌握：

- Continuous batching 与请求调度。
- Paged KV cache 与内存碎片。
- Tensor/pipeline/data/expert parallelism。
- Quantization、speculative decoding、chunked prefill。
- Admission control、backpressure、超时、取消、重试。
- 多租户 quota、优先级、公平性与 noisy neighbor。
- 滚动升级、模型 warm-up、故障摘除和容量规划。

### 1.3 Cloud 思维

把系统明确拆成两部分：

- Control plane：模型目录、路由策略、worker registry、部署策略、配置与实验。
- Data plane：gateway、推理 worker、batch scheduler、KV cache、流式响应。

线上设计必须回答：

- 哪个租户能调用哪个模型？
- 高峰期先拒绝、排队、降级还是换模型？
- 节点失联多久摘除？在途请求如何处理？
- 重试会不会导致重复工具调用或重复计费？
- 如何按 tenant/model/route reason 观测延迟、成本和质量？
- 路由策略如何 canary、回滚、复现与审计？

### 1.4 Coding Agent

一个 agent 不是“多聊几轮”，而是受控状态机：

```text
user goal -> plan -> tool call -> observation -> state update -> next action -> verify -> answer
```

需要掌握：

- 工具 schema 与输入验证。
- repo search、文件读写、terminal、测试、浏览器等工具。
- sandbox、权限、审批边界和 secret isolation。
- 会话状态、checkpoint、compaction 与可恢复执行。
- 最大步数、token/时间/成本预算和停止条件。
- 幂等性：重试不能重复执行有副作用的操作。
- Eval：任务成功率、补丁正确率、测试通过率、无效工具调用、安全违规。

## 2. 老板邮件中的三个主题

### 2.1 Intelligent routing by task complexity

问题：所有任务都发给最大模型，质量可能高但 GPU 成本和排队时间不可接受；全发给小模型则复杂 coding task 容易失败。

学习顺序：

1. 先做可解释规则基线，例如代码、调试、多步骤、工具使用、上下文长度。
2. 收集真实任务和成功标签。
3. 用轻量 classifier 预测“最小可成功模型层级”。
4. 用 shadow traffic 比较候选路由，不影响用户。
5. 以质量约束下的成本/延迟最小化为目标做在线实验。

不要只预测“任务难不难”；真正目标是：在给定 SLO 和质量门槛下，哪个模型配置最合适。

### 2.2 KV-cache-aware routing

相同 system prompt、repo context 或历史对话落在已有前缀缓存的 worker 上，可以避免重复 prefill。但缓存亲和性不能压倒一切：缓存节点若已拥塞，排队损失可能大于 prefill 收益。

路由器至少要看到：

- 稳定前缀 fingerprint。
- worker 上缓存了哪些前缀及可复用 token 数。
- 缓存剩余容量和淘汰概率。
- queue depth、running requests、TTFT。
- 模型、tokenizer、adapter 和 cache layout 是否兼容。

生产版应估算“节省的 prefill 时间”，而不只是使用布尔 `cache_hit`。

### 2.3 Intent drift

coding 会话可能从“修 parser”切换到“设计数据库”。旧上下文此时会：

- 浪费 KV cache 和 context window。
- 把模型锚定在过期目标上。
- 让旧检索结果或旧计划污染新任务。

可观测信号包括显式换题词、embedding 距离、文件/语言/工具集合变化、计划目标变化和用户纠正。漂移高时先创建新 task state、只保留稳定 system contract、重新检索 repo context，再对新的有效 prefix 查询 cache。不能仅仅降低旧 cache affinity，却仍把无关历史发给模型。

## 3. 本仓库的学习实验

当前第一阶段实现：

- `analyze_task`：可解释的复杂度基线。
- `intent_drift`：显式换题 + 词项相似度基线。
- `prefix_fingerprint`：稳定会话前缀标识。
- `WorkerRegistry`：heartbeat 与过期摘除。
- `IntelligentRouter`：综合模型层级、负载、TTFT 和 KV cache。
- `/v1/routing/decisions`：返回完整候选分数与选择原因。

推荐练习：

1. 调用 heartbeat 注册 3 个不同 tier 的 GPU worker。
2. 分别提交简单问答和复杂 coding task，观察 `required_tier`。
3. 把同一个 `prefix_fingerprint` 和 `cached_tokens` 上报给某 worker，观察节省的 prefill。
4. 增加该 worker 的 `estimated_queue_ms`，找出排队超过多少时冷节点胜出。
5. 在最后一轮加入“new task/新任务”，观察 context action 变成 `reset_to_system`。

## 4. 迭代路线

### Phase 1：可解释 router（当前）

完成离线纯函数、worker registry、控制面 API 和单元测试。

### Phase 2：真实 GPU data plane

- 用 vLLM 或 SGLang 启动 Kimi worker。
- gateway 把路由决策转发给 worker，并支持 SSE streaming。
- 从引擎采集真实 queue、KV blocks、TTFT/TPOT。
- 加 Prometheus metrics 和 OpenTelemetry trace。

### Phase 3：Coding agent MVP

- 建立 agent loop 和 repo/terminal/test 工具。
- 每个工具声明权限和幂等性。
- 保存 task state，支持超时恢复。
- 建立 50～100 个可重复 coding eval tasks。

### Phase 4：数据驱动路由

- 记录 task signals、route、model config、cache reuse、SLO 和 eval outcome。
- 训练复杂度/成功率预测器，先 shadow 再 canary。
- 用 quality-constrained cost optimization 替换固定权重。
- 针对意图漂移训练分类器并评估 false reset/false carry-over。

### Phase 5：生产化

- Kubernetes 部署、GPU topology-aware scheduling、autoscaling。
- 多租户鉴权、quota、审计、成本归因。
- 故障注入、容量测试、灰度发布与自动回滚。
- 路由策略版本化，保证一次线上决策可复现。

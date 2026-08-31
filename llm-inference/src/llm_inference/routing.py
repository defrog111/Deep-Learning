from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .schemas import (
    CandidateScore,
    ContextAction,
    Message,
    ModelTier,
    RoutingDecision,
    TaskSignals,
    WorkerHeartbeat,
)

_TIER_RANK: Dict[ModelTier, int] = {"fast": 0, "balanced": 1, "reasoning": 2}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*|[\u4e00-\u9fff]")
_CODE_RE = re.compile(
    r"```|\b(?:def|class|function|import|SELECT|FROM|pytest|stack trace|exception)\b",
    re.IGNORECASE,
)
_REASONING_RE = re.compile(
    r"\b(?:analy[sz]e|architecture|debug|optimi[sz]e|trade-?off|root cause|reason)\b"
    r"|分析|架构|调试|优化|权衡|根因|推理",
    re.IGNORECASE,
)
_MULTISTEP_RE = re.compile(
    r"\b(?:first|then|finally|step|plan|implement|test|deploy)\b"
    r"|首先|然后|最后|步骤|计划|实现|测试|部署",
    re.IGNORECASE,
)
_TOOL_RE = re.compile(
    r"\b(?:tool|terminal|browser|database|api|repository|file|test suite)\b"
    r"|工具|终端|浏览器|数据库|接口|仓库|文件|测试套件",
    re.IGNORECASE,
)
_INTENT_SHIFT_RE = re.compile(
    r"\b(?:instead|unrelated|new task|different topic|forget that|start over)\b"
    r"|换个|另外一件|新任务|不同话题|忽略之前|不要那个|重新开始",
    re.IGNORECASE,
)


class NoRouteAvailable(RuntimeError):
    """Raised when no live worker can admit the request."""


@dataclass(frozen=True)
class _WorkerRecord:
    heartbeat: WorkerHeartbeat
    updated_at: float


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Cheap control-plane estimate; exact admission should use the model tokenizer."""

    characters = sum(len(message.content) for message in messages)
    return max(1, math.ceil(characters / 4))


def _hash_messages(messages: Sequence[Message]) -> str:
    payload = [message.model_dump(mode="json") for message in messages]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _leading_system_messages(messages: Sequence[Message]) -> List[Message]:
    system_messages: List[Message] = []
    for message in messages:
        if message.role != "system":
            break
        system_messages.append(message)
    return system_messages


def reusable_prefix(
    messages: Sequence[Message], drift_score: float
) -> tuple[List[Message], ContextAction]:
    """Choose history reuse before computing the KV fingerprint.

    High intent drift starts a new task from the stable system contract. Otherwise,
    all messages before the newest user turn form the reusable coding-session prefix.
    """

    if not messages:
        raise ValueError("messages must not be empty")
    if drift_score >= 0.75:
        return _leading_system_messages(messages), "reset_to_system"
    prefix = list(messages[:-1]) if messages[-1].role == "user" else list(messages)
    return prefix, "reuse_history"


def prefix_fingerprint(messages: Sequence[Message], drift_score: float = 0.0) -> str:
    """Hash the prefix that the gateway should actually reuse."""

    prefix, _ = reusable_prefix(messages, drift_score)
    return _hash_messages(prefix)


def _terms(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def intent_drift(messages: Sequence[Message]) -> float:
    """Explainable baseline for a per-session coding-goal change."""

    user_messages = [message.content for message in messages if message.role == "user"]
    if len(user_messages) < 2:
        return 0.0

    previous, current = user_messages[-2:]
    if _INTENT_SHIFT_RE.search(current):
        return 0.95

    previous_terms, current_terms = _terms(previous), _terms(current)
    if not previous_terms or not current_terms:
        return 0.5
    overlap = len(previous_terms & current_terms)
    similarity = overlap / math.sqrt(len(previous_terms) * len(current_terms))
    drift = 1.0 - similarity

    # Short commands often continue the current task despite low lexical overlap.
    if len(current_terms) <= 5:
        drift = min(drift, 0.35)
    return round(max(0.0, min(1.0, drift)), 3)


def analyze_task(
    messages: Sequence[Message],
    prompt_tokens: Optional[int] = None,
    reusable_prefix_tokens: Optional[int] = None,
) -> TaskSignals:
    """Extract task complexity, intent, and reusable-context signals."""

    if not messages:
        raise ValueError("messages must not be empty")
    text = "\n".join(message.content for message in messages)
    estimated = prompt_tokens or estimate_tokens(messages)
    score = min(0.30, estimated / 20_000)
    features: List[str] = []

    for pattern, weight, name in (
        (_CODE_RE, 0.22, "code"),
        (_REASONING_RE, 0.20, "reasoning"),
        (_MULTISTEP_RE, 0.18, "multi_step"),
        (_TOOL_RE, 0.14, "tool_use"),
    ):
        if pattern.search(text):
            score += weight
            features.append(name)

    if len(messages) >= 8:
        score += 0.10
        features.append("long_conversation")
    score = round(min(1.0, score), 3)

    if score < 0.30:
        tier: ModelTier = "fast"
    elif score < 0.62:
        tier = "balanced"
    else:
        tier = "reasoning"

    drift = intent_drift(messages)
    prefix, context_action = reusable_prefix(messages, drift)
    if context_action == "reset_to_system":
        features.append("intent_shift")
    prefix_tokens = estimate_tokens(prefix) if prefix else 0
    if context_action == "reuse_history" and reusable_prefix_tokens is not None:
        prefix_tokens = min(reusable_prefix_tokens, estimated)
    return TaskSignals(
        complexity_score=score,
        required_tier=tier,
        estimated_prompt_tokens=estimated,
        reusable_prefix_tokens=prefix_tokens,
        intent_drift_score=drift,
        context_action=context_action,
        prefix_fingerprint=_hash_messages(prefix),
        features=features,
    )


class WorkerRegistry:
    """Thread-safe in-memory registry with heartbeat expiry."""

    def __init__(
        self,
        heartbeat_ttl_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.heartbeat_ttl_seconds = heartbeat_ttl_seconds
        self._clock = clock
        self._workers: Dict[str, _WorkerRecord] = {}
        self._lock = Lock()

    def update(self, heartbeat: WorkerHeartbeat) -> None:
        with self._lock:
            self._workers[heartbeat.worker_id] = _WorkerRecord(
                heartbeat=heartbeat,
                updated_at=self._clock(),
            )

    def healthy_workers(self) -> List[WorkerHeartbeat]:
        now = self._clock()
        with self._lock:
            return [
                record.heartbeat
                for record in self._workers.values()
                if record.heartbeat.healthy
                and now - record.updated_at <= self.heartbeat_ttl_seconds
            ]

    def all_workers(self) -> List[WorkerHeartbeat]:
        with self._lock:
            return [record.heartbeat for record in self._workers.values()]


class IntelligentRouter:
    """Route by minimum quality tier and predicted cache-aware TTFT."""

    def __init__(self, overserve_penalty_ms_per_tier: float = 75.0) -> None:
        self.overserve_penalty_ms_per_tier = overserve_penalty_ms_per_tier

    def route(
        self,
        messages: Sequence[Message],
        workers: Iterable[WorkerHeartbeat],
        prompt_tokens: Optional[int] = None,
        reusable_prefix_tokens: Optional[int] = None,
        required_context_tokens: Optional[int] = None,
    ) -> RoutingDecision:
        signals = analyze_task(
            messages,
            prompt_tokens=prompt_tokens,
            reusable_prefix_tokens=reusable_prefix_tokens,
        )
        required_context = required_context_tokens or signals.estimated_prompt_tokens
        eligible = [
            worker
            for worker in workers
            if worker.healthy and worker.max_context_tokens >= required_context
        ]
        if not eligible:
            raise NoRouteAvailable("no healthy worker has enough context capacity")

        required_rank = _TIER_RANK[signals.required_tier]
        capable = [
            worker
            for worker in eligible
            if _TIER_RANK[worker.model_tier] >= required_rank
        ]
        degraded = not capable
        if capable:
            selection_pool = capable
        else:
            best_available_rank = max(_TIER_RANK[worker.model_tier] for worker in eligible)
            selection_pool = [
                worker
                for worker in eligible
                if _TIER_RANK[worker.model_tier] == best_available_rank
            ]

        scored = [self._estimate(worker, signals) for worker in eligible]
        scored.sort(key=lambda candidate: (candidate.routing_cost_ms, candidate.worker_id))
        selectable_ids = {worker.worker_id for worker in selection_pool}
        winner_score = next(
            candidate for candidate in scored if candidate.worker_id in selectable_ids
        )
        winner = next(
            worker for worker in eligible if worker.worker_id == winner_score.worker_id
        )
        return RoutingDecision(
            worker_id=winner.worker_id,
            endpoint=winner.endpoint,
            model_id=winner.model_id,
            signals=signals,
            candidates=scored,
            degraded=degraded,
            reasons=[
                f"task requires {signals.required_tier} tier",
                f"context action={signals.context_action}",
                "degraded: no capable tier available"
                if degraded
                else "quality tier satisfied",
                *winner_score.reasons,
            ],
        )

    def _estimate(
        self, worker: WorkerHeartbeat, signals: TaskSignals
    ) -> CandidateScore:
        cache_by_fingerprint = {
            entry.prefix_fingerprint: entry.cached_tokens
            for entry in worker.cache_entries
        }
        advertised_tokens = cache_by_fingerprint.get(signals.prefix_fingerprint, 0)
        matched_tokens = min(
            advertised_tokens,
            signals.reusable_prefix_tokens,
            signals.estimated_prompt_tokens,
        )
        new_prefill_tokens = signals.estimated_prompt_tokens - matched_tokens
        estimated_prefill_ms = (
            new_prefill_tokens / worker.prefill_tokens_per_second * 1_000.0
        )
        estimated_ttft_ms = (
            worker.network_rtt_ms
            + worker.estimated_queue_ms
            + estimated_prefill_ms
            + worker.first_token_decode_ms
        )
        tier_gap = max(0, _TIER_RANK[worker.model_tier] - _TIER_RANK[signals.required_tier])
        overserve_penalty = tier_gap * self.overserve_penalty_ms_per_tier
        routing_cost_ms = estimated_ttft_ms + overserve_penalty
        utilization = min(1.0, worker.running_requests / worker.max_concurrency)

        return CandidateScore(
            worker_id=worker.worker_id,
            model_id=worker.model_id,
            matched_prefix_tokens=matched_tokens,
            new_prefill_tokens=new_prefill_tokens,
            estimated_queue_ms=round(worker.estimated_queue_ms, 3),
            estimated_prefill_ms=round(estimated_prefill_ms, 3),
            estimated_ttft_ms=round(estimated_ttft_ms, 3),
            routing_cost_ms=round(routing_cost_ms, 3),
            utilization=round(utilization, 3),
            cache_hit=matched_tokens > 0,
            reasons=[
                f"tier={worker.model_tier}",
                f"matched_prefix_tokens={matched_tokens}",
                f"new_prefill_tokens={new_prefill_tokens}",
                f"queue_ms={worker.estimated_queue_ms:.1f}",
                f"kv_usage={worker.kv_cache_usage_percent:.1%}",
                f"overserve_penalty_ms={overserve_penalty:.1f}",
            ],
        )


_registry = WorkerRegistry()
_router = IntelligentRouter()


def get_worker_registry() -> WorkerRegistry:
    return _registry


def get_router() -> IntelligentRouter:
    return _router

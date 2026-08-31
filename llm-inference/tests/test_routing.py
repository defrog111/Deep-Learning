from llm_inference.routing import (
    IntelligentRouter,
    WorkerRegistry,
    analyze_task,
    intent_drift,
    prefix_fingerprint,
)
from llm_inference.schemas import Message, PrefixCacheEntry, WorkerHeartbeat


def worker(
    worker_id: str,
    tier: str,
    *,
    cache_entries=None,
    running_requests: int = 0,
    estimated_queue_ms: float = 0.0,
    prefill_tokens_per_second: float = 10_000.0,
) -> WorkerHeartbeat:
    return WorkerHeartbeat(
        worker_id=worker_id,
        endpoint=f"http://{worker_id}:8000",
        model_id=f"kimi-{tier}",
        model_tier=tier,
        max_context_tokens=131_072,
        max_concurrency=8,
        running_requests=running_requests,
        estimated_queue_ms=estimated_queue_ms,
        prefill_tokens_per_second=prefill_tokens_per_second,
        cache_entries=cache_entries or [],
    )


def test_simple_question_uses_fast_tier() -> None:
    messages = [Message(role="user", content="What is a Python tuple?")]

    decision = IntelligentRouter().route(
        messages,
        [worker("fast-1", "fast"), worker("reasoning-1", "reasoning")],
    )

    assert decision.signals.required_tier == "fast"
    assert decision.worker_id == "fast-1"


def test_complex_coding_task_prefers_reasoning_tier() -> None:
    messages = [
        Message(
            role="user",
            content=(
                "Analyze the architecture, debug this exception, then implement and test "
                "a tool that reads repository files. ```python\ndef broken(): ...\n```"
            ),
        )
    ]

    decision = IntelligentRouter().route(
        messages,
        [worker("fast-1", "fast"), worker("reasoning-1", "reasoning")],
    )

    assert decision.signals.required_tier == "reasoning"
    assert decision.worker_id == "reasoning-1"
    assert {"code", "reasoning", "multi_step", "tool_use"}.issubset(
        decision.signals.features
    )


def test_quality_tier_is_a_constraint_when_capable_worker_exists() -> None:
    messages = [
        Message(
            role="user",
            content=(
                "Analyze the architecture, debug this exception, then implement and test "
                "a repository tool. ```python\nraise RuntimeError\n```"
            ),
        )
    ]

    decision = IntelligentRouter().route(
        messages,
        [
            worker("idle-fast", "fast"),
            worker(
                "busy-reasoning",
                "reasoning",
                running_requests=8,
                estimated_queue_ms=10_000,
            ),
        ],
    )

    assert decision.worker_id == "busy-reasoning"
    assert "quality tier satisfied" in decision.reasons


def test_kv_cache_savings_can_beat_a_moderate_queue() -> None:
    messages = [
        Message(role="system", content="You are a coding agent."),
        Message(role="user", content="Continue fixing the parser."),
    ]
    fingerprint = prefix_fingerprint(messages)

    decision = IntelligentRouter().route(
        messages,
        [
            worker(
                "warm-worker",
                "fast",
                cache_entries=[
                    PrefixCacheEntry(
                        prefix_fingerprint=fingerprint,
                        cached_tokens=9_500,
                    )
                ],
                estimated_queue_ms=200,
            ),
            worker("cold-worker", "fast"),
        ],
        prompt_tokens=10_000,
        reusable_prefix_tokens=9_500,
    )

    assert decision.worker_id == "warm-worker"
    assert decision.candidates[0].cache_hit is True
    assert decision.candidates[0].matched_prefix_tokens == 9_500
    assert decision.candidates[0].new_prefill_tokens == 500


def test_long_queue_can_outweigh_a_kv_cache_hit() -> None:
    messages = [
        Message(role="system", content="You are a coding agent."),
        Message(role="user", content="Continue fixing the parser."),
    ]
    fingerprint = prefix_fingerprint(messages)

    decision = IntelligentRouter().route(
        messages,
        [
            worker(
                "overloaded-warm-worker",
                "fast",
                cache_entries=[
                    PrefixCacheEntry(
                        prefix_fingerprint=fingerprint,
                        cached_tokens=9_500,
                    )
                ],
                estimated_queue_ms=1_200,
            ),
            worker("idle-cold-worker", "fast"),
        ],
        prompt_tokens=10_000,
        reusable_prefix_tokens=9_500,
    )

    assert decision.worker_id == "idle-cold-worker"


def test_explicit_intent_drift_reduces_old_cache_affinity() -> None:
    messages = [
        Message(role="system", content="You are a coding agent."),
        Message(role="user", content="Optimize the Python parser."),
        Message(role="assistant", content="I will profile it."),
        Message(role="user", content="New task: design an unrelated database schema instead."),
    ]
    drift = intent_drift(messages)
    fingerprint = prefix_fingerprint(messages, drift_score=drift)

    decision = IntelligentRouter().route(
        messages,
        [
            worker(
                "busy-warm-worker",
                "balanced",
                cache_entries=[
                    PrefixCacheEntry(
                        prefix_fingerprint=fingerprint,
                        cached_tokens=1_000,
                    )
                ],
                estimated_queue_ms=100,
            ),
            worker("idle-cold-worker", "balanced"),
        ],
    )

    assert decision.signals.intent_drift_score == 0.95
    assert decision.signals.context_action == "reset_to_system"
    assert decision.worker_id == "idle-cold-worker"


def test_stale_worker_is_removed_from_healthy_view() -> None:
    now = [0.0]
    registry = WorkerRegistry(heartbeat_ttl_seconds=10.0, clock=lambda: now[0])
    registry.update(worker("worker-1", "fast"))
    assert len(registry.healthy_workers()) == 1

    now[0] = 11.0
    assert registry.healthy_workers() == []


def test_analyze_task_has_stable_prefix_fingerprint() -> None:
    common = [
        Message(role="system", content="You are a coding agent."),
        Message(role="user", content="Inspect app.py"),
        Message(role="assistant", content="I found the bug."),
    ]
    first = analyze_task([*common, Message(role="user", content="Fix it")])
    second = analyze_task([*common, Message(role="user", content="Explain it")])

    assert first.prefix_fingerprint == second.prefix_fingerprint

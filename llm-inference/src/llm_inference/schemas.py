from __future__ import annotations

import time
import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    stream: bool = False


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage


ModelTier = Literal["fast", "balanced", "reasoning"]
ContextAction = Literal["reuse_history", "reset_to_system"]


class PrefixCacheEntry(BaseModel):
    """The largest currently available token prefix for one fingerprint."""

    prefix_fingerprint: str = Field(min_length=1)
    cached_tokens: int = Field(gt=0)


class WorkerHeartbeat(BaseModel):
    """A vLLM replica's capacity, latency estimate, and current KV directory."""

    worker_id: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_tier: ModelTier
    max_context_tokens: int = Field(gt=0)
    max_concurrency: int = Field(gt=0)
    running_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    estimated_queue_ms: float = Field(default=0.0, ge=0.0)
    prefill_tokens_per_second: float = Field(default=10_000.0, gt=0.0)
    first_token_decode_ms: float = Field(default=10.0, ge=0.0)
    network_rtt_ms: float = Field(default=1.0, ge=0.0)
    kv_cache_usage_percent: float = Field(default=0.0, ge=0.0, le=1.0)
    cache_entries: List[PrefixCacheEntry] = Field(default_factory=list)
    healthy: bool = True


class RoutingRequest(BaseModel):
    messages: List[Message] = Field(min_length=1)
    session_id: Optional[str] = None
    prompt_tokens: Optional[int] = Field(default=None, gt=0)
    reusable_prefix_tokens: Optional[int] = Field(default=None, ge=0)
    required_context_tokens: Optional[int] = Field(default=None, gt=0)


class TaskSignals(BaseModel):
    complexity_score: float = Field(ge=0.0, le=1.0)
    required_tier: ModelTier
    estimated_prompt_tokens: int = Field(ge=1)
    reusable_prefix_tokens: int = Field(ge=0)
    intent_drift_score: float = Field(ge=0.0, le=1.0)
    context_action: ContextAction
    prefix_fingerprint: str
    features: List[str] = Field(default_factory=list)


class CandidateScore(BaseModel):
    worker_id: str
    model_id: str
    matched_prefix_tokens: int = Field(ge=0)
    new_prefill_tokens: int = Field(ge=0)
    estimated_queue_ms: float = Field(ge=0.0)
    estimated_prefill_ms: float = Field(ge=0.0)
    estimated_ttft_ms: float = Field(ge=0.0)
    routing_cost_ms: float = Field(ge=0.0)
    utilization: float = Field(ge=0.0, le=1.0)
    cache_hit: bool
    reasons: List[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    worker_id: str
    endpoint: str
    model_id: str
    signals: TaskSignals
    candidates: List[CandidateScore]
    degraded: bool = False
    reasons: List[str]

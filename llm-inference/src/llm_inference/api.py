from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from .backend import TransformersBackend, get_backend
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    RoutingDecision,
    RoutingRequest,
    Usage,
    WorkerHeartbeat,
)
from .routing import (
    IntelligentRouter,
    NoRouteAvailable,
    WorkerRegistry,
    get_router,
    get_worker_registry,
)

app = FastAPI(title="LLM Inference API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.put("/v1/workers/{worker_id}/heartbeat", response_model=WorkerHeartbeat)
def worker_heartbeat(
    worker_id: str,
    heartbeat: WorkerHeartbeat,
    registry: WorkerRegistry = Depends(get_worker_registry),
) -> WorkerHeartbeat:
    if worker_id != heartbeat.worker_id:
        raise HTTPException(status_code=400, detail="worker_id path/body mismatch")
    registry.update(heartbeat)
    return heartbeat


@app.get("/v1/workers", response_model=list[WorkerHeartbeat])
def list_workers(
    registry: WorkerRegistry = Depends(get_worker_registry),
) -> list[WorkerHeartbeat]:
    return registry.all_workers()


@app.post("/v1/routing/decisions", response_model=RoutingDecision)
def routing_decision(
    request: RoutingRequest,
    registry: WorkerRegistry = Depends(get_worker_registry),
    router: IntelligentRouter = Depends(get_router),
) -> RoutingDecision:
    try:
        return router.route(
            messages=request.messages,
            workers=registry.healthy_workers(),
            prompt_tokens=request.prompt_tokens,
            reusable_prefix_tokens=request.reusable_prefix_tokens,
            required_context_tokens=request.required_context_tokens,
        )
    except NoRouteAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(
    request: ChatCompletionRequest,
    backend: TransformersBackend = Depends(get_backend),
) -> ChatCompletionResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented yet")

    result = backend.generate(
        messages=request.messages,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return ChatCompletionResponse(
        model=request.model or backend.model_id,
        choices=[Choice(message=Message(role="assistant", content=result.text))],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from .backend import TransformersBackend, get_backend
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

app = FastAPI(title="LLM Inference API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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


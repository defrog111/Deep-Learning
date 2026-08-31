from fastapi.testclient import TestClient

from llm_inference.api import app
from llm_inference.backend import GenerationResult, get_backend
from llm_inference.routing import WorkerRegistry, get_worker_registry


class FakeBackend:
    model_id = "test-model"

    def generate(self, messages, max_new_tokens, temperature, top_p):
        assert messages[-1].content == "Hello"
        return GenerationResult(text="Hi!", prompt_tokens=2, completion_tokens=1)


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_completion_shape() -> None:
    app.dependency_overrides[get_backend] = lambda: FakeBackend()
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "test-model"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hi!"}
    assert body["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }


def test_streaming_is_rejected_explicitly() -> None:
    app.dependency_overrides[get_backend] = lambda: FakeBackend()
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 400


def test_worker_heartbeat_and_routing_api() -> None:
    registry = WorkerRegistry()
    app.dependency_overrides[get_worker_registry] = lambda: registry
    client = TestClient(app)
    heartbeat = {
        "worker_id": "gpu-node-1",
        "endpoint": "http://gpu-node-1:8000",
        "model_id": "kimi-reasoning",
        "model_tier": "reasoning",
        "max_context_tokens": 131072,
        "max_concurrency": 16,
        "estimated_queue_ms": 25,
        "prefill_tokens_per_second": 20000,
    }
    try:
        registered = client.put("/v1/workers/gpu-node-1/heartbeat", json=heartbeat)
        decision = client.post(
            "/v1/routing/decisions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Analyze the architecture, debug this exception, then implement "
                            "and test a tool that reads repository files."
                        ),
                    }
                ],
                "prompt_tokens": 2000,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert registered.status_code == 200
    assert decision.status_code == 200
    body = decision.json()
    assert body["worker_id"] == "gpu-node-1"
    assert body["signals"]["required_tier"] == "reasoning"
    assert body["candidates"][0]["estimated_ttft_ms"] > 0


def test_routing_returns_503_without_workers() -> None:
    app.dependency_overrides[get_worker_registry] = lambda: WorkerRegistry()
    try:
        response = TestClient(app).post(
            "/v1/routing/decisions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503

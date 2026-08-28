from fastapi.testclient import TestClient

from llm_inference.api import app
from llm_inference.backend import GenerationResult, get_backend


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


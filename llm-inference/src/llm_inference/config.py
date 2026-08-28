from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"
    dtype: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    max_new_tokens: int = 256

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_id=os.getenv("LLM_MODEL_ID", cls.model_id),
            device=os.getenv("LLM_DEVICE", cls.device),
            dtype=os.getenv("LLM_DTYPE", cls.dtype),
            host=os.getenv("LLM_HOST", cls.host),
            port=int(os.getenv("LLM_PORT", str(cls.port))),
            max_new_tokens=int(
                os.getenv("LLM_MAX_NEW_TOKENS", str(cls.max_new_tokens))
            ),
        )


from __future__ import annotations

import argparse

from .backend import TransformersBackend
from .config import Settings
from .schemas import Message


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Run one local LLM inference request")
    parser.add_argument("prompt")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--model", default=settings.model_id)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--dtype", default=settings.dtype)
    parser.add_argument("--max-new-tokens", type=int, default=settings.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    backend = TransformersBackend(args.model, args.device, args.dtype)
    result = backend.generate(
        messages=[
            Message(role="system", content=args.system),
            Message(role="user", content=args.prompt),
        ],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(result.text)


if __name__ == "__main__":
    main()


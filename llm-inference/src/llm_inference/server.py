from __future__ import annotations

from .config import Settings


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run("llm_inference.api:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()


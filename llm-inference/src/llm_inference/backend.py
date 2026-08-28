from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional

from .schemas import Message


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class TransformersBackend:
    """Lazily loads a Hugging Face causal language model on first use."""

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._tokenizer = None
        self._model = None
        self._load_lock = Lock()
        self._generation_lock = Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype_map: Dict[str, object] = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            model_kwargs = {"torch_dtype": dtype_map.get(self.dtype, "auto")}
            if self.device == "auto":
                model_kwargs["device_map"] = "auto"

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            if self.device != "auto":
                model = model.to(self.device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model

    def generate(
        self,
        messages: List[Message],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationResult:
        self._load()
        assert self._tokenizer is not None and self._model is not None

        tokenizer = self._tokenizer
        model = self._model
        rendered = tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {name: value.to(model_device) for name, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p)

        import torch

        with self._generation_lock, torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
        generated_ids = output[0, prompt_tokens:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(generated_ids.shape[-1]),
        )


_backend: Optional[TransformersBackend] = None
_backend_lock = Lock()


def get_backend() -> TransformersBackend:
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                from .config import Settings

                settings = Settings.from_env()
                _backend = TransformersBackend(
                    settings.model_id, settings.device, settings.dtype
                )
    return _backend


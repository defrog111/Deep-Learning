"""例 10（推理）：加载训练 checkpoint，单独执行文本生成。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tiny_causal_lm_model import CharTokenizer, TinyCausalLM


def infer(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise SystemExit(
            "找不到 checkpoint，请先运行：python 10_tiny_causal_lm_train.py"
        )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    tokenizer = CharTokenizer(checkpoint["tokens"])
    model = TinyCausalLM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base / "artifacts" / "tiny_causal_lm.pt",
    )
    parser.add_argument("--prompt", default="transformer ")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())

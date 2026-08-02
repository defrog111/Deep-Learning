"""例 19（推理）：加载原生多层 causal LM，执行自回归生成。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from builtin_causal_lm_model import BuiltinCausalLM, CharTokenizer


def main() -> None:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base / "artifacts" / "builtin_causal_lm.pt",
    )
    parser.add_argument("--prompt", default="transformer ")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 使用 greedy；大于 0 时按温度采样。",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = CharTokenizer(checkpoint["tokens"])
    model = BuiltinCausalLM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()

"""例 10：只负责训练 Tiny Causal LM 并保存 checkpoint。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from tiny_causal_lm_model import CharTokenizer, TinyCausalLM


TRAIN_TEXT = (
    "transformer uses attention. "
    "attention connects tokens. "
    "tokens carry meaning. "
) * 20


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    tokenizer = CharTokenizer.from_text(TRAIN_TEXT)
    all_ids = torch.tensor(tokenizer.encode(TRAIN_TEXT), dtype=torch.long)
    model_config = {
        "vocab_size": len(tokenizer.tokens),
        "d_model": 32,
        "num_heads": 4,
        "num_layers": 2,
        "block_size": args.block_size,
    }
    model = TinyCausalLM(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)

    model.train()
    for step in range(1, args.steps + 1):
        starts = torch.randint(
            0,
            len(all_ids) - args.block_size - 1,
            (args.batch_size,),
            generator=generator,
        )
        sequences = torch.stack(
            [all_ids[start : start + args.block_size + 1] for start in starts]
        )
        inputs, targets = sequences[:, :-1], sequences[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, model_config["vocab_size"]),
            targets.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"step={step:03d} loss={loss.item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "tokens": tokenizer.tokens,
        },
        args.output,
    )
    print("checkpoint:", args.output)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "artifacts" / "tiny_causal_lm.pt",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

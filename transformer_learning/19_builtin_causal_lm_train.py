"""例 19（训练）：训练 4 层 PyTorch 原生 decoder-only Transformer LM。

训练使用 teacher forcing：输入 x[:, :-1]，目标是右移一位的 x[:, 1:]。
本文件只负责训练和保存 checkpoint；生成请运行对应 inference 文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from builtin_causal_lm_model import BuiltinCausalLM, CharTokenizer


TRAIN_TEXT = (
    "transformer uses attention. "
    "attention connects tokens. "
    "tokens carry meaning. "
    "a language model predicts the next token. "
) * 30


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    tokenizer = CharTokenizer.from_text(TRAIN_TEXT)
    all_ids = torch.tensor(tokenizer.encode(TRAIN_TEXT), dtype=torch.long)
    model_config = {
        "vocab_size": len(tokenizer.tokens),
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "block_size": args.block_size,
        "dropout": args.dropout,
    }
    model = BuiltinCausalLM(**model_config)
    print(f"原生 causal Transformer 层数={len(model.transformer.layers)}")
    print(f"训练字符数={len(all_ids)}，词表大小={len(tokenizer.tokens)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )
    rng = torch.Generator().manual_seed(args.seed)
    model.train()

    for step in range(1, args.steps + 1):
        starts = torch.randint(
            low=0,
            high=len(all_ids) - args.block_size - 1,
            size=(args.batch_size,),
            generator=rng,
        )
        sequences = torch.stack(
            [all_ids[start : start + args.block_size + 1] for start in starts]
        )
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, model_config["vocab_size"]),
            targets.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    print(f"checkpoint={args.output}")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "artifacts" / "builtin_causal_lm.pt",
    )
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

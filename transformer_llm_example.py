"""
Train and run inference with a tiny decoder-only Transformer language model.

This is a from-scratch PyTorch example meant to show how Transformer LLM
training works: token embedding, positional embedding, causal self-attention,
next-token loss, checkpoint saving, and autoregressive inference.

Install:
    pip install -r requirements.txt

Train on a small subset for a quick laptop check:
    python transformer_llm_example.py train --max_train_samples 2000 --epochs 1

Train on the full WikiText-103 train split, which is roughly 100M tokens:
    python transformer_llm_example.py train --max_train_samples 0 --epochs 1

Generate text from a saved checkpoint:
    python transformer_llm_example.py infer --prompt "The history of machine learning"
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PAD_TOKEN = 256
VOCAB_SIZE = 257
DEFAULT_CHECKPOINT = "transformer_llm_checkpoint.pt"


class ByteTokenizer:
    """Minimal byte-level tokenizer: every UTF-8 byte is a token."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, tokens: list[int]) -> str:
        byte_values = bytes(token for token in tokens if 0 <= token < 256)
        return byte_values.decode("utf-8", errors="ignore")


class TextChunkDataset(Dataset):
    def __init__(self, tokens: list[int], block_size: int):
        if len(tokens) <= block_size:
            raise ValueError("Not enough tokens. Lower --block_size or load more data.")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int):
        chunk = self.tokens[index : index + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.block_size}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.block_size :]
            logits = self(context)[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_wikitext_tokens(max_train_samples: int) -> list[int]:
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    if max_train_samples > 0:
        dataset = dataset.select(range(min(max_train_samples, len(dataset))))

    tokenizer = ByteTokenizer()
    tokens: list[int] = []
    for row in tqdm(dataset, desc="Tokenizing WikiText"):
        text = row["text"].strip()
        if text:
            tokens.extend(tokenizer.encode(text + "\n"))
    return tokens


def build_model(args) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )


def save_checkpoint(path: str, model: nn.Module, args) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_args": {
                "block_size": args.block_size,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            },
        },
        path,
    )


def load_checkpoint(path: str, device: str) -> DecoderOnlyTransformer:
    checkpoint = torch.load(path, map_location=device)
    model_args = argparse.Namespace(**checkpoint["model_args"])
    model = build_model(model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def train(args) -> None:
    device = get_device()
    print(f"Device: {device}")
    tokens = load_wikitext_tokens(args.max_train_samples)
    print(f"Training tokens loaded: {len(tokens):,}")

    dataset = TextChunkDataset(tokens, block_size=args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    total_params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {total_params:,}")

    model.train()
    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, (x, y) in enumerate(progress, start=1):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                ignore_index=PAD_TOKEN,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            progress.set_postfix(loss=f"{loss.item():.4f}")
            if args.max_steps > 0 and step >= args.max_steps:
                break

    save_checkpoint(args.checkpoint, model, args)
    print(f"Saved checkpoint to: {args.checkpoint}")


def infer(args) -> None:
    device = get_device()
    tokenizer = ByteTokenizer()
    model = load_checkpoint(args.checkpoint, device)

    prompt_tokens = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny Transformer LLM example.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)

    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--max_train_samples", type=int, default=2000)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--max_steps", type=int, default=300)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--block_size", type=int, default=128)
    train_parser.add_argument("--d_model", type=int, default=128)
    train_parser.add_argument("--n_heads", type=int, default=4)
    train_parser.add_argument("--n_layers", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--learning_rate", type=float, default=3e-4)
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser("infer", parents=[common])
    infer_parser.add_argument("--prompt", default="The history of machine learning")
    infer_parser.add_argument("--max_new_tokens", type=int, default=120)
    infer_parser.add_argument("--temperature", type=float, default=0.8)
    infer_parser.add_argument("--top_k", type=int, default=50)
    infer_parser.set_defaults(func=infer)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

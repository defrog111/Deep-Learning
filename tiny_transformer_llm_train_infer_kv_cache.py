"""
Train and infer with a tiny decoder-only Transformer LLM, including KV cache.

This is a compact from-scratch PyTorch example of the common LLM pieces:
token embedding, positional embedding, stacked Transformer decoder blocks,
multi-head causal self-attention, MLP, LayerNorm, next-token training loss,
checkpoint save/load, normal autoregressive inference, and KV-cache inference.

Run:
    python tiny_transformer_llm_train_infer_kv_cache.py train
    python tiny_transformer_llm_train_infer_kv_cache.py sft
    python tiny_transformer_llm_train_infer_kv_cache.py sft --input_checkpoint tiny_transformer_llm_kv_cache_checkpoint.pt
    python tiny_transformer_llm_train_infer_kv_cache.py infer --prompt "Question: 2+2? Answer:" --use_kv_cache
"""

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(23)
random.seed(23)

DEFAULT_CHECKPOINT = "tiny_transformer_llm_kv_cache_checkpoint.pt"
DEFAULT_SFT_CHECKPOINT = "tiny_transformer_llm_sft_checkpoint.pt"
TRAIN_TEXT = """
Question: 2+2? Answer: 4.
Question: capital of France? Answer: Paris.
Question: opposite of hot? Answer: cold.
Question: 3+5? Answer: 8.
Question: color of the sky? Answer: blue.
Question: animal that says meow? Answer: cat.
""" * 80
SFT_EXAMPLES = [
    ("User: 2+2?\nAssistant:", " 4."),
    ("User: What is the capital of France?\nAssistant:", " Paris."),
    ("User: What is the opposite of hot?\nAssistant:", " cold."),
    ("User: 3+5?\nAssistant:", " 8."),
    ("User: What color is the sky?\nAssistant:", " blue."),
    ("User: What animal says meow?\nAssistant:", " cat."),
]
SFT_TEXT = "\n".join(prompt + answer for prompt, answer in SFT_EXAMPLES)
TOKENIZER_TEXT = TRAIN_TEXT + SFT_TEXT


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.tokens = ["<pad>", "<bos>", "<eos>"] + chars
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi[ch] for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        pieces = []
        for index in ids:
            token = self.itos[int(index)]
            if token not in ("<pad>", "<bos>", "<eos>"):
                pieces.append(token)
        return "".join(pieces)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index):
        chunk = self.token_ids[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, block_size):
        self.tokenizer = tokenizer
        self.examples = examples
        self.block_size = block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        prompt, answer = self.examples[index]
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        answer_ids = self.tokenizer.encode(answer, add_eos=True)
        full = torch.cat([prompt_ids, answer_ids])
        if len(full) > self.block_size + 1:
            raise ValueError("SFT example is longer than block_size")

        x = full[:-1]
        y = full[1:].clone()
        y[: len(prompt_ids) - 1] = -100

        pad_len = self.block_size - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), self.tokenizer.pad_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
        return x, y


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        total_len = k.size(2)
        past_len = total_len - seq_len
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        query_positions = torch.arange(past_len, total_len, device=x.device).unsqueeze(1)
        key_positions = torch.arange(total_len, device=x.device).unsqueeze(0)
        causal_mask = key_positions > query_positions
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        present_kv = (k, v) if use_cache else None
        return self.out_proj(y), present_kv


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, block_size=96, d_model=96, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, targets=None, past_kv=None, use_cache=False):
        batch_size, seq_len = input_ids.shape
        past_len = 0 if past_kv is None else past_kv[0][0].size(2)
        if past_len + seq_len > self.block_size:
            raise ValueError("Sequence plus KV cache exceeds block_size")

        positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)

        present_kv = [] if use_cache else None
        for layer_index, block in enumerate(self.blocks):
            layer_past = None if past_kv is None else past_kv[layer_index]
            x, layer_present = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                present_kv.append(layer_present)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, present_kv


def build_tokenizer():
    return CharTokenizer(TOKENIZER_TEXT)


def build_model(tokenizer, args):
    return TinyTransformerLM(
        vocab_size=len(tokenizer.tokens),
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )


def save_checkpoint(path, model, args):
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


def load_checkpoint(path):
    tokenizer = build_tokenizer()
    checkpoint = torch.load(path, map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["model_args"])
    model = build_model(tokenizer, model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, model_args


def train(args):
    tokenizer = build_tokenizer()
    token_ids = tokenizer.encode(TRAIN_TEXT, add_bos=True, add_eos=True)
    dataset = TextDataset(token_ids, args.block_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(tokenizer, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(
        f"Training tiny Transformer: layers={args.n_layers}, heads={args.n_heads}, "
        f"d_model={args.d_model}, vocab={len(tokenizer.tokens)}"
    )
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for x, y in loader:
            _, loss, _ = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 25 == 0:
                print(f"epoch={epoch + 1} step={step:03d} loss={loss.item():.4f}")
            if args.max_steps > 0 and step >= args.max_steps:
                save_checkpoint(args.checkpoint, model, args)
                print(f"Saved checkpoint to: {args.checkpoint}")
                return

    save_checkpoint(args.checkpoint, model, args)
    print(f"Saved checkpoint to: {args.checkpoint}")


def sft(args):
    tokenizer = build_tokenizer()
    if args.input_checkpoint:
        model, _, model_args = load_checkpoint(args.input_checkpoint)
        print(f"Loaded pretrained checkpoint from: {args.input_checkpoint}")
    else:
        model_args = args
        model = build_model(tokenizer, model_args)
        print("No input checkpoint given; SFT starts from a fresh tiny Transformer.")

    dataset = SFTDataset(tokenizer, SFT_EXAMPLES, model.block_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(
        f"SFT tiny Transformer: layers={model_args.n_layers}, heads={model_args.n_heads}, "
        f"d_model={model_args.d_model}, supervised_examples={len(SFT_EXAMPLES)}"
    )
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for x, y in loader:
            _, loss, _ = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"epoch={epoch + 1} step={step:03d} sft_loss={loss.item():.4f}")
            if args.max_steps > 0 and step >= args.max_steps:
                save_checkpoint(args.checkpoint, model, model_args)
                print(f"Saved SFT checkpoint to: {args.checkpoint}")
                return

    save_checkpoint(args.checkpoint, model, model_args)
    print(f"Saved SFT checkpoint to: {args.checkpoint}")


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature):
    model.eval()
    ids = input_ids.clone()
    for _ in range(max_new_tokens):
        context = ids[:, -model.block_size :]
        logits, _, _ = model(context)
        next_logits = logits[:, -1, :] / temperature
        next_id = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def generate_with_kv_cache(model, input_ids, max_new_tokens, temperature):
    model.eval()
    ids = input_ids.clone()
    logits, _, past_kv = model(ids, use_cache=True)
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :] / temperature
        next_id = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        logits, _, past_kv = model(next_id, past_kv=past_kv, use_cache=True)
    return ids


def infer(args):
    model, tokenizer, _ = load_checkpoint(args.checkpoint)
    input_ids = tokenizer.encode(args.prompt, add_bos=True).unsqueeze(0)
    if args.use_kv_cache:
        output_ids = generate_with_kv_cache(model, input_ids, args.max_new_tokens, args.temperature)
    else:
        output_ids = generate(model, input_ids, args.max_new_tokens, args.temperature)
    mode = "kv_cache" if args.use_kv_cache else "no_cache"
    print(f"[{mode}] {tokenizer.decode(output_ids[0].tolist())}")


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny Transformer LLM with KV cache.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--max_steps", type=int, default=120)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--block_size", type=int, default=96)
    train_parser.add_argument("--d_model", type=int, default=96)
    train_parser.add_argument("--n_heads", type=int, default=4)
    train_parser.add_argument("--n_layers", type=int, default=3)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--learning_rate", type=float, default=3e-4)
    train_parser.set_defaults(func=train)

    sft_parser = subparsers.add_parser("sft")
    sft_parser.add_argument("--input_checkpoint", default="")
    sft_parser.add_argument("--checkpoint", default=DEFAULT_SFT_CHECKPOINT)
    sft_parser.add_argument("--epochs", type=int, default=80)
    sft_parser.add_argument("--max_steps", type=int, default=160)
    sft_parser.add_argument("--batch_size", type=int, default=6)
    sft_parser.add_argument("--block_size", type=int, default=96)
    sft_parser.add_argument("--d_model", type=int, default=96)
    sft_parser.add_argument("--n_heads", type=int, default=4)
    sft_parser.add_argument("--n_layers", type=int, default=3)
    sft_parser.add_argument("--dropout", type=float, default=0.1)
    sft_parser.add_argument("--learning_rate", type=float, default=5e-4)
    sft_parser.set_defaults(func=sft)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint", default=DEFAULT_SFT_CHECKPOINT)
    infer_parser.add_argument("--prompt", default="User: 2+2?\nAssistant:")
    infer_parser.add_argument("--max_new_tokens", type=int, default=40)
    infer_parser.add_argument("--temperature", type=float, default=0.8)
    infer_parser.add_argument("--use_kv_cache", action="store_true")
    infer_parser.set_defaults(func=infer)

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["train"])
    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

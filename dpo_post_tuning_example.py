"""
Tiny DPO post-tuning example for a toy causal language model.

DPO trains directly on preference pairs:
    prompt + chosen answer should become more likely than prompt + rejected answer.

Unlike PPO/GRPO, there is no sampled rollout loop and no reward model call in the
training step. The frozen reference model anchors the update.

Run:
    python dpo_post_tuning_example.py train
    python dpo_post_tuning_example.py infer --prompt "capital_france?"
"""

import argparse
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(11)
random.seed(11)

TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "2+2?",
    "capital_france?",
    "opposite_hot?",
    "4",
    "5",
    "Paris",
    "London",
    "cold",
    "warm",
]
stoi = {token: index for index, token in enumerate(TOKENS)}
itos = {index: token for token, index in stoi.items()}
PAD, BOS, EOS = stoi["<pad>"], stoi["<bos>"], stoi["<eos>"]
DEFAULT_CHECKPOINT = "dpo_post_tuning_checkpoint.pt"

PREFERENCE_PAIRS = [
    (["<bos>", "2+2?"], ["4", "<eos>"], ["5", "<eos>"]),
    (["<bos>", "capital_france?"], ["Paris", "<eos>"], ["London", "<eos>"]),
    (["<bos>", "opposite_hot?"], ["cold", "<eos>"], ["warm", "<eos>"]),
]


def encode(tokens):
    return torch.tensor([stoi[token] for token in tokens], dtype=torch.long)


def decode(ids):
    return " ".join(itos[int(index)] for index in ids if int(index) not in (BOS, EOS, PAD))


def valid_prompts():
    return [pair[0][-1] for pair in PREFERENCE_PAIRS]


def encode_prompt(prompt):
    if prompt not in valid_prompts():
        choices = ", ".join(valid_prompts())
        raise ValueError(f"Unknown prompt {prompt!r}. Choose one of: {choices}")
    return ["<bos>", prompt]


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=48):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        hidden, _ = self.rnn(self.embed(input_ids))
        return self.lm_head(hidden)


def completion_logprob(model, prompt_ids, completion_ids):
    full = torch.cat([prompt_ids, completion_ids])
    logits = model(full[:-1].unsqueeze(0))[0]
    targets = full[1:]
    log_probs = F.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    return log_probs[start:, :].gather(1, targets[start:].unsqueeze(1)).sum()


def supervised_warmup(model, optimizer, steps=80):
    for _ in range(steps):
        prompt, chosen, _ = random.choice(PREFERENCE_PAIRS)
        loss = -completion_logprob(model, encode(prompt), encode(chosen))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def dpo_loss(policy_model, reference_model, prompt, chosen, rejected, beta=0.2):
    prompt_ids = encode(prompt)
    chosen_ids = encode(chosen)
    rejected_ids = encode(rejected)

    policy_chosen = completion_logprob(policy_model, prompt_ids, chosen_ids)
    policy_rejected = completion_logprob(policy_model, prompt_ids, rejected_ids)

    with torch.no_grad():
        ref_chosen = completion_logprob(reference_model, prompt_ids, chosen_ids)
        ref_rejected = completion_logprob(reference_model, prompt_ids, rejected_ids)

    policy_margin = policy_chosen - policy_rejected
    reference_margin = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (policy_margin - reference_margin))


def dpo_post_tune(model, reference_model, optimizer, steps=160):
    for step in range(1, steps + 1):
        prompt, chosen, rejected = random.choice(PREFERENCE_PAIRS)
        loss = dpo_loss(model, reference_model, prompt, chosen, rejected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 40 == 0:
            print(f"step={step:03d} loss={loss.item():.4f}")


@torch.no_grad()
def greedy_answer(model, prompt):
    prompt_ids = encode(prompt)
    logits = model(prompt_ids.unsqueeze(0))[0, -1]
    logits[[PAD, BOS, EOS]] = -1e9
    answer_id = torch.argmax(logits).item()
    return itos[answer_id]


def show_answers(model):
    for prompt, chosen, rejected in PREFERENCE_PAIRS:
        print(
            f"{decode(encode(prompt))} -> {greedy_answer(model, prompt)} "
            f"(chosen={chosen[0]}, rejected={rejected[0]})"
        )


def save_checkpoint(path, model):
    torch.save({"policy_state_dict": model.state_dict()}, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = TinyCausalLM(len(TOKENS))
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model


def train(args):
    model = TinyCausalLM(len(TOKENS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print("Before tuning:")
    show_answers(model)

    supervised_warmup(model, optimizer)
    reference_model = copy.deepcopy(model).eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    dpo_post_tune(model, reference_model, optimizer)

    print("\nAfter DPO post-tuning:")
    show_answers(model)
    save_checkpoint(args.checkpoint, model)
    print(f"\nSaved checkpoint to: {args.checkpoint}")


@torch.no_grad()
def infer(args):
    model = load_checkpoint(args.checkpoint)
    prompt = encode_prompt(args.prompt)
    answer = greedy_answer(model, prompt)
    print(f"{decode(encode(prompt))} -> {answer}")


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny DPO post-tuning example.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    infer_parser.add_argument("--prompt", default="2+2?", choices=valid_prompts())
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

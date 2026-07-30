"""
Tiny GRPO post-tuning example for a toy causal language model.

GRPO is like a PPO-style policy update without a learned value model. For each
prompt, sample a group of completions, score them, normalize rewards within that
group, and use those relative advantages to update the policy.

Run:
    python grpo_post_tuning_example.py train
    python grpo_post_tuning_example.py infer --prompt "opposite_hot?"
"""

import argparse
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(17)
random.seed(17)

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
DEFAULT_CHECKPOINT = "grpo_post_tuning_checkpoint.pt"

PROMPTS = [
    ["<bos>", "2+2?"],
    ["<bos>", "capital_france?"],
    ["<bos>", "opposite_hot?"],
]
GOLD_ANSWERS = {
    "2+2?": "4",
    "capital_france?": "Paris",
    "opposite_hot?": "cold",
}


def encode(tokens):
    return torch.tensor([stoi[token] for token in tokens], dtype=torch.long)


def decode(ids):
    return " ".join(itos[int(index)] for index in ids if int(index) not in (BOS, EOS, PAD))


def encode_prompt(prompt):
    if prompt not in GOLD_ANSWERS:
        choices = ", ".join(GOLD_ANSWERS)
        raise ValueError(f"Unknown prompt {prompt!r}. Choose one of: {choices}")
    return encode(["<bos>", prompt])


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


@torch.no_grad()
def sample_completion(model, prompt_ids, max_new_tokens=2, temperature=1.0):
    generated = []
    context = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(context.unsqueeze(0))[0, -1] / temperature
        logits[[PAD, BOS]] = -1e9
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        context = torch.cat([context, torch.tensor([next_id])])
        if next_id == EOS:
            break
    if generated[-1] != EOS:
        generated.append(EOS)
    return torch.tensor(generated, dtype=torch.long)


def reward_for(prompt_ids, completion_ids):
    prompt_token = itos[int(prompt_ids[-1])]
    answer = next((itos[int(token)] for token in completion_ids if int(token) != EOS), "")
    correct = answer == GOLD_ANSWERS[prompt_token]
    concise = len(completion_ids) <= 2
    return float(correct) + 0.1 * float(concise)


def supervised_warmup(model, optimizer, steps=120):
    examples = [
        (encode(prompt), encode([GOLD_ANSWERS[prompt[-1]], "<eos>"]))
        for prompt in PROMPTS
    ]
    for _ in range(steps):
        prompt_ids, completion_ids = random.choice(examples)
        loss = -completion_logprob(model, prompt_ids, completion_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def grpo_post_tune(model, reference_model, optimizer, steps=100, group_size=6, kl_beta=0.03):
    for step in range(1, steps + 1):
        prompt_ids = encode(random.choice(PROMPTS))

        with torch.no_grad():
            completions = [sample_completion(model, prompt_ids) for _ in range(group_size)]
            rewards = torch.tensor([reward_for(prompt_ids, completion) for completion in completions])
            advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            ref_logprobs = [
                completion_logprob(reference_model, prompt_ids, completion)
                for completion in completions
            ]

        losses = []
        for completion, advantage, ref_logprob in zip(completions, advantages, ref_logprobs):
            logprob = completion_logprob(model, prompt_ids, completion)
            policy_loss = -advantage * logprob
            kl_penalty = logprob - ref_logprob
            losses.append(policy_loss + kl_beta * kl_penalty)

        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            best_index = int(torch.argmax(rewards))
            print(
                f"step={step:03d} mean_reward={rewards.mean().item():.2f} "
                f"best='{decode(completions[best_index])}' loss={loss.item():+.3f}"
            )


@torch.no_grad()
def show_generations(model):
    for prompt in PROMPTS:
        prompt_ids = encode(prompt)
        completion = sample_completion(model, prompt_ids, temperature=0.3)
        print(f"{decode(prompt_ids)} -> {decode(completion)}")


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
    show_generations(model)

    supervised_warmup(model, optimizer)
    reference_model = copy.deepcopy(model).eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    print("\nAfter supervised warmup:")
    show_generations(model)

    grpo_post_tune(model, reference_model, optimizer)

    print("\nAfter GRPO post-tuning:")
    show_generations(model)
    save_checkpoint(args.checkpoint, model)
    print(f"\nSaved checkpoint to: {args.checkpoint}")


@torch.no_grad()
def infer(args):
    model = load_checkpoint(args.checkpoint)
    prompt_ids = encode_prompt(args.prompt)
    completion = sample_completion(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"{decode(prompt_ids)} -> {decode(completion)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny GRPO post-tuning example.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    infer_parser.add_argument("--prompt", default="2+2?", choices=list(GOLD_ANSWERS))
    infer_parser.add_argument("--max_new_tokens", type=int, default=2)
    infer_parser.add_argument("--temperature", type=float, default=0.3)
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

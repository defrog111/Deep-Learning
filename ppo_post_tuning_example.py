"""
Tiny PPO post-tuning example for a toy causal language model.

This file intentionally avoids RLHF frameworks so the moving parts are visible:
1. Supervised warmup teaches a tiny LM basic prompt -> answer behavior.
2. A reward model learns to score chosen answers above rejected answers.
3. A frozen reference model keeps the policy from drifting too far.
4. A value model learns to predict reward-model scores.
5. PPO updates the policy from sampled completions, advantages, and KL penalty.

Run:
    python ppo_post_tuning_example.py train
    python ppo_post_tuning_example.py infer --prompt "2+2?"
"""

import argparse
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(7)
random.seed(7)

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
DEFAULT_CHECKPOINT = "ppo_post_tuning_checkpoint.pt"

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
PREFERENCE_PAIRS = [
    (["<bos>", "2+2?"], ["4", "<eos>"], ["5", "<eos>"]),
    (["<bos>", "capital_france?"], ["Paris", "<eos>"], ["London", "<eos>"]),
    (["<bos>", "opposite_hot?"], ["cold", "<eos>"], ["warm", "<eos>"]),
]


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


class RewardModel(nn.Module):
    """Scores a full prompt + completion with one scalar reward."""

    def __init__(self, vocab_size, hidden_size=48):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids):
        hidden, _ = self.rnn(self.embed(input_ids.unsqueeze(0)))
        last_hidden = hidden[0, -1]
        return self.reward_head(last_hidden).squeeze()


class ValueModel(nn.Module):
    """Predicts expected reward for a full prompt + completion."""

    def __init__(self, vocab_size, hidden_size=48):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids):
        hidden, _ = self.rnn(self.embed(input_ids.unsqueeze(0)))
        last_hidden = hidden[0, -1]
        return self.value_head(last_hidden).squeeze()


def completion_logprob(model, prompt_ids, completion_ids):
    full = torch.cat([prompt_ids, completion_ids])
    logits = model(full[:-1].unsqueeze(0))[0]
    targets = full[1:]
    log_probs = F.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    return log_probs[start:, :].gather(1, targets[start:].unsqueeze(1)).sum()


@torch.no_grad()
def sample_completion(model, prompt_ids, max_new_tokens=2, temperature=0.9):
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


def reward_score(reward_model, prompt_ids, completion_ids):
    full = torch.cat([prompt_ids, completion_ids])
    return reward_model(full)


def value_score(value_model, prompt_ids, completion_ids):
    full = torch.cat([prompt_ids, completion_ids])
    return value_model(full)


def supervised_warmup(model, optimizer, steps=150):
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


def train_reward_model(reward_model, optimizer, steps=250):
    """Train chosen completions to score higher than rejected completions."""
    for step in range(1, steps + 1):
        prompt, chosen, rejected = random.choice(PREFERENCE_PAIRS)
        prompt_ids = encode(prompt)
        chosen_ids = encode(chosen)
        rejected_ids = encode(rejected)

        chosen_reward = reward_score(reward_model, prompt_ids, chosen_ids)
        rejected_reward = reward_score(reward_model, prompt_ids, rejected_ids)
        loss = -F.logsigmoid(chosen_reward - rejected_reward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            margin = chosen_reward.item() - rejected_reward.item()
            print(f"reward_model_step={step:03d} loss={loss.item():.4f} margin={margin:+.2f}")


@torch.no_grad()
def show_reward_model_scores(reward_model):
    for prompt, chosen, rejected in PREFERENCE_PAIRS:
        prompt_ids = encode(prompt)
        chosen_reward = reward_score(reward_model, prompt_ids, encode(chosen)).item()
        rejected_reward = reward_score(reward_model, prompt_ids, encode(rejected)).item()
        print(
            f"{decode(prompt_ids)} chosen={chosen[0]}:{chosen_reward:+.2f} "
            f"rejected={rejected[0]}:{rejected_reward:+.2f}"
        )


def ppo_post_tune(
    model,
    reference_model,
    reward_model,
    value_model,
    policy_optimizer,
    value_optimizer,
    epochs=80,
    clip_eps=0.2,
    kl_beta=0.05,
    value_coef=0.5,
):
    for step in range(1, epochs + 1):
        prompt_ids = encode(random.choice(PROMPTS))
        completion_ids = sample_completion(model, prompt_ids)

        with torch.no_grad():
            old_logprob = completion_logprob(model, prompt_ids, completion_ids)
            ref_logprob = completion_logprob(reference_model, prompt_ids, completion_ids)
            reward = reward_score(reward_model, prompt_ids, completion_ids)
            old_value = value_score(value_model, prompt_ids, completion_ids)
            advantage = reward - old_value

        new_logprob = completion_logprob(model, prompt_ids, completion_ids)
        new_value = value_score(value_model, prompt_ids, completion_ids)
        ratio = torch.exp(new_logprob - old_logprob)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage

        policy_loss = -torch.min(unclipped, clipped)
        kl_penalty = new_logprob - ref_logprob
        value_loss = F.mse_loss(new_value, reward)
        loss = policy_loss + kl_beta * kl_penalty + value_coef * value_loss

        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()
        value_optimizer.step()

        if step % 20 == 0:
            print(
                f"step={step:03d} reward={reward.item():+.1f} "
                f"value={new_value.item():+.1f} advantage={advantage.item():+.1f} "
                f"sample='{decode(completion_ids)}' loss={loss.item():+.3f}"
            )


@torch.no_grad()
def show_generations(model):
    for prompt in PROMPTS:
        prompt_ids = encode(prompt)
        completion = sample_completion(model, prompt_ids, temperature=0.3)
        print(f"{decode(prompt_ids)} -> {decode(completion)}")


def save_checkpoint(path, model, reward_model, value_model):
    torch.save(
        {
            "policy_state_dict": model.state_dict(),
            "reward_state_dict": reward_model.state_dict(),
            "value_state_dict": value_model.state_dict(),
        },
        path,
    )


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = TinyCausalLM(len(TOKENS))
    reward_model = RewardModel(len(TOKENS))
    value_model = ValueModel(len(TOKENS))
    model.load_state_dict(checkpoint["policy_state_dict"])
    reward_model.load_state_dict(checkpoint["reward_state_dict"])
    value_model.load_state_dict(checkpoint["value_state_dict"])
    model.eval()
    reward_model.eval()
    value_model.eval()
    return model, reward_model, value_model


def train(args):
    model = TinyCausalLM(len(TOKENS))
    policy_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    reward_model = RewardModel(len(TOKENS))
    reward_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=3e-3)
    value_model = ValueModel(len(TOKENS))
    value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=3e-3)

    print("Before tuning:")
    show_generations(model)

    supervised_warmup(model, policy_optimizer)
    reference_model = copy.deepcopy(model).eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    print("\nAfter supervised warmup:")
    show_generations(model)

    print("\nTraining reward model:")
    train_reward_model(reward_model, reward_optimizer)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad_(False)

    print("\nReward model scores:")
    show_reward_model_scores(reward_model)

    ppo_post_tune(
        model,
        reference_model,
        reward_model,
        value_model,
        policy_optimizer,
        value_optimizer,
    )

    print("\nAfter PPO post-tuning:")
    show_generations(model)
    save_checkpoint(args.checkpoint, model, reward_model, value_model)
    print(f"\nSaved checkpoint to: {args.checkpoint}")


@torch.no_grad()
def infer(args):
    model, reward_model, value_model = load_checkpoint(args.checkpoint)
    prompt_ids = encode_prompt(args.prompt)
    completion = sample_completion(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    reward = reward_score(reward_model, prompt_ids, completion).item()
    value = value_score(value_model, prompt_ids, completion).item()
    print(f"{decode(prompt_ids)} -> {decode(completion)}")
    print(f"reward_model_score={reward:+.2f} value_model_score={value:+.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny PPO post-tuning example.")
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

"""
LoRA fine-tuning + inference example for a Hugging Face language model.

Install dependencies:
    pip install torch transformers datasets peft accelerate

Train a LoRA adapter:
    python lora_finetune_example.py train

Run inference with the trained adapter:
    python lora_finetune_example.py infer --prompt "User: What is LoRA?\nAssistant:"

This file uses a tiny online model by default so the full train/infer loop can run
on a normal laptop. Replace --model_name with a larger causal language model when
you are ready, such as gpt2, Qwen/Qwen2.5-0.5B, or TinyLlama/TinyLlama-1.1B-Chat-v1.0.
"""

import argparse
from pathlib import Path

import torch


DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"
DEFAULT_OUTPUT_DIR = "lora-tiny-gpt2-example"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_training_dependencies():
    try:
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install dependencies with:\n"
            "    pip install -r requirements.txt"
        ) from exc

    return {
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def build_toy_dataset(deps):
    """Small instruction-style dataset. Replace this with your real training data."""
    samples = [
        {
            "instruction": "What is LoRA?",
            "answer": "LoRA is a parameter-efficient fine-tuning method that trains small adapter weights.",
        },
        {
            "instruction": "Why use LoRA?",
            "answer": "LoRA reduces memory usage and training cost because most base model weights stay frozen.",
        },
        {
            "instruction": "What is fine tuning?",
            "answer": "Fine tuning adapts a pretrained model to a more specific task or response style.",
        },
        {
            "instruction": "How should answers be written?",
            "answer": "Answers should be concise, direct, and useful.",
        },
        {
            "instruction": "Give one benefit of adapter training.",
            "answer": "Adapter training lets you save and share a small set of task-specific weights.",
        },
        {
            "instruction": "What gets saved after LoRA training?",
            "answer": "The LoRA adapter is saved, while the original base model can be loaded separately.",
        },
    ]
    texts = [
        f"User: {item['instruction']}\nAssistant: {item['answer']}"
        for item in samples
    ]
    return deps["Dataset"].from_dict({"text": texts})


def load_tokenizer(model_name: str, deps):
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, deps):
    model = deps["AutoModelForCausalLM"].from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id
    return model


def build_lora_model(model_name: str, deps):
    model = load_base_model(model_name, deps)

    # GPT-2 style models use "c_attn" for the fused query/key/value projection.
    # For other model families you may need different target_modules, for example:
    # LLaMA/Qwen style: ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = deps["LoraConfig"](
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type=deps["TaskType"].CAUSAL_LM,
    )
    return deps["get_peft_model"](model, lora_config)


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    device = get_device()
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def train(args) -> None:
    deps = require_training_dependencies()
    output_dir = Path(args.output_dir)
    tokenizer = load_tokenizer(args.model_name, deps)
    model = build_lora_model(args.model_name, deps)
    model.print_trainable_parameters()

    dataset = build_toy_dataset(deps)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)

    training_args = deps["TrainingArguments"](
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=deps["DataCollatorForLanguageModeling"](
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nLoRA adapter saved to: {output_dir}")

    print("\nQuick inference check:")
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(text)


def infer(args) -> None:
    deps = require_training_dependencies()
    adapter_dir = Path(args.output_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}. Run train first."
        )

    tokenizer = load_tokenizer(args.model_name, deps)
    base_model = load_base_model(args.model_name, deps)
    model = deps["PeftModel"].from_pretrained(base_model, adapter_dir)

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and run inference with a LoRA adapter."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    common.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    common.add_argument(
        "--prompt",
        default="User: What is LoRA?\nAssistant:",
        help="Prompt used for inference.",
    )
    common.add_argument("--max_new_tokens", type=int, default=50)
    common.add_argument("--temperature", type=float, default=0.7)

    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=2)
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train_parser.add_argument("--learning_rate", type=float, default=2e-4)
    train_parser.add_argument("--max_length", type=int, default=128)
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser("infer", parents=[common])
    infer_parser.set_defaults(func=infer)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Device: {get_device()}")
    args.func(args)


if __name__ == "__main__":
    main()

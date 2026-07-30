"""例 12：读取 CSV，使用手写 Transformer Encoder 完成表格多分类。

关键设计：把每个数值特征视为一个 token，再添加 [CLS] token。
Transformer 学习特征之间的关系，最终用 [CLS] 表示完成分类。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from components import EncoderBlock
from importlib import import_module


CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def ensure_dataset(csv_path: Path) -> None:
    if csv_path.exists():
        return
    downloader = import_module("11_download_iris_csv")
    downloader.download_iris(csv_path)


def load_csv(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    labels: list[int] = []
    label_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    with csv_path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            features.append(
                [
                    float(row["sepal_length"]),
                    float(row["sepal_width"]),
                    float(row["petal_length"]),
                    float(row["petal_width"]),
                ]
            )
            labels.append(label_to_id[row["species"]])
    return torch.tensor(features), torch.tensor(labels, dtype=torch.long)


def stratified_split(
    features: torch.Tensor, labels: torch.Tensor, seed: int
) -> tuple[torch.Tensor, ...]:
    """按类别分层切为 60% train、20% validation、20% test。"""
    generator = random.Random(seed)
    split_indices = [[], [], []]
    for class_id in range(len(CLASS_NAMES)):
        indices = torch.where(labels == class_id)[0].tolist()
        generator.shuffle(indices)
        train_end = int(len(indices) * 0.6)
        valid_end = int(len(indices) * 0.8)
        for bucket, values in zip(
            split_indices,
            (indices[:train_end], indices[train_end:valid_end], indices[valid_end:]),
        ):
            bucket.extend(values)
    return tuple(
        tensor[index]
        for index in split_indices
        for tensor in (features, labels)
    )


class NumericFeatureTokenizer(nn.Module):
    """把每个 scalar 特征映射成一个 d_model 维 token。"""

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))
        self.feature_identity = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.unsqueeze(-1) * self.weight
            + self.bias
            + self.feature_identity
        )


class TabularTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(num_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    hidden_dim=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, features: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(features)
        cls = self.cls_token.expand(features.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        attention = None
        for block in self.blocks:
            tokens, attention = block(tokens)
        logits = self.classifier(self.norm(tokens[:, 0]))
        if return_attention:
            return logits, attention
        return logits


def accuracy(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(features).argmax(dim=1)
    return (predictions == labels).float().mean().item()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    ensure_dataset(args.csv)
    features, labels = load_csv(args.csv)
    train_x, train_y, valid_x, valid_y, test_x, test_y = stratified_split(
        features, labels, args.seed
    )

    # 只能用训练集统计量，避免 data leakage。
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    valid_x = (valid_x - mean) / std
    test_x = (test_x - mean) / std

    model = TabularTransformerClassifier(4, len(CLASS_NAMES))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    best_state = None
    best_validation = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        validation = accuracy(model, valid_x, valid_y)
        if validation > best_validation:
            best_validation = validation
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} "
                f"loss={total_loss / len(train_x):.4f} "
                f"valid_accuracy={validation:.3f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    test_accuracy = accuracy(model, test_x, test_y)
    print(f"best_valid_accuracy={best_validation:.3f}")
    print(f"test_accuracy={test_accuracy:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": mean,
            "std": std,
            "class_names": CLASS_NAMES,
            "model_config": {
                "num_features": 4,
                "num_classes": len(CLASS_NAMES),
            },
        },
        args.output,
    )
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_validation_accuracy": best_validation,
                "test_accuracy": test_accuracy,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"checkpoint={args.output}")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=base / "data" / "iris.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "artifacts" / "iris_transformer.pt",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

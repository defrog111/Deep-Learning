"""例 18（训练）：读取真实 CSV，用 4 层 PyTorch 原生 Transformer 分类。

完整数据流：CSV -> 分层切分 -> 仅用训练集标准化 -> 特征 token ->
4 x nn.TransformerEncoderLayer -> [CLS] -> 三分类 -> 保存 checkpoint。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from importlib import import_module
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from builtin_csv_classifier_model import BuiltinTabularTransformerClassifier


CLASS_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def ensure_dataset(csv_path: Path) -> None:
    """本地没有 iris.csv 时，调用例 11 从 UCI 下载并生成。"""
    if not csv_path.exists():
        import_module("11_download_iris_csv").download_iris(csv_path)


def load_csv(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 CSV；X 是 float32，y 是类别 id。"""
    label_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    features: list[list[float]] = []
    labels: list[int] = []
    with csv_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            features.append([float(row[name]) for name in FEATURE_NAMES])
            labels.append(label_to_id[row["species"]])
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels)


def stratified_split(
    features: torch.Tensor, labels: torch.Tensor, seed: int
) -> tuple[torch.Tensor, ...]:
    """每个类别分别打乱，再按 60%/20%/20% 切分。"""
    rng = random.Random(seed)
    train_indices: list[int] = []
    valid_indices: list[int] = []
    test_indices: list[int] = []
    for class_id in range(len(CLASS_NAMES)):
        indices = torch.where(labels == class_id)[0].tolist()
        rng.shuffle(indices)
        train_end = int(len(indices) * 0.6)
        valid_end = int(len(indices) * 0.8)
        train_indices.extend(indices[:train_end])
        valid_indices.extend(indices[train_end:valid_end])
        test_indices.extend(indices[valid_end:])
    return (
        features[train_indices], labels[train_indices],
        features[valid_indices], labels[valid_indices],
        features[test_indices], labels[test_indices],
    )


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return (model(x).argmax(dim=1) == y).float().mean().item()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    ensure_dataset(args.csv)
    features, labels = load_csv(args.csv)
    train_x, train_y, valid_x, valid_y, test_x, test_y = stratified_split(
        features, labels, args.seed
    )

    # 防止 data leakage：均值、标准差只能从 train 计算。
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    valid_x = (valid_x - mean) / std
    test_x = (test_x - mean) / std

    model_config = {
        "num_features": len(FEATURE_NAMES),
        "num_classes": len(CLASS_NAMES),
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    model = BuiltinTabularTransformerClassifier(**model_config)
    print(f"读取 CSV: {args.csv}，样本数={len(features)}")
    print(f"原生 TransformerEncoder 层数={len(model.encoder.layers)}")

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    loss_fn = nn.CrossEntropyLoss()
    best_validation = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                f"epoch={epoch:03d} loss={total_loss / len(train_x):.4f} "
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
            "model_config": model_config,
            "mean": mean,
            "std": std,
            "feature_names": FEATURE_NAMES,
            "class_names": CLASS_NAMES,
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "best_validation_accuracy": best_validation,
                "test_accuracy": test_accuracy,
                "num_layers": args.num_layers,
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
        default=base / "artifacts" / "builtin_iris_transformer.pt",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=30)
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

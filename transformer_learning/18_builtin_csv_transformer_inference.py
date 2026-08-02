"""例 18（推理）：独立加载原生多层 CSV Transformer checkpoint。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from builtin_csv_classifier_model import BuiltinTabularTransformerClassifier


def main() -> None:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base / "artifacts" / "builtin_iris_transformer.pt",
    )
    parser.add_argument(
        "--features",
        type=float,
        nargs=4,
        default=[5.1, 3.5, 1.4, 0.2],
        metavar=("SL", "SW", "PL", "PW"),
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = BuiltinTabularTransformerClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    features = torch.tensor([args.features], dtype=torch.float32)
    normalized = (features - checkpoint["mean"]) / checkpoint["std"]
    with torch.no_grad():
        probabilities = model(normalized).softmax(dim=1)[0]

    print("输入特征:")
    for name, value in zip(checkpoint["feature_names"], args.features):
        print(f"  {name:13s}: {value:.2f}")
    print("类别概率:")
    for name, probability in zip(checkpoint["class_names"], probabilities):
        print(f"  {name:10s}: {probability.item():.4f}")
    predicted = probabilities.argmax().item()
    print("预测类别:", checkpoint["class_names"][predicted])


if __name__ == "__main__":
    main()

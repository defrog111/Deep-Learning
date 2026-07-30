"""例 13：加载 CSV Transformer checkpoint，对一朵花做推理。"""

import argparse
from pathlib import Path

import torch

from importlib import import_module


training = import_module("12_csv_transformer_classifier")


def main() -> None:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base / "artifacts" / "iris_transformer.pt",
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
    model = training.TabularTransformerClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    features = torch.tensor([args.features], dtype=torch.float32)
    normalized = (features - checkpoint["mean"]) / checkpoint["std"]
    with torch.no_grad():
        probabilities = model(normalized).softmax(dim=1)[0]
    predicted = probabilities.argmax().item()

    print("probabilities:")
    for name, probability in zip(checkpoint["class_names"], probabilities):
        print(f"  {name:10s}: {probability.item():.4f}")
    print("prediction:", checkpoint["class_names"][predicted])


if __name__ == "__main__":
    main()

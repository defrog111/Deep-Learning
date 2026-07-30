"""例 15：可视化表格 Transformer 的 [CLS] 对各特征注意力。"""

from pathlib import Path
from importlib import import_module

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


training = import_module("12_csv_transformer_classifier")
BASE = Path(__file__).parent
CHECKPOINT = BASE / "artifacts" / "iris_transformer.pt"
OUTPUT = BASE / "artifacts" / "iris_attention.png"

if not CHECKPOINT.exists():
    raise SystemExit("请先运行：python 12_csv_transformer_classifier.py")

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
model = training.TabularTransformerClassifier(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state"])
model.eval()

raw_features = torch.tensor([[5.1, 3.5, 1.4, 0.2]])
features = (raw_features - checkpoint["mean"]) / checkpoint["std"]
with torch.no_grad():
    logits, attention = model(features, return_attention=True)

# attention: (B, H, query, key)。取 CLS query，再对 head 求平均。
cls_attention = attention[0, :, 0, 1:].mean(dim=0).numpy()
feature_names = ["sepal length", "sepal width", "petal length", "petal width"]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(7, 4))
plt.bar(feature_names, cls_attention)
plt.ylabel("mean attention weight")
plt.title("[CLS] attention to Iris features")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=160)
print("prediction:", checkpoint["class_names"][logits.argmax(dim=1).item()])
print("saved:", OUTPUT)

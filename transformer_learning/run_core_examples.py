"""依次运行不需要下载数据的核心示例。"""

import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).parent
EXAMPLES = [
    "01_embeddings.py",
    "02_positional_encoding.py",
    "03_scaled_dot_product_attention.py",
    "04_multi_head_attention.py",
    "05_residual_norm_ffn.py",
    "06_encoder_block.py",
    "07_attention_masks.py",
    "08_decoder_block.py",
    "09_encoder_decoder_transformer.py",
    "10_tiny_causal_language_model.py",
    "14_pytorch_builtin_comparison.py",
    "16_torch_transformer_encoder.py",
    "17_torch_transformer_decoder.py",
]

for example in EXAMPLES:
    print(f"\n{'=' * 18} {example} {'=' * 18}", flush=True)
    subprocess.run([sys.executable, str(BASE / example)], check=True)

print("\n全部核心示例运行成功。")

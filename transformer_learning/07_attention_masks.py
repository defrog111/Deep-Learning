"""例 07：padding mask 与 causal mask 的含义和组合。"""

import torch

from components import causal_mask, padding_mask


token_ids = torch.tensor([[4, 8, 2, 0, 0], [3, 7, 1, 9, 0]])
pad = padding_mask(token_ids, pad_id=0)  # 屏蔽补齐位置
causal = causal_mask(token_ids.size(1))  # 屏蔽未来位置
combined = pad & causal

print("padding mask:\n", pad[0, 0].int())
print("causal mask:\n", causal[0, 0].int())
print("combined mask for sample 0:\n", combined[0, 0].int())

"""例 10：Decoder-only 思路的最小语言模型、next-token loss 和生成。

为保持代码短小，这里用带 causal mask 的 Encoder Block 实现 GPT 风格主干。
"""

import torch
from torch import nn
from torch.nn import functional as F

from components import EncoderBlock, SinusoidalPositionalEncoding, causal_mask


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_len=128)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, 4, d_model * 4) for _ in range(2)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.position(self.embedding(token_ids))
        mask = causal_mask(token_ids.size(1), token_ids.device)
        for block in self.blocks:
            x, _ = block(x, mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, steps: int) -> torch.Tensor:
        self.eval()
        for _ in range(steps):
            logits = self(token_ids)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_id], dim=1)
        return token_ids


torch.manual_seed(0)
vocab_size = 12
model = TinyCausalLM(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
sequence = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])

for step in range(101):
    inputs, targets = sequence[:, :-1], sequence[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 25 == 0:
        print(f"step={step:03d} loss={loss.item():.4f}")

prompt = torch.tensor([[1, 2, 3]])
print("generated ids:", model.generate(prompt, steps=5).tolist())

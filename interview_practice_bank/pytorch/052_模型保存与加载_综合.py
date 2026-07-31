"""
题目 052：模型保存与加载_综合

要求：完成“模型保存与加载”的综合题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

from pathlib import Path  # 导入路径工具。
import tempfile  # 导入临时目录工具。
import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
torch.manual_seed(0)  # 固定初始化。
model = nn.Linear(3, 2)  # 创建待保存模型。
with tempfile.TemporaryDirectory() as folder:  # 使用自动清理的临时目录。
    path = Path(folder) / 'model.pt'  # 组成checkpoint路径。
    torch.save({'model_state': model.state_dict(), 'epoch': 5}, path)  # 保存state_dict和元数据。
    checkpoint = torch.load(path, weights_only=True)  # 安全加载仅含Tensor的checkpoint。
    restored = nn.Linear(3, 2)  # 创建相同结构模型。
    restored.load_state_dict(checkpoint['model_state'])  # 恢复参数。
assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))  # 验证参数完全一致。
print(checkpoint['epoch'])  # 输出恢复的训练元数据。

"""
题目 050：模型保存与加载_变式

要求：完成“模型保存与加载”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定初始化。
2. 创建待保存模型。
3. 使用自动清理的临时目录。
4. 组成checkpoint路径。
5. 保存state_dict和元数据。
6. 安全加载仅含Tensor的checkpoint。
7. 创建相同结构模型。
8. 恢复参数。
9. 完整checkpoint保存轮数、模型和优化器。

完成标准：
- 验证参数完全一致。
- 验证恢复训练必需字段。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
checkpoint_model = nn.Linear(2, 1); checkpoint_optimizer = torch.optim.SGD(checkpoint_model.parameters(), lr=0.1, momentum=0.9); checkpoint = {'epoch': 3, 'model': checkpoint_model.state_dict(), 'optimizer': checkpoint_optimizer.state_dict()}  # 完整checkpoint保存轮数、模型和优化器。
assert checkpoint['epoch'] == 3 and set(checkpoint) == {'epoch', 'model', 'optimizer'}  # 验证恢复训练必需字段。

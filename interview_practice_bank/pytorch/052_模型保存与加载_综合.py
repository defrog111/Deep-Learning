"""
题目 052：模型保存与加载_综合

要求：完成“模型保存与加载”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入路径工具。
2. 导入临时目录工具。
3. 导入 PyTorch。
4. 导入模型模块。
5. 导入内存二进制流。
6. 使用map_location跨设备恢复完整checkpoint。
7. 综合验证序列化和权重恢复。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import tempfile  # 导入临时目录工具。
import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
import io  # 导入内存二进制流。
serialized_model = nn.Linear(2, 1); serialized_optimizer = torch.optim.SGD(serialized_model.parameters(), lr=0.1); serialized_checkpoint = {'epoch': 3, 'model': serialized_model.state_dict(), 'optimizer': serialized_optimizer.state_dict()}; buffer = io.BytesIO(); torch.save(serialized_checkpoint, buffer); buffer.seek(0); restored_checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)  # 使用map_location跨设备恢复完整checkpoint。
restored_model = nn.Linear(2, 1); restored_model.load_state_dict(restored_checkpoint['model']); assert restored_checkpoint['epoch'] == 3  # 综合验证序列化和权重恢复。

"""
题目 006：reshapeviewpermute与contiguous_变式

要求：完成“reshapeviewpermute与contiguous”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 创建三维Tensor。
3. 使用flatten和unflatten显式管理维度。
4. 验证往返变形。

完成标准：
- 验证往返变形。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
tensor = torch.arange(24).reshape(2, 3, 4)  # 创建三维Tensor。
flattened_extra = tensor.flatten(start_dim=1); restored_extra = flattened_extra.unflatten(1, (3, 4))  # 使用flatten和unflatten显式管理维度。
assert restored_extra.shape == tensor.shape and torch.equal(restored_extra, tensor)  # 验证往返变形。

"""
题目 008：reshapeviewpermute与contiguous_综合

要求：完成“reshapeviewpermute与contiguous”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 创建三维Tensor。
3. 综合使用movedim、unsqueeze和squeeze。
4. 验证轴移动和长度1维。

完成标准：
- 验证轴移动和长度1维。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
tensor = torch.arange(24).reshape(2, 3, 4)  # 创建三维Tensor。
moved_extra = torch.movedim(tensor, 0, -1); squeezed_extra = tensor.unsqueeze(0).squeeze(0)  # 综合使用movedim、unsqueeze和squeeze。
assert moved_extra.shape[-1] == tensor.shape[0] and torch.equal(squeezed_extra, tensor)  # 验证轴移动和长度1维。

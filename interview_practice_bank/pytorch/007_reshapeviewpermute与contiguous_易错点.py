"""
题目 007：reshapeviewpermute与contiguous_易错点

要求：完成“reshapeviewpermute与contiguous”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建三维Tensor。
2. transpose通常产生非连续视图。
3. view前先contiguous。

完成标准：
- view前先contiguous。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
tensor = torch.arange(24).reshape(2, 3, 4)  # 创建三维Tensor。
transposed_extra = tensor.transpose(1, 2)  # transpose通常产生非连续视图。
assert not transposed_extra.is_contiguous() and transposed_extra.contiguous().view(tensor.size(0), -1).is_contiguous()  # view前先contiguous。

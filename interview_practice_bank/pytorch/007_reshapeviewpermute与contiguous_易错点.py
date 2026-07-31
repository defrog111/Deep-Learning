"""
题目 007：reshapeviewpermute与contiguous_易错点

要求：完成“reshapeviewpermute与contiguous”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建三维Tensor。
2. 重排维度得到shape(4,2,3)。
3. permute通常产生非连续视图。
4. 先连续化再使用view展平。
5. reshape必要时自动复制以满足布局。

完成标准：
- 验证元素数不变。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
tensor = torch.arange(24).reshape(2, 3, 4)  # 创建三维Tensor。
permuted = tensor.permute(2, 0, 1)  # 重排维度得到shape(4,2,3)。
is_contiguous_before = permuted.is_contiguous()  # permute通常产生非连续视图。
flattened = permuted.contiguous().view(-1)  # 先连续化再使用view展平。
reshaped = permuted.reshape(4, 6)  # reshape必要时自动复制以满足布局。
assert flattened.numel() == tensor.numel() and reshaped.shape == (4, 6)  # 验证元素数不变。
print(is_contiguous_before, tensor.stride(), permuted.stride())  # 输出连续性和步幅。

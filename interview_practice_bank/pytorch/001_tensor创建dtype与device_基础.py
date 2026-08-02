"""
题目 001：Tensor创建dtype与device_基础

要求：完成“Tensor创建dtype与device”的基础题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 从Python数据创建二维float32 Tensor。
3. 验证Tensor的shape和维数。
4. 验证元素数和单元素字节数。
5. 输出基础Tensor元数据。

完成标准：
- 验证Tensor的shape和维数。
- 验证元素数和单元素字节数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)  # 从Python数据创建二维float32 Tensor。
assert tensor.shape == (2, 3) and tensor.ndim == 2  # 验证Tensor的shape和维数。
assert tensor.numel() == 6 and tensor.element_size() == 4  # 验证元素数和单元素字节数。
print(tensor, tensor.dtype, tensor.device, tensor.stride())  # 输出基础Tensor元数据。

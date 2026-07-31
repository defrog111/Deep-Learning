"""
题目 002：Tensor创建dtype与device_变式

要求：完成“Tensor创建dtype与device”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 显式指定浮点dtype创建Tensor。
2. 创建同shape、dtype和device的零Tensor。
3. 使用to转换dtype并返回新Tensor。
4. 自动选择可用计算设备。
5. 把数据移动到目标设备。

完成标准：
- 验证shape和类型。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 显式指定浮点dtype创建Tensor。
zeros = torch.zeros_like(tensor)  # 创建同shape、dtype和device的零Tensor。
converted = tensor.to(dtype=torch.float64)  # 使用to转换dtype并返回新Tensor。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择可用计算设备。
on_device = tensor.to(device)  # 把数据移动到目标设备。
assert tensor.shape == (2, 2) and converted.dtype == torch.float64  # 验证shape和类型。
print(tensor, zeros, on_device.device)  # 输出Tensor与设备。

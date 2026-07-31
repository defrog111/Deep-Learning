"""
题目 001：Tensor创建dtype与device_基础

要求：完成“Tensor创建dtype与device”的基础题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
tensor = torch.tensor([[1, 2], [3, 3]], dtype=torch.float32)  # 显式指定浮点dtype创建Tensor。
zeros = torch.zeros_like(tensor)  # 创建同shape、dtype和device的零Tensor。
converted = tensor.to(dtype=torch.float64)  # 使用to转换dtype并返回新Tensor。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择可用计算设备。
on_device = tensor.to(device)  # 把数据移动到目标设备。
assert tensor.shape == (2, 2) and converted.dtype == torch.float64  # 验证shape和类型。
print(tensor, zeros, on_device.device)  # 输出Tensor与设备。

"""
题目 002：Tensor创建dtype与device_变式

要求：完成“Tensor创建dtype与device”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 显式指定浮点dtype创建Tensor。
2. 创建同shape、dtype和device的零Tensor。
3. 使用to转换dtype并返回新Tensor。
4. 自动选择可用计算设备。
5. 把数据移动到目标设备。
6. from_numpy共享CPU内存，torch.tensor复制数据。

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
numpy_source = __import__('numpy').arange(4, dtype='float32'); shared_tensor = torch.from_numpy(numpy_source); copied_tensor = torch.tensor(numpy_source)  # from_numpy共享CPU内存，torch.tensor复制数据。
numpy_source[0] = 99; assert shared_tensor[0].item() == 99 and copied_tensor[0].item() != 99  # 验证共享与复制差异。

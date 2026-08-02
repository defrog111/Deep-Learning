"""
题目 002：Tensor创建dtype与device_变式

要求：完成“Tensor创建dtype与device”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建CPU NumPy源数组。
2. from_numpy与源数组共享CPU内存。
3. as_tensor在类型兼容时也尽量避免复制。
4. torch.tensor始终复制输入数据。
5. 修改NumPy源数组测试三种构造方式。

完成标准：
- 验证两个零复制Tensor看到修改。
- 验证复制与共享内存。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy用于演示内存关系。
import torch  # 导入 PyTorch。
source = np.arange(4, dtype=np.float32)  # 创建CPU NumPy源数组。
shared = torch.from_numpy(source)  # from_numpy与源数组共享CPU内存。
also_shared = torch.as_tensor(source)  # as_tensor在类型兼容时也尽量避免复制。
copied = torch.tensor(source)  # torch.tensor始终复制输入数据。
source[0] = 99  # 修改NumPy源数组测试三种构造方式。
assert shared[0].item() == also_shared[0].item() == 99  # 验证两个零复制Tensor看到修改。
assert copied[0].item() == 0 and shared.data_ptr() == also_shared.data_ptr()  # 验证复制与共享内存。
print(shared, also_shared, copied)  # 输出共享与复制结果。

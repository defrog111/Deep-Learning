"""
题目 004：Tensor创建dtype与device_综合

要求：完成“Tensor创建dtype与device”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 自动选择当前可用计算设备。
2. 在目标设备创建整数序列。
3. 在目标设备创建含终点等距序列。
4. 创建明确ij索引规则的二维网格。
5. 把坐标网格堆成最后维为2的特征Tensor。
6. 继承shape、dtype和device并开启梯度。

完成标准：
- 验证序列和网格shape。
- 验证叶子Tensor和设备。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择当前可用计算设备。
indices = torch.arange(start=0, end=12, step=2, dtype=torch.int64, device=device)  # 在目标设备创建整数序列。
points = torch.linspace(start=0.0, end=1.0, steps=6, device=device)  # 在目标设备创建含终点等距序列。
grid_x, grid_y = torch.meshgrid(points[:3], points[3:], indexing='ij')  # 创建明确ij索引规则的二维网格。
features = torch.stack((grid_x, grid_y), dim=-1)  # 把坐标网格堆成最后维为2的特征Tensor。
trainable = torch.zeros_like(features, requires_grad=True)  # 继承shape、dtype和device并开启梯度。
assert indices.tolist() == [0, 2, 4, 6, 8, 10] and features.shape == (3, 3, 2)  # 验证序列和网格shape。
assert trainable.is_leaf and trainable.requires_grad and trainable.device == device  # 验证叶子Tensor和设备。
print(indices, points, features.shape, trainable.device)  # 输出综合构造结果。

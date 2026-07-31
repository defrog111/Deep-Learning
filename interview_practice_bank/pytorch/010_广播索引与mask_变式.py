"""
题目 010：广播索引与mask_变式

要求：完成“广播索引与mask”的变式题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
matrix = torch.arange(12).reshape(3, 4)  # 创建3×4 Tensor。
bias = torch.tensor([10, 20, 30, 40])  # 创建可广播的列偏置。
shifted = matrix + bias  # 广播完成逐列加法。
mask = shifted.gt(14)  # 创建布尔mask。
selected = shifted[mask]  # 使用布尔索引提取一维结果。
gathered = shifted.gather(1, torch.tensor([[0], [1], [2]]))  # 每行按不同列索引收集元素。
assert gathered.shape == (3, 1)  # 验证gather输出shape。
print(selected, gathered)  # 输出索引结果。

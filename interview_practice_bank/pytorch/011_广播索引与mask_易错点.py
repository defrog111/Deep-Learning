"""
题目 011：广播索引与mask_易错点

要求：完成“广播索引与mask”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建3×4 Tensor。
2. 创建可广播的列偏置。
3. 广播完成逐列加法。
4. 创建布尔mask。
5. 使用布尔索引提取一维结果。
6. 每行按不同列索引收集元素。

完成标准：
- 验证gather输出shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
matrix = torch.arange(12).reshape(3, 4)  # 创建3×4 Tensor。
bias = torch.tensor([10, 20, 30, 40])  # 创建可广播的列偏置。
shifted = matrix + bias  # 广播完成逐列加法。
mask = shifted.gt(15)  # 创建布尔mask。
selected = shifted[mask]  # 使用布尔索引提取一维结果。
gathered = shifted.gather(1, torch.tensor([[0], [1], [2]]))  # 每行按不同列索引收集元素。
assert gathered.shape == (3, 1)  # 验证gather输出shape。
print(selected, gathered)  # 输出索引结果。

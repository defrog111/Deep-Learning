"""
题目 012：广播索引与mask_综合

要求：完成“广播索引与mask”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. scatter是gather的写入对应操作。
3. 验证one-hot式写入。

完成标准：
- 验证one-hot式写入。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
scatter_target = torch.zeros(2, 4); scatter_index = torch.tensor([[1], [3]]); scattered = scatter_target.scatter(1, scatter_index, 1.0)  # scatter是gather的写入对应操作。
assert scattered.tolist() == [[0, 1, 0, 0], [0, 0, 0, 1]]  # 验证one-hot式写入。

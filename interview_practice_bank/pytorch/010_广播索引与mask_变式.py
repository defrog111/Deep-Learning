"""
题目 010：广播索引与mask_变式

要求：完成“广播索引与mask”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. gather按指定轴和同shape索引取值。
3. 验证逐行选择。

完成标准：
- 验证逐行选择。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
mask_values = torch.arange(12).reshape(3, 4); row_index = torch.tensor([[0], [2], [1]]); gathered = torch.gather(mask_values, 1, row_index)  # gather按指定轴和同shape索引取值。
assert gathered.squeeze(1).tolist() == [0, 6, 9]  # 验证逐行选择。

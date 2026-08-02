"""
题目 011：广播索引与mask_易错点

要求：完成“广播索引与mask”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. Attention中常用负无穷屏蔽无效位置。
3. 验证被mask位置概率为零。

完成标准：
- 验证被mask位置概率为零。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
masked_logits = torch.tensor([[1.0, 2.0, 3.0]]); invalid = torch.tensor([[False, True, False]]); safe_logits = masked_logits.masked_fill(invalid, float('-inf'))  # Attention中常用负无穷屏蔽无效位置。
assert torch.softmax(safe_logits, dim=-1)[0, 1] == 0  # 验证被mask位置概率为零。

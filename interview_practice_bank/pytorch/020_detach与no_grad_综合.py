"""
题目 020：detach与no_grad_综合

要求：完成“detach与no_grad”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 综合控制只有一条分支回传梯度。
3. detach分支不贡献额外梯度3。

完成标准：
- detach分支不贡献额外梯度3。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
trainable = torch.tensor(2.0, requires_grad=True); frozen_branch = (trainable * 3).detach(); combined = trainable**2 + frozen_branch; combined.backward()  # 综合控制只有一条分支回传梯度。
assert trainable.grad.item() == 4  # detach分支不贡献额外梯度3。

"""
题目 020：detach与no_grad_综合

要求：完成“detach与no_grad”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建模型参数。
2. 默认建立计算图。
3. detach与原Tensor共享数据但切断梯度。
4. 在推理上下文中关闭梯度记录。
5. 该运算不会构建计算图。
6. 综合控制只有一条分支回传梯度。
7. detach分支不贡献额外梯度3。

完成标准：
- 验证三种梯度状态。
- detach分支不贡献额外梯度3。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
parameter = torch.tensor([2.0], requires_grad=True)  # 创建模型参数。
tracked = parameter * 3  # 默认建立计算图。
detached = tracked.detach()  # detach与原Tensor共享数据但切断梯度。
with torch.no_grad():  # 在推理上下文中关闭梯度记录。
    prediction = parameter * 4  # 该运算不会构建计算图。
assert tracked.requires_grad and not detached.requires_grad and not prediction.requires_grad  # 验证三种梯度状态。
print(tracked, detached, prediction)  # 输出不同状态Tensor。
trainable = torch.tensor(2.0, requires_grad=True); frozen_branch = (trainable * 3).detach(); combined = trainable**2 + frozen_branch; combined.backward()  # 综合控制只有一条分支回传梯度。
assert trainable.grad.item() == 4  # detach分支不贡献额外梯度3。

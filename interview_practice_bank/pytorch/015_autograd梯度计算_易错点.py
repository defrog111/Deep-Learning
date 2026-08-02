"""
题目 015：Autograd梯度计算_易错点

要求：完成“Autograd梯度计算”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 非叶张量默认不保存grad，需retain_grad。
3. 区分叶子与非叶梯度。

完成标准：
- 区分叶子与非叶梯度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
leaf = torch.ones(2, requires_grad=True); non_leaf = leaf * 2; non_leaf.retain_grad(); non_leaf.sum().backward()  # 非叶张量默认不保存grad，需retain_grad。
assert leaf.grad.tolist() == [2, 2] and non_leaf.grad.tolist() == [1, 1]  # 区分叶子与非叶梯度。

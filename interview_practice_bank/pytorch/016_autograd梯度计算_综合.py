"""
题目 016：Autograd梯度计算_综合

要求：完成“Autograd梯度计算”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 计算向量函数Jacobian。
3. 验证Jacobian。

完成标准：
- 验证Jacobian。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
jacobian = torch.autograd.functional.jacobian(lambda value: torch.stack([value[0]**2, value[0] * value[1]]), torch.tensor([2.0, 3.0]))  # 计算向量函数Jacobian。
assert jacobian.shape == (2, 2) and torch.allclose(jacobian, torch.tensor([[4.0, 0.0], [3.0, 2.0]]))  # 验证Jacobian。

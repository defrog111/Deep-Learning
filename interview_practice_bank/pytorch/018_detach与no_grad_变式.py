"""
题目 018：detach与no_grad_变式

要求：完成“detach与no_grad”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. detach切断图但可能共享存储，clone再创建独立副本。
3. 验证存储和梯度均独立。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
inference_input = torch.ones(2, requires_grad=True); detached_clone = inference_input.detach().clone()  # detach切断图但可能共享存储，clone再创建独立副本。
detached_clone.add_(1); assert inference_input.tolist() == [1, 1] and detached_clone.requires_grad is False  # 验证存储和梯度均独立。

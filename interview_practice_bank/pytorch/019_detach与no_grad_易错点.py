"""
题目 019：detach与no_grad_易错点

要求：完成“detach与no_grad”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. inference_mode比no_grad进一步关闭版本跟踪，适合纯推理。
    inference_output = (torch.ones(3) * 2).sum()  # 执行不建图计算。
3. 验证推理张量不跟踪梯度。

完成标准：
- 验证推理张量不跟踪梯度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
with torch.inference_mode():  # inference_mode比no_grad进一步关闭版本跟踪，适合纯推理。
    inference_output = (torch.ones(3) * 2).sum()  # 执行不建图计算。
assert inference_output.requires_grad is False  # 验证推理张量不跟踪梯度。

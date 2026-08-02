"""
题目 099：指标计算与混淆矩阵_易错点

要求：完成“指标计算与混淆矩阵”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 无预测类别的precision分母需防零。
3. 验证零除防护。

完成标准：
- 验证零除防护。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
zero_confusion = torch.tensor([[5, 0], [0, 0]], dtype=torch.float32); safe_precision = zero_confusion.diag() / zero_confusion.sum(0).clamp_min(1)  # 无预测类别的precision分母需防零。
assert torch.isfinite(safe_precision).all() and safe_precision[1] == 0  # 验证零除防护。

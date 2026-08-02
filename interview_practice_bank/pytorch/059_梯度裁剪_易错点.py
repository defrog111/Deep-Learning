"""
题目 059：梯度裁剪_易错点

要求：完成“梯度裁剪”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. 构造非有限梯度。
4. 要求裁剪器遇到非有限范数时报错。
    nn.utils.clip_grad_norm_([bad_parameter], 1.0, error_if_nonfinite=True)  # 开启严格检查。
except RuntimeError:  # 捕获预期错误。
    nonfinite_detected = True  # 记录检测成功。
5. 验证NaN梯度防护。

完成标准：
- 验证NaN梯度防护。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
bad_parameter = nn.Parameter(torch.tensor(1.0)); bad_parameter.grad = torch.tensor(float('nan'))  # 构造非有限梯度。
try:  # 要求裁剪器遇到非有限范数时报错。
    nn.utils.clip_grad_norm_([bad_parameter], 1.0, error_if_nonfinite=True)  # 开启严格检查。
except RuntimeError:  # 捕获预期错误。
    nonfinite_detected = True  # 记录检测成功。
assert nonfinite_detected  # 验证NaN梯度防护。

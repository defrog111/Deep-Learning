"""
题目 051：模型保存与加载_易错点

要求：完成“模型保存与加载”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入路径工具。
2. 导入临时目录工具。
3. 导入 PyTorch。
4. 导入模型模块。
5. shape不同的同名权重即使strict=False也会报尺寸不匹配。
6. 在加载前显式审计参数shape。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import tempfile  # 导入临时目录工具。
import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
source_model = nn.Linear(2, 1); incompatible_model = nn.Linear(3, 1)  # shape不同的同名权重即使strict=False也会报尺寸不匹配。
shape_mismatch = source_model.weight.shape != incompatible_model.weight.shape; assert shape_mismatch  # 在加载前显式审计参数shape。

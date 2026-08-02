"""
题目 056：向量点积与矩阵乘法_综合

要求：完成“向量点积与矩阵乘法”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. tensordot显式指定收缩轴。
3. 验证张量收缩shape。

完成标准：
- 验证张量收缩shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
contracted = np.tensordot(np.arange(24).reshape(2, 3, 4), np.ones((4, 5)), axes=([2], [0]))  # tensordot显式指定收缩轴。
assert contracted.shape == (2, 3, 5)  # 验证张量收缩shape。

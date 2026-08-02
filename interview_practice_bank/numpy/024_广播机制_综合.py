"""
题目 024：广播机制_综合

要求：完成“广播机制”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 用einsum完成批量加权求和。
3. 验证综合广播运算shape。

完成标准：
- 验证综合广播运算shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
batch = np.arange(24).reshape(2, 3, 4); weights = np.arange(4); weighted = np.einsum('bij,j->bi', batch, weights)  # 用einsum完成批量加权求和。
assert weighted.shape == (2, 3)  # 验证综合广播运算shape。

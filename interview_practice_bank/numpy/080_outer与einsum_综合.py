"""
题目 080：outer与einsum_综合

要求：完成“outer与einsum”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. einsum计算每个batch的外积。
3. 验证批量外积。

完成标准：
- 验证批量外积。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
batch_outer = np.einsum('bi,bj->bij', np.ones((3, 2)), np.arange(12).reshape(3, 4))  # einsum计算每个batch的外积。
assert batch_outer.shape == (3, 2, 4)  # 验证批量外积。

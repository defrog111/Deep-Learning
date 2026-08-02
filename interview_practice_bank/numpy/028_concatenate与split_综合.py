"""
题目 028：concatenate与split_综合

要求：完成“concatenate与split”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建左侧矩阵。
2. 创建右侧矩阵。
3. 使用block按二维布局组合小矩阵。

完成标准：
- 验证分块矩阵shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
left = np.arange(6).reshape(2, 3)  # 创建左侧矩阵。
right = np.arange(6, 12).reshape(2, 3)  # 创建右侧矩阵。
blocked = np.block([[left, right], [right, left]])  # 使用block按二维布局组合小矩阵。
assert blocked.shape == (4, 6)  # 验证分块矩阵shape。

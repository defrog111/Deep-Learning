"""
题目 004：数组创建dtype与shape_综合

要求：完成“数组创建dtype与shape”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建指定shape和dtype的零矩阵。
2. 继承zeros的shape和dtype创建全1矩阵。
3. 创建指定填充值矩阵。
4. 创建三阶单位矩阵。
5. 一次创建每个位置的行列坐标矩阵。
6. 沿新batch轴堆叠三种矩阵。

完成标准：
- 验证综合构造shape和单位阵。
- 验证坐标网格shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
zeros = np.zeros((2, 3), dtype=np.float64)  # 创建指定shape和dtype的零矩阵。
ones = np.ones_like(zeros)  # 继承zeros的shape和dtype创建全1矩阵。
filled = np.full((2, 3), fill_value=7.5)  # 创建指定填充值矩阵。
identity = np.eye(3, dtype=np.float64)  # 创建三阶单位矩阵。
row_grid, column_grid = np.indices((2, 3))  # 一次创建每个位置的行列坐标矩阵。
stacked = np.stack((zeros, ones, filled), axis=0)  # 沿新batch轴堆叠三种矩阵。
assert stacked.shape == (3, 2, 3) and np.trace(identity) == 3  # 验证综合构造shape和单位阵。
assert row_grid.shape == column_grid.shape == (2, 3)  # 验证坐标网格shape。
print(stacked, identity, row_grid, column_grid)  # 输出综合构造结果。

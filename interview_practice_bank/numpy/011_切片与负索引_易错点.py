"""
题目 011：切片与负索引_易错点

要求：完成“切片与负索引”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建4×4矩阵。
3. 基本切片是视图，修改它会影响原数组。
4. 验证共享内存陷阱。

完成标准：
- 验证共享内存陷阱。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.arange(1, 17).reshape(4, 4)  # 创建4×4矩阵。
slice_view = matrix[:, :2]; slice_view[0, 0] = -1  # 基本切片是视图，修改它会影响原数组。
assert matrix[0, 0] == -1 and np.shares_memory(matrix, slice_view)  # 验证共享内存陷阱。

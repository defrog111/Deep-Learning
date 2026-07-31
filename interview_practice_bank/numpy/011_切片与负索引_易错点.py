"""
题目 011：切片与负索引_易错点

要求：完成“切片与负索引”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建4×4矩阵。
2. 同时切片行和列取得中心块。
3. 使用负步长翻转行顺序。
4. 在展平视图上按步长抽样。
5. 基本切片是视图，修改它会影响原数组。

完成标准：
- 验证二维切片shape。
- 验证共享内存陷阱。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.arange(1, 17).reshape(4, 4)  # 创建4×4矩阵。
center = matrix[1:3, 1:3]  # 同时切片行和列取得中心块。
reversed_rows = matrix[::-1]  # 使用负步长翻转行顺序。
every_n = matrix.ravel()[::3]  # 在展平视图上按步长抽样。
assert center.shape == (2, 2)  # 验证二维切片shape。
print(center, '\n', reversed_rows, '\n', every_n)  # 输出切片结果。
slice_view = matrix[:, :2]; slice_view[0, 0] = -1  # 基本切片是视图，修改它会影响原数组。
assert matrix[0, 0] == -1 and np.shares_memory(matrix, slice_view)  # 验证共享内存陷阱。

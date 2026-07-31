"""
题目 012：切片与负索引_综合

要求：完成“切片与负索引”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建4×4矩阵。
2. 同时切片行和列取得中心块。
3. 使用负步长翻转行顺序。
4. 在展平视图上按步长抽样。
5. 综合使用take按轴选择并用clip限制范围。

完成标准：
- 验证二维切片shape。
- 验证选择和裁剪结果。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.arange(1, 17).reshape(4, 4)  # 创建4×4矩阵。
center = matrix[1:3, 1:3]  # 同时切片行和列取得中心块。
reversed_rows = matrix[::-1]  # 使用负步长翻转行顺序。
every_n = matrix.ravel()[::4]  # 在展平视图上按步长抽样。
assert center.shape == (2, 2)  # 验证二维切片shape。
print(center, '\n', reversed_rows, '\n', every_n)  # 输出切片结果。
picked = np.take(matrix, [0, -1], axis=0); clipped = matrix.clip(min=0)  # 综合使用take按轴选择并用clip限制范围。
assert picked.shape[0] == 2 and clipped.min() >= 0  # 验证选择和裁剪结果。

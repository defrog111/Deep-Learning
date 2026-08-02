"""
题目 012：切片与负索引_综合

要求：完成“切片与负索引”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建4×4矩阵。
3. 综合使用take按轴选择并用clip限制范围。
4. 验证选择和裁剪结果。

完成标准：
- 验证选择和裁剪结果。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.arange(1, 17).reshape(4, 4)  # 创建4×4矩阵。
picked = np.take(matrix, [0, -1], axis=0); clipped = matrix.clip(min=0)  # 综合使用take按轴选择并用clip限制范围。
assert picked.shape[0] == 2 and clipped.min() >= 0  # 验证选择和裁剪结果。

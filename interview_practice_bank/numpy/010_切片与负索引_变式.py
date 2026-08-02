"""
题目 010：切片与负索引_变式

要求：完成“切片与负索引”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建4×4矩阵。
3. 使用省略号和newaxis在末尾增加维度。
4. 验证新增长度为1的轴。

完成标准：
- 验证新增长度为1的轴。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.arange(1, 17).reshape(4, 4)  # 创建4×4矩阵。
expanded = matrix[..., np.newaxis]  # 使用省略号和newaxis在末尾增加维度。
assert expanded.shape == matrix.shape + (1,)  # 验证新增长度为1的轴。

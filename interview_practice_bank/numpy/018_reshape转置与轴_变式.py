"""
题目 018：reshape转置与轴_变式

要求：完成“reshape转置与轴”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建24个元素。
2. 重塑为三维数组。
3. 使用moveaxis和swapaxes改变轴顺序。

完成标准：
- 验证轴操作。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
array = np.arange(24)  # 创建24个元素。
cube = array.reshape(2, 3, 4)  # 重塑为三维数组。
moved = np.moveaxis(cube, 0, -1); swapped = np.swapaxes(cube, 0, 1)  # 使用moveaxis和swapaxes改变轴顺序。
assert moved.shape == (3, 4, 2) and swapped.shape == (3, 2, 4)  # 验证轴操作。

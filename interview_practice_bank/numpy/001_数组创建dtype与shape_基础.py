"""
题目 001：数组创建dtype与shape_基础

要求：完成“数组创建dtype与shape”的基础题，写出关键数组的shape并解释结果。

操作步骤：
1. 从嵌套列表创建二维float32数组。
2. 检查行列shape和维数。
3. 检查元素数和每个float32元素字节数。

完成标准：
- 检查行列shape和维数。
- 检查元素数和每个float32元素字节数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # 从嵌套列表创建二维float32数组。
assert matrix.shape == (2, 3) and matrix.ndim == 2  # 检查行列shape和维数。
assert matrix.size == 6 and matrix.itemsize == 4  # 检查元素数和每个float32元素字节数。
print(matrix, matrix.dtype, matrix.shape, matrix.nbytes)  # 输出基础数组元数据。

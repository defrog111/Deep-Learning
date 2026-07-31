"""
题目 004：数组创建dtype与shape_综合

要求：完成“数组创建dtype与shape”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建二维浮点数组。
2. 显式转换dtype并产生新数组。
3. 综合使用zeros、ones、full和stack构造矩阵。
4. 同时复习eye单位矩阵。

完成标准：
- 验证shape和维数。
- 同时复习eye单位矩阵。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
array = np.array([[1, 2, 6], [4, 5, 6]], dtype=np.float32)  # 创建二维浮点数组。
converted = array.astype(np.int64)  # 显式转换dtype并产生新数组。
assert array.shape == (2, 3) and array.ndim == 2  # 验证shape和维数。
print(array.dtype, converted.dtype, array.size, array.itemsize)  # 查看类型、元素数和单元素字节数。
constructed = np.stack([np.zeros(3), np.ones(3), np.full(3, 2)])  # 综合使用zeros、ones、full和stack构造矩阵。
assert constructed.shape == (3, 3) and np.eye(3).trace() == 3  # 同时复习eye单位矩阵。

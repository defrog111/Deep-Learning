"""
题目 003：数组创建dtype与shape_易错点

要求：完成“数组创建dtype与shape”的易错点题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
array = np.array([[1, 2, 5], [4, 5, 6]], dtype=np.float32)  # 创建二维浮点数组。
converted = array.astype(np.int64)  # 显式转换dtype并产生新数组。
assert array.shape == (2, 3) and array.ndim == 2  # 验证shape和维数。
print(array.dtype, converted.dtype, array.size, array.itemsize)  # 查看类型、元素数和单元素字节数。

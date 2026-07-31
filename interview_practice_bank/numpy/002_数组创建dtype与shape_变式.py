"""
题目 002：数组创建dtype与shape_变式

要求：完成“数组创建dtype与shape”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建二维浮点数组。
2. 显式转换dtype并产生新数组。
3. 使用asarray接收已有序列，输入已是数组时通常避免复制。

完成标准：
- 验证shape和维数。
- 验证整数序列推导为整数dtype。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
array = np.array([[1, 2, 4], [4, 5, 6]], dtype=np.float32)  # 创建二维浮点数组。
converted = array.astype(np.int64)  # 显式转换dtype并产生新数组。
assert array.shape == (2, 3) and array.ndim == 2  # 验证shape和维数。
print(array.dtype, converted.dtype, array.size, array.itemsize)  # 查看类型、元素数和单元素字节数。
from_list = np.asarray([1, 2, 3])  # 使用asarray接收已有序列，输入已是数组时通常避免复制。
assert from_list.dtype.kind in 'iu'  # 验证整数序列推导为整数dtype。

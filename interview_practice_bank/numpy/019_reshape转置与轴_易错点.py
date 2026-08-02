"""
题目 019：reshape转置与轴_易错点

要求：完成“reshape转置与轴”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建24个元素。
2. 重塑为三维数组。
3. 转置后先连续化可避免布局误解。

完成标准：
- 验证结果采用C连续布局。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
array = np.arange(24)  # 创建24个元素。
cube = array.reshape(2, 3, 4)  # 重塑为三维数组。
non_contiguous = cube.transpose(2, 1, 0); safe_shape = np.ascontiguousarray(non_contiguous).reshape(4, -1)  # 转置后先连续化可避免布局误解。
assert safe_shape.flags.c_contiguous  # 验证结果采用C连续布局。

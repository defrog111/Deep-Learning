"""
题目 019：reshape转置与轴_易错点

要求：完成“reshape转置与轴”的易错点题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
array = np.arange(24)  # 创建24个元素。
cube = array.reshape(2, 3, 4)  # 重塑为三维数组。
permuted = cube.transpose(2, 0, 1)  # 把轴顺序由(0,1,2)改为(2,0,1)。
flattened = permuted.reshape(-1)  # 使用-1自动推导展平长度。
assert permuted.shape == (4, 2, 3) and flattened.size == 24  # 验证轴变换只改布局不改元素数。
print(cube.strides, permuted.strides)  # 查看转置前后的内存步幅。

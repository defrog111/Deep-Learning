"""
题目 019：reshape转置与轴_易错点

要求：完成“reshape转置与轴”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建24个元素。
2. 重塑为三维数组。
3. 把轴顺序由(0,1,2)改为(2,0,1)。
4. 使用-1自动推导展平长度。

完成标准：
- 验证轴变换只改布局不改元素数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
array = np.arange(24)  # 创建24个元素。
cube = array.reshape(2, 3, 4)  # 重塑为三维数组。
permuted = cube.transpose(2, 0, 1)  # 把轴顺序由(0,1,2)改为(2,0,1)。
flattened = permuted.reshape(-1)  # 使用-1自动推导展平长度。
assert permuted.shape == (4, 2, 3) and flattened.size == 24  # 验证轴变换只改布局不改元素数。
print(cube.strides, permuted.strides)  # 查看转置前后的内存步幅。

"""
题目 020：reshape转置与轴_综合

要求：完成“reshape转置与轴”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建24个元素。
2. 重塑为三维数组。
3. 把轴顺序由(0,1,2)改为(2,0,1)。
4. 使用-1自动推导展平长度。
5. ravel尽量返回视图，flatten总是复制。
6. 综合比较reshape、ravel和flatten。

完成标准：
- 验证轴变换只改布局不改元素数。
- 综合比较reshape、ravel和flatten。
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
flat_view = cube.ravel(); flat_copy = cube.flatten(); flat_view[0] = -1  # ravel尽量返回视图，flatten总是复制。
assert cube.ravel()[0] == -1 and flat_copy[0] != -1  # 综合比较reshape、ravel和flatten。

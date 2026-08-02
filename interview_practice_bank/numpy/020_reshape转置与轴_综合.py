"""
题目 020：reshape转置与轴_综合

要求：完成“reshape转置与轴”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建24个元素。
2. 重塑为三维数组。
3. ravel尽量返回视图，flatten总是复制。
4. 综合比较reshape、ravel和flatten。

完成标准：
- 综合比较reshape、ravel和flatten。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
array = np.arange(24)  # 创建24个元素。
cube = array.reshape(2, 3, 4)  # 重塑为三维数组。
flat_view = cube.ravel(); flat_copy = cube.flatten(); flat_view[0] = -1  # ravel尽量返回视图，flatten总是复制。
assert cube.ravel()[0] == -1 and flat_copy[0] != -1  # 综合比较reshape、ravel和flatten。

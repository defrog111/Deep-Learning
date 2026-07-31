"""
题目 032：聚合函数与axis_综合

要求：完成“聚合函数与axis”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建三维数组。
2. 消去batch轴得到shape(3,4)。
3. 消去最后轴但保留长度1维。
4. 返回每行最大值的位置。
5. 综合使用ufunc的accumulate和reduce。

完成标准：
- 验证keepdims便于后续广播。
- 验证累计和连乘。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
cube = np.arange(24).reshape(2, 3, 4)  # 创建三维数组。
sum_axis0 = cube.sum(axis=0)  # 消去batch轴得到shape(3,4)。
mean_last = cube.mean(axis=-1, keepdims=True)  # 消去最后轴但保留长度1维。
maximum_positions = cube.argmax(axis=2)  # 返回每行最大值的位置。
assert mean_last.shape == (2, 3, 1)  # 验证keepdims便于后续广播。
print(sum_axis0, '\n', mean_last, '\n', maximum_positions)  # 输出聚合结果。
accumulated = np.add.accumulate(np.arange(1, 6)); reduced = np.multiply.reduce(np.arange(1, 6))  # 综合使用ufunc的accumulate和reduce。
assert accumulated[-1] == 15 and reduced == 120  # 验证累计和连乘。

"""
题目 006：arange与linspace_变式

要求：完成“arange与linspace”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 按步长生成左闭右开整数序列。
2. 按点数生成包含终点的等距序列。
3. 计算相邻点间隔。
4. arange三种等价常用写法。

完成标准：
- 验证linspace等距。
- 验证三种参数形式结果一致。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
integers = np.arange(0, 16, 4)  # 按步长生成左闭右开整数序列。
points = np.linspace(0.0, 1.0, num=6, endpoint=True)  # 按点数生成包含终点的等距序列。
differences = np.diff(points)  # 计算相邻点间隔。
assert np.allclose(differences, differences[0])  # 验证linspace等距。
print(integers, points)  # 输出两种序列。
range_short = np.arange(100); range_full = np.arange(0, 100); range_keywords = np.arange(start=0, stop=100, step=1)  # arange三种等价常用写法。
assert np.array_equal(range_short, range_full) and np.array_equal(range_full, range_keywords)  # 验证三种参数形式结果一致。

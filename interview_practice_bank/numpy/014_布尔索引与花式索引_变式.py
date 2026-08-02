"""
题目 014：布尔索引与花式索引_变式

要求：完成“布尔索引与花式索引”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 用独立成绩数据练习take和flatnonzero。
3. 验证按位置重排和条件位置提取。

完成标准：
- 验证按位置重排和条件位置提取。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
scores = np.array([72, 95, 81, 95, 60]); positions = np.array([3, 1, 2]); taken = np.take(scores, positions); high_positions = np.flatnonzero(scores >= 90)  # 用独立成绩数据练习take和flatnonzero。
assert taken.tolist() == [95, 95, 81] and high_positions.tolist() == [1, 3]  # 验证按位置重排和条件位置提取。

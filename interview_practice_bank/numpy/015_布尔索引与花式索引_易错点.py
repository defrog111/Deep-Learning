"""
题目 015：布尔索引与花式索引_易错点

要求：完成“布尔索引与花式索引”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建一维数组。
2. 使用布尔mask筛选满足阈值的元素。
3. 定义花式索引顺序。
4. 花式索引总是返回副本。
5. 重复花式索引的原地累加只写回一次，是高频陷阱。
6. 如需累计重复位置应使用np.add.at。

完成标准：
- 验证按指定位置重排。
- 如需累计重复位置应使用np.add.at。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
array = np.array([3, 8, 1, 9, 4, 7])  # 创建一维数组。
selected = array[array >= 5]  # 使用布尔mask筛选满足阈值的元素。
order = np.array([3, 0, 5])  # 定义花式索引顺序。
reordered = array[order]  # 花式索引总是返回副本。
assert reordered.tolist() == [9, 3, 7]  # 验证按指定位置重排。
print(selected, reordered)  # 输出两种索引结果。
repeated = np.zeros(3, dtype=int); repeated[[0, 0]] += 1  # 重复花式索引的原地累加只写回一次，是高频陷阱。
assert repeated[0] == 1  # 如需累计重复位置应使用np.add.at。

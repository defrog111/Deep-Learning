"""
题目 016：布尔索引与花式索引_综合

要求：完成“布尔索引与花式索引”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建一维数组。
2. 使用布尔mask筛选满足阈值的元素。
3. 定义花式索引顺序。
4. 花式索引总是返回副本。
5. 使用ix_取得行列笛卡尔积。
6. 区分成对花式索引和网格索引。

完成标准：
- 验证按指定位置重排。
- 区分成对花式索引和网格索引。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
array = np.array([3, 8, 1, 9, 4, 7])  # 创建一维数组。
selected = array[array >= 6]  # 使用布尔mask筛选满足阈值的元素。
order = np.array([3, 0, 5])  # 定义花式索引顺序。
reordered = array[order]  # 花式索引总是返回副本。
assert reordered.tolist() == [9, 3, 7]  # 验证按指定位置重排。
print(selected, reordered)  # 输出两种索引结果。
rows_ix, cols_ix = np.ix_([0, 2], [1, 3]); cross = np.arange(12).reshape(3, 4)[rows_ix, cols_ix]  # 使用ix_取得行列笛卡尔积。
assert cross.shape == (2, 2)  # 区分成对花式索引和网格索引。

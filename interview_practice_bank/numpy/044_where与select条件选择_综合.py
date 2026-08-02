"""
题目 044：where与select条件选择_综合

要求：完成“where与select条件选择”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. choose按整数选择器从多组候选取值。
3. 验证多分支选择。

完成标准：
- 验证多分支选择。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
choices = np.choose(np.array([0, 1, 2]), [np.full(3, 10), np.full(3, 20), np.full(3, 30)])  # choose按整数选择器从多组候选取值。
assert choices.tolist() == [10, 20, 30]  # 验证多分支选择。

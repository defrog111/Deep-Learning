"""
题目 065：MultiIndex多级索引_基础

要求：完成“MultiIndex多级索引”的基础题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 创建多级索引。
2. 构造MultiIndex表。
3. xs按某一级快速截面选择。
4. 把year级别旋转到列。
5. stack恢复多级行索引。

完成标准：
- 验证stack/unstack可逆。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
index = pd.MultiIndex.from_product([['A', 'B'], [2024, 2025]], names=['team', 'year'])  # 创建多级索引。
frame = pd.DataFrame({'sales': [10, 12, 20, 25]}, index=index)  # 构造MultiIndex表。
team_a = frame.xs('A', level='team')  # xs按某一级快速截面选择。
unstacked = frame.unstack('year')  # 把year级别旋转到列。
restored = unstacked.stack('year', future_stack=True).sort_index()  # stack恢复多级行索引。
assert restored.equals(frame)  # 验证stack/unstack可逆。
print(team_a, '\n', unstacked)  # 输出截面和宽表。

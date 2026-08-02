"""
题目 083：itertools组合迭代_易错点

要求：完成“itertools组合迭代”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入组合迭代工具。
2. 导入相邻分组工具。
3. groupby前必须按同一key排序。
4. 验证连续分组。

完成标准：
- 验证连续分组。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from itertools import chain, combinations, groupby, islice  # 导入组合迭代工具。
from itertools import groupby  # 导入相邻分组工具。
records = [('b', 1), ('a', 2), ('b', 3)]; grouped_records = {key: list(group) for key, group in groupby(sorted(records), key=lambda row: row[0])}  # groupby前必须按同一key排序。
assert [row[1] for row in grouped_records['b']] == [1, 3]  # 验证连续分组。

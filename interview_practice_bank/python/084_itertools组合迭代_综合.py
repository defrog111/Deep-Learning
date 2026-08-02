"""
题目 084：itertools组合迭代_综合

要求：完成“itertools组合迭代”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入组合迭代工具。
2. 导入笛卡尔积、组合和排列。
3. 综合验证三个计数公式。

完成标准：
- 综合验证三个计数公式。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from itertools import chain, combinations, groupby, islice  # 导入组合迭代工具。
from itertools import product, combinations, permutations  # 导入笛卡尔积、组合和排列。
assert len(list(product(range(2), repeat=3))) == 8 and len(list(combinations(range(4), 2))) == 6 and len(list(permutations(range(3)))) == 6  # 综合验证三个计数公式。

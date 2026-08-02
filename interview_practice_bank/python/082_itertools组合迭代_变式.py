"""
题目 082：itertools组合迭代_变式

要求：完成“itertools组合迭代”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入组合迭代工具。
2. 导入惰性连接和切片。
3. 不构建完整中间列表取得前五项。
4. 验证惰性流水线。

完成标准：
- 验证惰性流水线。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from itertools import chain, combinations, groupby, islice  # 导入组合迭代工具。
from itertools import chain, islice  # 导入惰性连接和切片。
first_five = list(islice(chain(range(3), range(3, 10)), 5))  # 不构建完整中间列表取得前五项。
assert first_five == [0, 1, 2, 3, 4]  # 验证惰性流水线。

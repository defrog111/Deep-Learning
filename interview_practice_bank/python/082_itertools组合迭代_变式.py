"""
题目 082：itertools组合迭代_变式

要求：完成“itertools组合迭代”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建已按分组键排序的序列。
2. groupby只合并相邻相同键。
3. 生成不重复二元组合。
4. 惰性连接多个序列。
5. 从惰性迭代器取前n项。
6. 不构建完整中间列表取得前五项。

完成标准：
- 验证组合数和展平。
- 验证惰性流水线。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from itertools import chain, combinations, groupby, islice  # 导入组合迭代工具。
values = [1, 1, 2, 2, 3]  # 创建已按分组键排序的序列。
groups = {key: list(group) for key, group in groupby(values)}  # groupby只合并相邻相同键。
pairs = list(combinations([1, 2, 3, 4], 2))  # 生成不重复二元组合。
flattened = list(chain.from_iterable([[1, 2], [3], [4, 5]]))  # 惰性连接多个序列。
prefix = list(islice(range(100), 4))  # 从惰性迭代器取前n项。
assert len(pairs) == 6 and flattened == [1, 2, 3, 4, 5]  # 验证组合数和展平。
print(groups, pairs, prefix)  # 输出itertools结果。
from itertools import chain, islice  # 导入惰性连接和切片。
first_five = list(islice(chain(range(3), range(3, 10)), 5))  # 不构建完整中间列表取得前五项。
assert first_five == [0, 1, 2, 3, 4]  # 验证惰性流水线。

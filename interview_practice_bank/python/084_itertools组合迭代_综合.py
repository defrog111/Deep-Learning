"""
题目 084：itertools组合迭代_综合

要求：完成“itertools组合迭代”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

from itertools import chain, combinations, groupby, islice  # 导入组合迭代工具。
values = [1, 1, 2, 2, 3]  # 创建已按分组键排序的序列。
groups = {key: list(group) for key, group in groupby(values)}  # groupby只合并相邻相同键。
pairs = list(combinations([1, 2, 3, 4], 2))  # 生成不重复二元组合。
flattened = list(chain.from_iterable([[1, 2], [3], [4, 5]]))  # 惰性连接多个序列。
prefix = list(islice(range(100), 6))  # 从惰性迭代器取前n项。
assert len(pairs) == 6 and flattened == [1, 2, 3, 4, 5]  # 验证组合数和展平。
print(groups, pairs, prefix)  # 输出itertools结果。

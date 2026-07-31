"""
题目 008：列表切片与步长_综合

要求：完成“列表切片与步长”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建0到9的列表。
2. 使用起点终点步长切片。
3. 使用负步长反转列表。
4. 使用切片循环左移。
5. 全切片创建浅拷贝。
6. 对迭代器使用islice而不是下标切片。

完成标准：
- 验证值相等但对象不同。
- 验证惰性与列表切片结果一致。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

values = list(range(10))  # 创建0到9的列表。
middle = values[2:8:2]  # 使用起点终点步长切片。
reversed_values = values[::-1]  # 使用负步长反转列表。
rotated = values[4:] + values[:4]  # 使用切片循环左移。
copy = values[:]  # 全切片创建浅拷贝。
assert copy == values and copy is not values  # 验证值相等但对象不同。
print(middle, reversed_values, rotated)  # 输出切片结果。
from itertools import islice  # 导入惰性切片工具。
lazy_slice = list(islice(iter(values), 1, None, 2))  # 对迭代器使用islice而不是下标切片。
assert lazy_slice == values[1::2]  # 验证惰性与列表切片结果一致。

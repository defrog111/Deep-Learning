"""
题目 006：列表切片与步长_变式

要求：完成“列表切片与步长”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建0到9的列表。
2. 使用起点终点步长切片。
3. 使用负步长反转列表。
4. 使用切片循环左移。
5. 全切片创建浅拷贝。
6. slice对象可复用并提高复杂切片可读性。

完成标准：
- 验证值相等但对象不同。
- 验证slice对象与冒号语法等价。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

values = list(range(10))  # 创建0到9的列表。
middle = values[2:8:2]  # 使用起点终点步长切片。
reversed_values = values[::-1]  # 使用负步长反转列表。
rotated = values[2:] + values[:2]  # 使用切片循环左移。
copy = values[:]  # 全切片创建浅拷贝。
assert copy == values and copy is not values  # 验证值相等但对象不同。
print(middle, reversed_values, rotated)  # 输出切片结果。
window = slice(1, None, 2); sliced_twice = values[window]  # slice对象可复用并提高复杂切片可读性。
assert sliced_twice == values[1::2]  # 验证slice对象与冒号语法等价。

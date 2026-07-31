"""
题目 008：列表切片与步长_综合

要求：完成“列表切片与步长”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

values = list(range(10))  # 创建0到9的列表。
middle = values[2:8:2]  # 使用起点终点步长切片。
reversed_values = values[::-1]  # 使用负步长反转列表。
rotated = values[4:] + values[:4]  # 使用切片循环左移。
copy = values[:]  # 全切片创建浅拷贝。
assert copy == values and copy is not values  # 验证值相等但对象不同。
print(middle, reversed_values, rotated)  # 输出切片结果。

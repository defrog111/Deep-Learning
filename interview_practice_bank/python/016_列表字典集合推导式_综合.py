"""
题目 016：列表字典集合推导式_综合

要求：完成“列表字典集合推导式”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

values = range(1, 8)  # 创建惰性整数序列。
squares = [value**2 for value in values if value % 2 == 0]  # 列表推导筛选偶数平方。
mapping = {value: value**2 for value in values}  # 字典推导创建映射。
remainders = {value % 3 for value in values}  # 集合推导自动去重。
matrix = [[row + column for column in range(3)] for row in range(2)]  # 嵌套推导创建二维列表。
assert squares == [4, 16, 36] and remainders == {0, 1, 2}  # 验证推导结果。
print(squares, mapping, remainders, matrix)  # 输出各种推导式。

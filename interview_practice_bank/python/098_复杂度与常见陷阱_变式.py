"""
题目 098：复杂度与常见陷阱_变式

要求：完成“复杂度与常见陷阱”的变式题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

values = list(range(1000))  # list顺序存储且按位置访问O(1)。
lookup = set(values)  # set平均O(1)成员测试但需额外内存。
target = 40  # 设置查找目标。
in_list = target in values  # list成员测试最坏O(n)。
in_set = target in lookup  # set成员测试平均O(1)。
matrix_bad = [[0] * 3] * 3  # 重复同一个内部列表引用的常见陷阱。
matrix_good = [[0 for _ in range(3)] for _ in range(3)]  # 每行创建独立列表。
matrix_bad[0][0] = 1  # 修改一行会影响所有共享行。
matrix_good[0][0] = 1  # 正确矩阵只修改第一行。
assert in_list == in_set and sum(row[0] for row in matrix_bad) == 3  # 验证复杂度容器和别名陷阱。
print(matrix_bad, matrix_good)  # 输出错误与正确矩阵。

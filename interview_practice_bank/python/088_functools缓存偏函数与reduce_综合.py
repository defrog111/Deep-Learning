"""
题目 088：functools缓存偏函数与reduce_综合

要求：完成“functools缓存偏函数与reduce”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

from functools import cache, partial, reduce  # 导入函数式工具。
@cache  # 缓存纯函数的输入输出。
def fibonacci(number):  # 定义递归斐波那契。
    return number if number < 2 else fibonacci(number - 1) + fibonacci(number - 2)  # 使用缓存避免指数级重复计算。
power_of_two = partial(pow, 2)  # 固定pow的第一个参数创建新函数。
product = reduce(lambda left, right: left * right, range(1, 6), 1)  # 归约计算连乘。
value = fibonacci(11)  # 计算随变式变化的斐波那契数。
assert power_of_two(5) == 32 and product == 120  # 验证partial和reduce。
print(value, fibonacci.cache_info(), product)  # 输出缓存命中信息。

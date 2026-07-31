"""
题目 033：生成器与yield_基础

要求：完成“生成器与yield”的基础题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

def fibonacci(limit):  # 定义生成器函数。
    first, second = 0, 1  # 初始化前两项。
    while first < limit:  # 惰性生成直到上限。
        yield first  # 暂停并产出当前值。
        first, second = second, first + second  # 更新状态。
generator = fibonacci(15)  # 创建生成器但尚未执行函数体。
first_value = next(generator)  # 手动消费第一个值。
remaining = list(generator)  # 消费剩余值用于变式1。
assert first_value == 0 and all(value < 15 for value in remaining)  # 验证边界。
print(first_value, remaining)  # 输出生成序列。

"""
题目 088：functools缓存偏函数与reduce_综合

要求：完成“functools缓存偏函数与reduce”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 缓存纯函数的输入输出。
2. 定义递归斐波那契。
3. 使用缓存避免指数级重复计算。
4. 固定pow的第一个参数创建新函数。
5. 归约计算连乘。
6. 计算随变式变化的斐波那契数。
7. 定义问候类。
    def greet(self, prefix, name):  # 定义通用方法。
        return f'{prefix} {name}'  # 拼接问候。
    hello = partialmethod(greet, 'Hello')  # 派生固定前缀的方法。

完成标准：
- 验证partial和reduce。
- 验证partialmethod绑定self。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
from functools import partialmethod  # 导入类方法参数预绑定工具。
class Greeter:  # 定义问候类。
    def greet(self, prefix, name):  # 定义通用方法。
        return f'{prefix} {name}'  # 拼接问候。
    hello = partialmethod(greet, 'Hello')  # 派生固定前缀的方法。
assert Greeter().hello('Ada') == 'Hello Ada'  # 验证partialmethod绑定self。

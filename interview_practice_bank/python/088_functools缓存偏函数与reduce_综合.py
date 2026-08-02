"""
题目 088：functools缓存偏函数与reduce_综合

要求：完成“functools缓存偏函数与reduce”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入函数式工具。
2. 导入类方法参数预绑定工具。
3. 定义问候类。
    def greet(self, prefix, name):  # 定义通用方法。
        return f'{prefix} {name}'  # 拼接问候。
    hello = partialmethod(greet, 'Hello')  # 派生固定前缀的方法。
4. 验证partialmethod绑定self。

完成标准：
- 验证partialmethod绑定self。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import cache, partial, reduce  # 导入函数式工具。
from functools import partialmethod  # 导入类方法参数预绑定工具。
class Greeter:  # 定义问候类。
    def greet(self, prefix, name):  # 定义通用方法。
        return f'{prefix} {name}'  # 拼接问候。
    hello = partialmethod(greet, 'Hello')  # 派生固定前缀的方法。
assert Greeter().hello('Ada') == 'Hello Ada'  # 验证partialmethod绑定self。

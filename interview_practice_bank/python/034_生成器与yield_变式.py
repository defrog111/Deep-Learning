"""
题目 034：生成器与yield_变式

要求：完成“生成器与yield”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义委托生成器。
    yield from range(3)  # yield from转发子迭代器。
2. 验证yield from。

完成标准：
- 验证yield from。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

def delegated():  # 定义委托生成器。
    yield from range(3)  # yield from转发子迭代器。
assert list(delegated()) == [0, 1, 2]  # 验证yield from。

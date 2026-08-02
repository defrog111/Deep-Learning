"""
题目 038：迭代器协议_变式

要求：完成“迭代器协议”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. iter(callable, sentinel)循环调用直到哨兵值。
2. 验证哨兵迭代形式。

完成标准：
- 验证哨兵迭代形式。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

sentinel_values = iter([1, 2, 0, 3]); until_zero = list(iter(lambda: next(sentinel_values), 0))  # iter(callable, sentinel)循环调用直到哨兵值。
assert until_zero == [1, 2]  # 验证哨兵迭代形式。

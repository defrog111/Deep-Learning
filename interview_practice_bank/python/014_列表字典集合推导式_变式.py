"""
题目 014：列表字典集合推导式_变式

要求：完成“列表字典集合推导式”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 生成器表达式惰性产生结果。
2. 验证一次性消费。

完成标准：
- 验证一次性消费。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

generator_expression = (value * value for value in range(5))  # 生成器表达式惰性产生结果。
assert next(generator_expression) == 0 and list(generator_expression) == [1, 4, 9, 16]  # 验证一次性消费。

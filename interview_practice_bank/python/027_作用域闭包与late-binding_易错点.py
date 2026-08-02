"""
题目 027：作用域闭包与late-binding_易错点

要求：完成“作用域闭包与late-binding”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 闭包延迟查找循环变量，三个函数都看到最终值。
2. 验证late binding陷阱。

完成标准：
- 验证late binding陷阱。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

wrong_functions = [lambda: index for index in range(3)]  # 闭包延迟查找循环变量，三个函数都看到最终值。
assert [function() for function in wrong_functions] == [2, 2, 2]  # 验证late binding陷阱。

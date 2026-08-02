"""
题目 026：作用域闭包与late-binding_变式

要求：完成“作用域闭包与late-binding”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建保存状态的闭包。
    count = 0  # 定义封闭变量。
    def increment():  # 定义内部函数。
        nonlocal count  # 声明修改最近一层封闭作用域。
        count += 1  # 更新闭包状态。
        return count  # 返回新计数。
    return increment  # 返回闭包。
2. 验证nonlocal状态。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

def make_counter():  # 创建保存状态的闭包。
    count = 0  # 定义封闭变量。
    def increment():  # 定义内部函数。
        nonlocal count  # 声明修改最近一层封闭作用域。
        count += 1  # 更新闭包状态。
        return count  # 返回新计数。
    return increment  # 返回闭包。
counter = make_counter(); assert [counter(), counter()] == [1, 2]  # 验证nonlocal状态。

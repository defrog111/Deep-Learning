"""
题目 028：作用域闭包与late-binding_综合

要求：完成“作用域闭包与late-binding”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入偏函数。
2. 固定pow的底数形成新函数。
3. 验证partial参数绑定。

完成标准：
- 验证partial参数绑定。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import partial  # 导入偏函数。
powers_of_two = list(map(partial(pow, 2), range(5)))  # 固定pow的底数形成新函数。
assert powers_of_two == [1, 2, 4, 8, 16]  # 验证partial参数绑定。

"""
题目 044：上下文管理器_综合

要求：完成“上下文管理器”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入生成器式上下文管理器。
2. 导入动态上下文组合工具。
3. 按运行时条件管理多个资源。
    first = stack.enter_context(nullcontext('A'))  # 加入第一个无需清理的上下文。
    second = stack.enter_context(nullcontext('B'))  # 加入第二个上下文。
4. 验证动态上下文组合。

完成标准：
- 验证动态上下文组合。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from contextlib import contextmanager  # 导入生成器式上下文管理器。
from contextlib import ExitStack, nullcontext  # 导入动态上下文组合工具。
with ExitStack() as stack:  # 按运行时条件管理多个资源。
    first = stack.enter_context(nullcontext('A'))  # 加入第一个无需清理的上下文。
    second = stack.enter_context(nullcontext('B'))  # 加入第二个上下文。
assert first + second == 'AB'  # 验证动态上下文组合。

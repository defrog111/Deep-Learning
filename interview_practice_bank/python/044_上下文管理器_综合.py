"""
题目 044：上下文管理器_综合

要求：完成“上下文管理器”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 把生成器函数转换为上下文管理器。
2. 定义资源生命周期。
3. __enter__阶段获取资源。
4. 确保退出逻辑总会执行。
5. 把资源交给with代码块。
6. 正常或异常退出都会进入finally。
7. 释放资源。
8. 记录生命周期。
9. 使用上下文管理器。
10. 在with内部使用资源。

完成标准：
- 验证进入使用退出顺序。
- 验证动态上下文组合。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from contextlib import contextmanager  # 导入生成器式上下文管理器。
@contextmanager  # 把生成器函数转换为上下文管理器。
def managed_resource(events):  # 定义资源生命周期。
    events.append('open')  # __enter__阶段获取资源。
    try:  # 确保退出逻辑总会执行。
        yield 'resource'  # 把资源交给with代码块。
    finally:  # 正常或异常退出都会进入finally。
        events.append('close')  # 释放资源。
events = []  # 记录生命周期。
with managed_resource(events) as resource:  # 使用上下文管理器。
    events.append(resource)  # 在with内部使用资源。
assert events == ['open', 'resource', 'close']  # 验证进入使用退出顺序。
print(events)  # 输出生命周期。
from contextlib import ExitStack, nullcontext  # 导入动态上下文组合工具。
with ExitStack() as stack:  # 按运行时条件管理多个资源。
    first = stack.enter_context(nullcontext('A'))  # 加入第一个无需清理的上下文。
    second = stack.enter_context(nullcontext('B'))  # 加入第二个上下文。
assert first + second == 'AB'  # 验证动态上下文组合。

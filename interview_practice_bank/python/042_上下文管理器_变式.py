"""
题目 042：上下文管理器_变式

要求：完成“上下文管理器”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入生成器式上下文管理器。
2. 导入生成器上下文装饰器。
3. 保存进入退出顺序。
@contextmanager  # 把生成器转换为上下文管理器。
def managed():  # 定义资源生命周期。
    events.append('enter')  # 进入时执行。
    try:  # 保证清理。
        yield 'resource'  # 把资源交给with块。
    finally:  # 无论是否异常都执行。
        events.append('exit')  # 记录退出。
with managed() as resource:  # 使用资源。
    events.append(resource)  # 记录块内行为。
4. 验证上下文顺序。

完成标准：
- 验证上下文顺序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from contextlib import contextmanager  # 导入生成器式上下文管理器。
from contextlib import contextmanager  # 导入生成器上下文装饰器。
events = []  # 保存进入退出顺序。
@contextmanager  # 把生成器转换为上下文管理器。
def managed():  # 定义资源生命周期。
    events.append('enter')  # 进入时执行。
    try:  # 保证清理。
        yield 'resource'  # 把资源交给with块。
    finally:  # 无论是否异常都执行。
        events.append('exit')  # 记录退出。
with managed() as resource:  # 使用资源。
    events.append(resource)  # 记录块内行为。
assert events == ['enter', 'resource', 'exit']  # 验证上下文顺序。

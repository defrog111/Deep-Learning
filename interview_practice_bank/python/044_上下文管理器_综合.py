"""
题目 044：上下文管理器_综合

要求：完成“上下文管理器”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
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

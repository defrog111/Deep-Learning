"""
题目 043：上下文管理器_易错点

要求：完成“上下文管理器”的易错点题，并说明时间复杂度、对象身份或协议行为。

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
- 能执行到这里说明异常已按协议抑制。
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
class SuppressValueError:  # 实现可选择抑制异常的上下文管理器。
    def __enter__(self):  # 进入上下文。
        return self  # 返回管理器。
    def __exit__(self, error_type, error, traceback):  # 接收异常信息。
        return error_type is ValueError  # 返回True只抑制ValueError。
with SuppressValueError():  # 使用异常抑制器。
    raise ValueError('handled')  # 该异常不会传播。
assert True  # 能执行到这里说明异常已按协议抑制。

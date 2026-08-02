"""
题目 043：上下文管理器_易错点

要求：完成“上下文管理器”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入生成器式上下文管理器。
2. 实现可选择抑制异常的上下文管理器。
    def __enter__(self):  # 进入上下文。
        return self  # 返回管理器。
    def __exit__(self, error_type, error, traceback):  # 接收异常信息。
        return error_type is ValueError  # 返回True只抑制ValueError。
with SuppressValueError():  # 使用异常抑制器。
    raise ValueError('handled')  # 该异常不会传播。
3. 能执行到这里说明异常已按协议抑制。

完成标准：
- 能执行到这里说明异常已按协议抑制。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from contextlib import contextmanager  # 导入生成器式上下文管理器。
class SuppressValueError:  # 实现可选择抑制异常的上下文管理器。
    def __enter__(self):  # 进入上下文。
        return self  # 返回管理器。
    def __exit__(self, error_type, error, traceback):  # 接收异常信息。
        return error_type is ValueError  # 返回True只抑制ValueError。
with SuppressValueError():  # 使用异常抑制器。
    raise ValueError('handled')  # 该异常不会传播。
assert True  # 能执行到这里说明异常已按协议抑制。

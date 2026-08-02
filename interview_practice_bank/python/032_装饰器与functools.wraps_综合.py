"""
题目 032：装饰器与functools.wraps_综合

要求：完成“装饰器与functools.wraps”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入保留函数元数据的装饰器。
2. 用类实现有状态装饰器。
    def __init__(self, function):  # 保存被装饰函数。
        self.function, self.count = function, 0  # 初始化函数和计数。
    def __call__(self, *args, **kwargs):  # 让实例可调用。
        self.count += 1  # 累计调用次数。
        return self.function(*args, **kwargs)  # 转发调用。
@CountCalls  # 使用类装饰器。
def identity(value):  # 定义示例函数。
    return value  # 原样返回。
3. 验证类装饰器状态。

完成标准：
- 验证类装饰器状态。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import wraps  # 导入保留函数元数据的装饰器。
class CountCalls:  # 用类实现有状态装饰器。
    def __init__(self, function):  # 保存被装饰函数。
        self.function, self.count = function, 0  # 初始化函数和计数。
    def __call__(self, *args, **kwargs):  # 让实例可调用。
        self.count += 1  # 累计调用次数。
        return self.function(*args, **kwargs)  # 转发调用。
@CountCalls  # 使用类装饰器。
def identity(value):  # 定义示例函数。
    return value  # 原样返回。
assert identity(3) == 3 and identity.count == 1  # 验证类装饰器状态。

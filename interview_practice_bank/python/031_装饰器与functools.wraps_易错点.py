"""
题目 031：装饰器与functools.wraps_易错点

要求：完成“装饰器与functools.wraps”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入保留函数元数据的装饰器。
2. 保存装饰发生和调用顺序。
def order_decorator(label):  # 创建记录顺序的装饰器。
    def decorate(function):  # 接收函数。
        calls.append('decorate-' + label)  # 装饰在定义阶段从下向上发生。
        return function  # 保持函数行为。
    return decorate  # 返回装饰器。
@order_decorator('top')  # 外层后执行。
@order_decorator('bottom')  # 内层先执行。
def ordered():  # 定义被装饰函数。
    return None  # 返回空值。
3. 验证装饰器应用顺序。

完成标准：
- 验证装饰器应用顺序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import wraps  # 导入保留函数元数据的装饰器。
calls = []  # 保存装饰发生和调用顺序。
def order_decorator(label):  # 创建记录顺序的装饰器。
    def decorate(function):  # 接收函数。
        calls.append('decorate-' + label)  # 装饰在定义阶段从下向上发生。
        return function  # 保持函数行为。
    return decorate  # 返回装饰器。
@order_decorator('top')  # 外层后执行。
@order_decorator('bottom')  # 内层先执行。
def ordered():  # 定义被装饰函数。
    return None  # 返回空值。
assert calls == ['decorate-bottom', 'decorate-top']  # 验证装饰器应用顺序。

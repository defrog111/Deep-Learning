"""
题目 030：装饰器与functools.wraps_变式

要求：完成“装饰器与functools.wraps”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入保留函数元数据的装饰器。
2. 定义带参数的装饰器工厂。
    def decorate(function):  # 接收被装饰函数。
        @wraps(function)  # 保留元数据。
        def wrapper(*args, **kwargs):  # 接受任意调用参数。
            return [function(*args, **kwargs) for _ in range(times)]  # 重复调用并收集结果。
        return wrapper  # 返回包装函数。
    return decorate  # 返回真正装饰器。
3. 应用参数化装饰器。
def doubled_name(name):  # 定义示例函数。
    return name.upper()  # 返回大写名字。
4. 验证行为和元数据。

完成标准：
- 验证行为和元数据。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import wraps  # 导入保留函数元数据的装饰器。
def repeat(times):  # 定义带参数的装饰器工厂。
    def decorate(function):  # 接收被装饰函数。
        @wraps(function)  # 保留元数据。
        def wrapper(*args, **kwargs):  # 接受任意调用参数。
            return [function(*args, **kwargs) for _ in range(times)]  # 重复调用并收集结果。
        return wrapper  # 返回包装函数。
    return decorate  # 返回真正装饰器。
@repeat(2)  # 应用参数化装饰器。
def doubled_name(name):  # 定义示例函数。
    return name.upper()  # 返回大写名字。
assert doubled_name('a') == ['A', 'A'] and doubled_name.__name__ == 'doubled_name'  # 验证行为和元数据。

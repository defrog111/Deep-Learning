"""
题目 030：装饰器与functools.wraps_变式

要求：完成“装饰器与functools.wraps”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义带状态的装饰器。
2. 状态保存在闭包中。
3. 保留原函数名称和文档。
4. 接收任意原函数参数。
5. 声明修改外层局部变量。
6. 累加调用次数。
7. 返回次数和原结果。
8. 返回包装函数。
9. 应用装饰器。
10. 定义被装饰函数。

完成标准：
- 验证元数据和状态。
- 验证行为和元数据。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import wraps  # 导入保留函数元数据的装饰器。
def count_calls(function):  # 定义带状态的装饰器。
    calls = 0  # 状态保存在闭包中。
    @wraps(function)  # 保留原函数名称和文档。
    def wrapper(*args, **kwargs):  # 接收任意原函数参数。
        nonlocal calls  # 声明修改外层局部变量。
        calls += 1  # 累加调用次数。
        return calls, function(*args, **kwargs)  # 返回次数和原结果。
    return wrapper  # 返回包装函数。
@count_calls  # 应用装饰器。
def add(a, b):  # 定义被装饰函数。
    return a + b  # 返回加法结果。
result = add(4, 4)  # 调用包装后的函数。
assert add.__name__ == 'add' and result[0] == 1  # 验证元数据和状态。
print(result)  # 输出装饰器结果。
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

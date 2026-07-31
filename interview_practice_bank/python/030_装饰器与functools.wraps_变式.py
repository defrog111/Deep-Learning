"""
题目 030：装饰器与functools.wraps_变式

要求：完成“装饰器与functools.wraps”的变式题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
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

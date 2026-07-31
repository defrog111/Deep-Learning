"""
题目 048：异常处理与自定义异常_综合

要求：完成“异常处理与自定义异常”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

class ValidationError(ValueError):  # 定义语义明确的业务异常。
    pass  # 继承父类行为即可。
def positive(value):  # 定义输入校验函数。
    if value <= 0:  # 检查非法条件。
        raise ValidationError('value must be positive')  # 抛出自定义异常。
    return value  # 合法时返回。
try:  # 捕获预期业务异常。
    positive(-6)  # 传入非法值触发异常。
except ValidationError as error:  # 只捕获具体异常而不是裸except。
    message = str(error)  # 保存错误信息。
else:  # 未发生异常才执行。
    message = 'ok'  # 设置成功信息。
finally:  # 无论如何都执行清理逻辑。
    cleaned = True  # 模拟资源清理。
assert message and cleaned  # 验证异常路径和finally。
print(type(message).__name__, message)  # 输出异常处理结果。

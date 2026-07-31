"""
题目 046：异常处理与自定义异常_变式

要求：完成“异常处理与自定义异常”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义语义明确的业务异常。
2. 继承父类行为即可。
3. 定义输入校验函数。
4. 检查非法条件。
5. 抛出自定义异常。
6. 合法时返回。
7. 捕获预期业务异常。
8. 传入非法值触发异常。
9. 只捕获具体异常而不是裸except。
10. 保存错误信息。

完成标准：
- 验证异常路径和finally。
- 验证else只在成功时运行。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class ValidationError(ValueError):  # 定义语义明确的业务异常。
    pass  # 继承父类行为即可。
def positive(value):  # 定义输入校验函数。
    if value <= 0:  # 检查非法条件。
        raise ValidationError('value must be positive')  # 抛出自定义异常。
    return value  # 合法时返回。
try:  # 捕获预期业务异常。
    positive(-4)  # 传入非法值触发异常。
except ValidationError as error:  # 只捕获具体异常而不是裸except。
    message = str(error)  # 保存错误信息。
else:  # 未发生异常才执行。
    message = 'ok'  # 设置成功信息。
finally:  # 无论如何都执行清理逻辑。
    cleaned = True  # 模拟资源清理。
assert message and cleaned  # 验证异常路径和finally。
print(type(message).__name__, message)  # 输出异常处理结果。
try:  # 演示try/except/else。
    parsed = int('12')  # 执行可能失败的转换。
except ValueError:  # 只捕获预期异常。
    parsed = 0  # 提供失败兜底。
else:  # 无异常时执行。
    parsed += 1  # 处理成功结果。
assert parsed == 13  # 验证else只在成功时运行。

"""
题目 045：异常处理与自定义异常_基础

要求：完成“异常处理与自定义异常”的基础题，并说明时间复杂度、对象身份或协议行为。

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
    positive(-3)  # 传入非法值触发异常。
except ValidationError as error:  # 只捕获具体异常而不是裸except。
    message = str(error)  # 保存错误信息。
else:  # 未发生异常才执行。
    message = 'ok'  # 设置成功信息。
finally:  # 无论如何都执行清理逻辑。
    cleaned = True  # 模拟资源清理。
assert message and cleaned  # 验证异常路径和finally。
print(type(message).__name__, message)  # 输出异常处理结果。

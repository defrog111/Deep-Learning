"""
题目 048：异常处理与自定义异常_综合

要求：完成“异常处理与自定义异常”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 自定义更具体的业务异常。
    pass  # 继承标准异常语义。
def require_positive(value):  # 定义带异常链的校验函数。
    if value <= 0:  # 检查业务约束。
        raise ValidationError('must be positive')  # 抛出明确错误。
    return value  # 返回合法值。
2. 验证自定义异常。
    require_positive(0)  # 传入非法值。
except ValidationError as error:  # 精确捕获业务异常。
    validation_message = str(error)  # 保存信息。
3. 验证异常类型和消息。

完成标准：
- 验证异常类型和消息。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class ValidationError(ValueError):  # 自定义更具体的业务异常。
    pass  # 继承标准异常语义。
def require_positive(value):  # 定义带异常链的校验函数。
    if value <= 0:  # 检查业务约束。
        raise ValidationError('must be positive')  # 抛出明确错误。
    return value  # 返回合法值。
try:  # 验证自定义异常。
    require_positive(0)  # 传入非法值。
except ValidationError as error:  # 精确捕获业务异常。
    validation_message = str(error)  # 保存信息。
assert validation_message == 'must be positive'  # 验证异常类型和消息。

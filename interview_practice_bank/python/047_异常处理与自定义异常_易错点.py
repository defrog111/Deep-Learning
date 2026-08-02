"""
题目 047：异常处理与自定义异常_易错点

要求：完成“异常处理与自定义异常”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 演示finally总会执行。
    raise RuntimeError('boom')  # 主动产生异常。
except RuntimeError as error:  # 捕获具体异常。
    message = str(error)  # 保存错误。
finally:  # 无论是否捕获都会运行。
    cleaned_up = True  # 模拟资源清理。
2. 验证异常与清理。

完成标准：
- 验证异常与清理。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

try:  # 演示finally总会执行。
    raise RuntimeError('boom')  # 主动产生异常。
except RuntimeError as error:  # 捕获具体异常。
    message = str(error)  # 保存错误。
finally:  # 无论是否捕获都会运行。
    cleaned_up = True  # 模拟资源清理。
assert message == 'boom' and cleaned_up  # 验证异常与清理。

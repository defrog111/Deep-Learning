"""
题目 046：异常处理与自定义异常_变式

要求：完成“异常处理与自定义异常”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 演示try/except/else。
    parsed = int('12')  # 执行可能失败的转换。
except ValueError:  # 只捕获预期异常。
    parsed = 0  # 提供失败兜底。
else:  # 无异常时执行。
    parsed += 1  # 处理成功结果。
2. 验证else只在成功时运行。

完成标准：
- 验证else只在成功时运行。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

try:  # 演示try/except/else。
    parsed = int('12')  # 执行可能失败的转换。
except ValueError:  # 只捕获预期异常。
    parsed = 0  # 提供失败兜底。
else:  # 无异常时执行。
    parsed += 1  # 处理成功结果。
assert parsed == 13  # 验证else只在成功时运行。

"""
题目 024：函数参数args与kwargs_综合

要求：完成“函数参数args与kwargs”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入函数签名检查工具。
2. 定义同时含位置限定和关键字限定的独立示例。
    return value * scale  # 返回缩放结果。
3. 在调用前按签名绑定参数。
4. 验证参数协议。

完成标准：
- 验证参数协议。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import inspect  # 导入函数签名检查工具。
def inspected_function(value, /, *, scale=1):  # 定义同时含位置限定和关键字限定的独立示例。
    return value * scale  # 返回缩放结果。
signature = inspect.signature(inspected_function); bound = signature.bind(5, scale=4)  # 在调用前按签名绑定参数。
assert bound.arguments == {'value': 5, 'scale': 4}  # 验证参数协议。

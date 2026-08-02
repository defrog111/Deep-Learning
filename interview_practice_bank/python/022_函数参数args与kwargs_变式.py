"""
题目 022：函数参数args与kwargs_变式

要求：完成“函数参数args与kwargs”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 斜杠前参数只能按位置，星号后参数只能按关键字。
    return value * scale  # 返回缩放结果。
2. 验证位置限定和关键字限定。

完成标准：
- 验证位置限定和关键字限定。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

def positional_and_keyword(value, /, *, scale=1):  # 斜杠前参数只能按位置，星号后参数只能按关键字。
    return value * scale  # 返回缩放结果。
assert positional_and_keyword(3, scale=2) == 6  # 验证位置限定和关键字限定。

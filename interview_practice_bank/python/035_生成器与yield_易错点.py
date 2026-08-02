"""
题目 035：生成器与yield_易错点

要求：完成“生成器与yield”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义可接收send值的生成器。
    received = yield 'ready'  # 第一次next停在yield，send把值送回表达式。
    yield received * 2  # 产生处理结果。
2. 验证send协议。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

def receiver():  # 定义可接收send值的生成器。
    received = yield 'ready'  # 第一次next停在yield，send把值送回表达式。
    yield received * 2  # 产生处理结果。
channel = receiver(); assert next(channel) == 'ready' and channel.send(5) == 10  # 验证send协议。

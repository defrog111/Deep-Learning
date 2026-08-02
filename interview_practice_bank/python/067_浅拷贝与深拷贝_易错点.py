"""
题目 067：浅拷贝与深拷贝_易错点

要求：完成“浅拷贝与深拷贝”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入复制工具。
2. 导入弱引用。
class Payload:  # 定义可弱引用对象。
    pass  # 无需额外行为。
3. 弱引用不增加强引用所有权。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import copy  # 导入复制工具。
import weakref  # 导入弱引用。
class Payload:  # 定义可弱引用对象。
    pass  # 无需额外行为。
payload = Payload(); reference = weakref.ref(payload); assert reference() is payload  # 弱引用不增加强引用所有权。

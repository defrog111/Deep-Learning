"""
题目 095：类型注解与Protocol_易错点

要求：完成“类型注解与Protocol”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入结构化类型工具。
2. 导入运行时协议检查装饰器。
3. 允许isinstance进行结构化检查。
class RuntimeLen(Protocol):  # 定义运行时可检查协议。
    def __len__(self) -> int:  # 声明长度接口。
        ...  # 协议无实现。
4. 验证结构化子类型。

完成标准：
- 验证结构化子类型。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from typing import Protocol, TypeVar  # 导入结构化类型工具。
from typing import runtime_checkable  # 导入运行时协议检查装饰器。
@runtime_checkable  # 允许isinstance进行结构化检查。
class RuntimeLen(Protocol):  # 定义运行时可检查协议。
    def __len__(self) -> int:  # 声明长度接口。
        ...  # 协议无实现。
assert isinstance([1, 2], RuntimeLen)  # 验证结构化子类型。

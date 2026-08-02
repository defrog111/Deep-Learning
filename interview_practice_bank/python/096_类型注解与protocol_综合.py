"""
题目 096：类型注解与Protocol_综合

要求：完成“类型注解与Protocol”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入结构化类型工具。
2. 定义泛型类型变量。
3. 导入泛型基类。
4. 定义保存任意T的容器。
    def __init__(self, value: T):  # 接收泛型值。
        self.value = value  # 保存同类型值。
    def get(self) -> T:  # 返回原类型。
        return self.value  # 取出值。
5. 验证泛型容器运行行为。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from typing import Protocol, TypeVar  # 导入结构化类型工具。
T = TypeVar('T')  # 定义泛型类型变量。
from typing import Generic  # 导入泛型基类。
class Box(Generic[T]):  # 定义保存任意T的容器。
    def __init__(self, value: T):  # 接收泛型值。
        self.value = value  # 保存同类型值。
    def get(self) -> T:  # 返回原类型。
        return self.value  # 取出值。
typed_box = Box[int](3); assert typed_box.get() == 3  # 验证泛型容器运行行为。

"""
题目 051：dataclass数据类_易错点

要求：完成“dataclass数据类”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入数据类工具。
2. 导入安全默认工厂。
3. 定义含列表的数据类。
class Basket:  # 定义购物篮。
    items: list = field(default_factory=list)  # 每个实例创建独立列表，避免共享默认值。
4. 验证默认工厂。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from dataclasses import dataclass, field  # 导入数据类工具。
from dataclasses import dataclass, field  # 导入安全默认工厂。
@dataclass  # 定义含列表的数据类。
class Basket:  # 定义购物篮。
    items: list = field(default_factory=list)  # 每个实例创建独立列表，避免共享默认值。
left_basket, right_basket = Basket(), Basket(); left_basket.items.append('x'); assert right_basket.items == []  # 验证默认工厂。

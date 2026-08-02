"""
题目 050：dataclass数据类_变式

要求：完成“dataclass数据类”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入数据类工具。
2. 导入数据类和不可变更新工具。
3. 创建不可变值对象。
class Point:  # 定义坐标。
    x: int  # 声明横坐标。
    y: int = 0  # 声明带默认值纵坐标。
4. 使用replace创建修改后的新对象。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from dataclasses import dataclass, field  # 导入数据类工具。
from dataclasses import dataclass, replace  # 导入数据类和不可变更新工具。
@dataclass(frozen=True)  # 创建不可变值对象。
class Point:  # 定义坐标。
    x: int  # 声明横坐标。
    y: int = 0  # 声明带默认值纵坐标。
point = Point(1); moved = replace(point, x=2); assert point.x == 1 and moved.x == 2  # 使用replace创建修改后的新对象。

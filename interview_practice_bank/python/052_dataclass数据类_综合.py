"""
题目 052：dataclass数据类_综合

要求：完成“dataclass数据类”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入数据类工具。
2. 导入具名元组。
3. 定义轻量不可变记录。
    x: int  # 声明横坐标。
    y: int  # 声明纵坐标。
4. 对比tuple与字段访问。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from dataclasses import dataclass, field  # 导入数据类工具。
from typing import NamedTuple  # 导入具名元组。
class Coordinate(NamedTuple):  # 定义轻量不可变记录。
    x: int  # 声明横坐标。
    y: int  # 声明纵坐标。
coordinate = Coordinate(1, 2); assert coordinate[0] == coordinate.x and coordinate._asdict()['y'] == 2  # 对比tuple与字段访问。

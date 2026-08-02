"""
题目 094：类型注解与Protocol_变式

要求：完成“类型注解与Protocol”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入结构化类型工具。
2. 导入字典结构和字面量类型。
3. 定义固定键配置结构。
    mode: Literal['train', 'eval']  # 限制模式候选值。
    epochs: int  # 声明整数轮数。
4. 运行时仍是普通dict，约束由类型检查器检查。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from typing import Protocol, TypeVar  # 导入结构化类型工具。
from typing import TypedDict, Literal  # 导入字典结构和字面量类型。
class Config(TypedDict):  # 定义固定键配置结构。
    mode: Literal['train', 'eval']  # 限制模式候选值。
    epochs: int  # 声明整数轮数。
config: Config = {'mode': 'train', 'epochs': 3}; assert config['epochs'] == 3  # 运行时仍是普通dict，约束由类型检查器检查。

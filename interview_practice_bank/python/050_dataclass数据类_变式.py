"""
题目 050：dataclass数据类_变式

要求：完成“dataclass数据类”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 自动生成比较、repr并设为不可变。
2. 定义值对象。
3. 排序首先使用分数。
4. 姓名不参与排序比较。
5. 创建数据类实例。
6. 使用自动生成的顺序方法。
7. frozen数据类可哈希可作字典键。

完成标准：
- 验证排序和哈希。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from dataclasses import dataclass, field  # 导入数据类工具。
@dataclass(order=True, frozen=True)  # 自动生成比较、repr并设为不可变。
class Student:  # 定义值对象。
    score: int  # 排序首先使用分数。
    name: str = field(compare=False)  # 姓名不参与排序比较。
students = [Student(80, 'A'), Student(84, 'B')]  # 创建数据类实例。
best = max(students)  # 使用自动生成的顺序方法。
mapping = {student: student.name for student in students}  # frozen数据类可哈希可作字典键。
assert best.name == 'B' and len(mapping) == 2  # 验证排序和哈希。
print(students, best)  # 输出数据类。

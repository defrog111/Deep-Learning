"""
题目 056：继承super与多态_综合

要求：完成“继承super与多态”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入抽象基类协议。
2. 定义抽象形状。
    @abstractmethod  # 强制子类实现面积。
    def area(self):  # 声明接口。
        raise NotImplementedError  # 提供明确占位。
class Square(Shape):  # 实现具体形状。
    def __init__(self, side):  # 保存边长。
        self.side = side  # 设置实例状态。
    def area(self):  # 实现抽象方法。
        return self.side**2  # 计算面积。
3. 验证抽象接口与多态。

完成标准：
- 验证抽象接口与多态。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from abc import ABC, abstractmethod  # 导入抽象基类协议。
class Shape(ABC):  # 定义抽象形状。
    @abstractmethod  # 强制子类实现面积。
    def area(self):  # 声明接口。
        raise NotImplementedError  # 提供明确占位。
class Square(Shape):  # 实现具体形状。
    def __init__(self, side):  # 保存边长。
        self.side = side  # 设置实例状态。
    def area(self):  # 实现抽象方法。
        return self.side**2  # 计算面积。
assert Square(3).area() == 9  # 验证抽象接口与多态。

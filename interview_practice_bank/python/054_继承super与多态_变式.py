"""
题目 054：继承super与多态_变式

要求：完成“继承super与多态”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义左侧父类。
    def label(self):  # 提供同名方法。
        return 'left'  # 返回标识。
class Right:  # 定义右侧父类。
    def label(self):  # 提供同名方法。
        return 'right'  # 返回标识。
class Child(Left, Right):  # 多继承按MRO解析方法。
    pass  # 无需覆盖。
2. 验证MRO顺序。

完成标准：
- 验证MRO顺序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class Left:  # 定义左侧父类。
    def label(self):  # 提供同名方法。
        return 'left'  # 返回标识。
class Right:  # 定义右侧父类。
    def label(self):  # 提供同名方法。
        return 'right'  # 返回标识。
class Child(Left, Right):  # 多继承按MRO解析方法。
    pass  # 无需覆盖。
assert Child().label() == 'left' and Child.__mro__[:3] == (Child, Left, Right)  # 验证MRO顺序。

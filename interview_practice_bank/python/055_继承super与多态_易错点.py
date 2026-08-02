"""
题目 055：继承super与多态_易错点

要求：完成“继承super与多态”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义协作式初始化基类。
    def __init__(self, **kwargs):  # 接收并转发剩余参数。
        self.a = kwargs.pop('a')  # 消费自己的参数。
        super().__init__(**kwargs)  # 按MRO继续初始化。
2. 验证多继承应使用协作式super。

完成标准：
- 验证多继承应使用协作式super。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class BaseA:  # 定义协作式初始化基类。
    def __init__(self, **kwargs):  # 接收并转发剩余参数。
        self.a = kwargs.pop('a')  # 消费自己的参数。
        super().__init__(**kwargs)  # 按MRO继续初始化。
assert 'super' in BaseA.__init__.__code__.co_names  # 验证多继承应使用协作式super。

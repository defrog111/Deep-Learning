"""
题目 059：property与封装_易错点

要求：完成“property与封装”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义数据描述符。
    def __set_name__(self, owner, name):  # 获得绑定属性名。
        self.storage = '_' + name  # 创建内部存储名。
    def __get__(self, instance, owner):  # 控制读取。
        return self if instance is None else getattr(instance, self.storage)  # 类访问返回描述符，实例访问返回值。
    def __set__(self, instance, value):  # 控制赋值。
        if value <= 0:  # 检查正数约束。
            raise ValueError('positive only')  # 拒绝非法值。
        setattr(instance, self.storage, value)  # 保存合法值。
class Product:  # 使用描述符。
    price = Positive()  # 多实例复用校验逻辑。
    def __init__(self, price):  # 初始化价格。
        self.price = price  # 触发描述符。
2. 验证descriptor协议。

完成标准：
- 验证descriptor协议。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class Positive:  # 定义数据描述符。
    def __set_name__(self, owner, name):  # 获得绑定属性名。
        self.storage = '_' + name  # 创建内部存储名。
    def __get__(self, instance, owner):  # 控制读取。
        return self if instance is None else getattr(instance, self.storage)  # 类访问返回描述符，实例访问返回值。
    def __set__(self, instance, value):  # 控制赋值。
        if value <= 0:  # 检查正数约束。
            raise ValueError('positive only')  # 拒绝非法值。
        setattr(instance, self.storage, value)  # 保存合法值。
class Product:  # 使用描述符。
    price = Positive()  # 多实例复用校验逻辑。
    def __init__(self, price):  # 初始化价格。
        self.price = price  # 触发描述符。
assert Product(3).price == 3  # 验证descriptor协议。

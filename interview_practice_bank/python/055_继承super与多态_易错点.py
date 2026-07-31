"""
题目 055：继承super与多态_易错点

要求：完成“继承super与多态”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义父类。
2. 初始化公共属性。
3. 保存名称。
4. 定义可覆盖的方法。
5. 返回默认行为。
6. 通过继承扩展父类。
7. 增加子类参数。
8. 使用super调用父类初始化。
9. 保存子类属性。
10. 覆盖方法实现多态。

完成标准：
- 验证多态。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class Animal:  # 定义父类。
    def __init__(self, name):  # 初始化公共属性。
        self.name = name  # 保存名称。
    def speak(self):  # 定义可覆盖的方法。
        return 'unknown'  # 返回默认行为。
class Dog(Animal):  # 通过继承扩展父类。
    def __init__(self, name, breed):  # 增加子类参数。
        super().__init__(name)  # 使用super调用父类初始化。
        self.breed = breed  # 保存子类属性。
    def speak(self):  # 覆盖方法实现多态。
        return f'{self.name}: woof'  # 返回Dog行为。
animals = [Animal('A'), Dog('B', 'Lab')]  # 用统一父类接口保存不同对象。
sounds = [animal.speak() for animal in animals]  # 动态分派调用实际类型方法。
assert sounds == ['unknown', 'B: woof']  # 验证多态。
print(sounds)  # 输出方法结果。

"""
题目 060：property与封装_综合

要求：完成“property与封装”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义封装温度的类。
2. 初始化时走属性校验。
3. 调用property setter。
4. 把方法暴露为只读式属性接口。
5. 定义getter。
6. 返回内部存储。
7. 定义同名属性setter。
8. 接收待设置值。
9. 检查绝对零度。
10. 拒绝非法状态。

完成标准：
- 验证派生属性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class Temperature:  # 定义封装温度的类。
    def __init__(self, celsius):  # 初始化时走属性校验。
        self.celsius = celsius  # 调用property setter。
    @property  # 把方法暴露为只读式属性接口。
    def celsius(self):  # 定义getter。
        return self._celsius  # 返回内部存储。
    @celsius.setter  # 定义同名属性setter。
    def celsius(self, value):  # 接收待设置值。
        if value < -273.15:  # 检查绝对零度。
            raise ValueError('below absolute zero')  # 拒绝非法状态。
        self._celsius = float(value)  # 规范化并保存。
    @property  # 定义派生只读属性。
    def fahrenheit(self):  # 计算华氏温度。
        return self.celsius * 9 / 5 + 32  # 返回转换结果。
temperature = Temperature(60)  # 创建合法实例。
assert temperature.fahrenheit > temperature.celsius  # 验证派生属性。
print(temperature.celsius, temperature.fahrenheit)  # 输出两种温标。

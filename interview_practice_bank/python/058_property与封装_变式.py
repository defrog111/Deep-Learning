"""
题目 058：property与封装_变式

要求：完成“property与封装”的变式题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
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
temperature = Temperature(40)  # 创建合法实例。
assert temperature.fahrenheit > temperature.celsius  # 验证派生属性。
print(temperature.celsius, temperature.fahrenheit)  # 输出两种温标。

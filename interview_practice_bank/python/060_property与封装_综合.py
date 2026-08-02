"""
题目 060：property与封装_综合

要求：完成“property与封装”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义带双向转换属性的类。
    def __init__(self, value):  # 保存摄氏温度。
        self.celsius = value  # 使用setter校验。
    @property  # 暴露华氏只读计算属性。
    def fahrenheit(self):  # 定义转换读取。
        return self.celsius * 9 / 5 + 32  # 计算华氏度。
    @fahrenheit.setter  # 允许按华氏度反向设置。
    def fahrenheit(self, value):  # 定义反向转换。
        self.celsius = (value - 32) * 5 / 9  # 更新底层摄氏值。
2. 验证property双向封装。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class Celsius:  # 定义带双向转换属性的类。
    def __init__(self, value):  # 保存摄氏温度。
        self.celsius = value  # 使用setter校验。
    @property  # 暴露华氏只读计算属性。
    def fahrenheit(self):  # 定义转换读取。
        return self.celsius * 9 / 5 + 32  # 计算华氏度。
    @fahrenheit.setter  # 允许按华氏度反向设置。
    def fahrenheit(self, value):  # 定义反向转换。
        self.celsius = (value - 32) * 5 / 9  # 更新底层摄氏值。
temperature = Celsius(0); temperature.fahrenheit = 212; assert temperature.celsius == 100  # 验证property双向封装。

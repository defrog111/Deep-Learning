"""
题目 058：property与封装_变式

要求：完成“property与封装”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入只计算一次的属性。
2. 定义带缓存属性的类。
    calls = 0  # 记录计算次数。
    @cached_property  # 首次访问后把结果写入实例字典。
    def value(self):  # 定义昂贵计算。
        self.calls += 1  # 累加调用。
        return 42  # 返回结果。
3. 验证cached_property。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import cached_property  # 导入只计算一次的属性。
class Expensive:  # 定义带缓存属性的类。
    calls = 0  # 记录计算次数。
    @cached_property  # 首次访问后把结果写入实例字典。
    def value(self):  # 定义昂贵计算。
        self.calls += 1  # 累加调用。
        return 42  # 返回结果。
expensive = Expensive(); assert expensive.value == expensive.value == 42 and expensive.calls == 1  # 验证cached_property。

"""
题目 062：常用魔术方法_变式

要求：完成“常用魔术方法”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入排序方法补全装饰器。
2. 只实现eq和lt即可补全其他比较。
class Version:  # 定义可排序版本号。
    def __init__(self, number):  # 保存数值。
        self.number = number  # 设置状态。
    def __eq__(self, other):  # 定义相等。
        return self.number == other.number  # 比较数值。
    def __lt__(self, other):  # 定义小于。
        return self.number < other.number  # 比较数值。
3. 验证补全的比较协议。

完成标准：
- 验证补全的比较协议。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import total_ordering  # 导入排序方法补全装饰器。
@total_ordering  # 只实现eq和lt即可补全其他比较。
class Version:  # 定义可排序版本号。
    def __init__(self, number):  # 保存数值。
        self.number = number  # 设置状态。
    def __eq__(self, other):  # 定义相等。
        return self.number == other.number  # 比较数值。
    def __lt__(self, other):  # 定义小于。
        return self.number < other.number  # 比较数值。
assert Version(1) < Version(2) and Version(2) >= Version(2)  # 验证补全的比较协议。

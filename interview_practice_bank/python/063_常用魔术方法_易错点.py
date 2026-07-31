"""
题目 063：常用魔术方法_易错点

要求：完成“常用魔术方法”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义支持运算符的二维向量。
2. 初始化坐标。
3. 同时赋值两个属性。
4. 定义开发者友好表示。
5. 返回可读字符串。
6. 定义len协议。
7. 二维向量长度固定为2个分量。
8. 定义加号行为。
9. 返回新向量。
10. 定义值相等语义。

完成标准：
- 验证协议实现。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class Vector:  # 定义支持运算符的二维向量。
    def __init__(self, x, y):  # 初始化坐标。
        self.x, self.y = x, y  # 同时赋值两个属性。
    def __repr__(self):  # 定义开发者友好表示。
        return f'Vector({self.x}, {self.y})'  # 返回可读字符串。
    def __len__(self):  # 定义len协议。
        return 2  # 二维向量长度固定为2个分量。
    def __add__(self, other):  # 定义加号行为。
        return Vector(self.x + other.x, self.y + other.y)  # 返回新向量。
    def __eq__(self, other):  # 定义值相等语义。
        return isinstance(other, Vector) and (self.x, self.y) == (other.x, other.y)  # 比较类型和坐标。
answer = Vector(1, 2) + Vector(5, 4)  # 使用重载运算符。
assert len(answer) == 2 and answer == Vector(6, 6)  # 验证协议实现。
print(answer)  # 调用__repr__输出。

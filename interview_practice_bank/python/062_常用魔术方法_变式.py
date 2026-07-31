"""
题目 062：常用魔术方法_变式

要求：完成“常用魔术方法”的变式题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
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
answer = Vector(1, 2) + Vector(4, 4)  # 使用重载运算符。
assert len(answer) == 2 and answer == Vector(5, 6)  # 验证协议实现。
print(answer)  # 调用__repr__输出。

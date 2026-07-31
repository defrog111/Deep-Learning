"""
题目 038：迭代器协议_变式

要求：完成“迭代器协议”的变式题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

class Countdown:  # 定义可迭代且自身为迭代器的类。
    def __init__(self, start):  # 保存起始状态。
        self.current = start  # 初始化当前值。
    def __iter__(self):  # 迭代器必须返回自身。
        return self  # 返回实现__next__的对象。
    def __next__(self):  # 定义每次next行为。
        if self.current <= 0:  # 判断结束条件。
            raise StopIteration  # 用协议异常结束迭代。
        value = self.current  # 保存待返回值。
        self.current -= 1  # 推进内部状态。
        return value  # 返回本轮值。
values = list(Countdown(4))  # 消费自定义迭代器。
assert values[0] == 4 and values[-1] == 1  # 验证倒计时。
print(values)  # 输出迭代结果。

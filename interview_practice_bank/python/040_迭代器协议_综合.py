"""
题目 040：迭代器协议_综合

要求：完成“迭代器协议”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 实现自定义反向迭代器。
    def __init__(self, values):  # 保存数据和当前位置。
        self.values, self.index = values, len(values)  # 从末尾开始。
    def __iter__(self):  # 返回迭代器自身。
        return self  # 满足迭代器协议。
    def __next__(self):  # 返回下一个元素。
        if self.index == 0:  # 检查终止条件。
            raise StopIteration  # 发出结束信号。
        self.index -= 1  # 向前移动。
        return self.values[self.index]  # 返回当前位置值。
2. 验证完整协议。

完成标准：
- 验证完整协议。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class ReverseIterator:  # 实现自定义反向迭代器。
    def __init__(self, values):  # 保存数据和当前位置。
        self.values, self.index = values, len(values)  # 从末尾开始。
    def __iter__(self):  # 返回迭代器自身。
        return self  # 满足迭代器协议。
    def __next__(self):  # 返回下一个元素。
        if self.index == 0:  # 检查终止条件。
            raise StopIteration  # 发出结束信号。
        self.index -= 1  # 向前移动。
        return self.values[self.index]  # 返回当前位置值。
assert list(ReverseIterator([1, 2, 3])) == [3, 2, 1]  # 验证完整协议。

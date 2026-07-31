"""
题目 040：迭代器协议_综合

要求：完成“迭代器协议”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义可迭代且自身为迭代器的类。
2. 保存起始状态。
3. 初始化当前值。
4. 迭代器必须返回自身。
5. 返回实现__next__的对象。
6. 定义每次next行为。
7. 判断结束条件。
8. 用协议异常结束迭代。
9. 保存待返回值。
10. 推进内部状态。

完成标准：
- 验证倒计时。
- 验证完整协议。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
values = list(Countdown(6))  # 消费自定义迭代器。
assert values[0] == 6 and values[-1] == 1  # 验证倒计时。
print(values)  # 输出迭代结果。
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

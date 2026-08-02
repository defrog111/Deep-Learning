"""
题目 064：常用魔术方法_综合

要求：完成“常用魔术方法”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 实现常用容器魔术方法。
    def __init__(self, values):  # 保存数据。
        self.values = list(values)  # 建立内部列表。
    def __len__(self):  # 支持len和真值判断。
        return len(self.values)  # 返回元素数。
    def __getitem__(self, index):  # 支持索引并间接支持迭代。
        return self.values[index]  # 返回指定元素。
    def __contains__(self, value):  # 自定义in测试。
        return value in self.values  # 委托内部列表。
2. 验证容器协议。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class SizedContainer:  # 实现常用容器魔术方法。
    def __init__(self, values):  # 保存数据。
        self.values = list(values)  # 建立内部列表。
    def __len__(self):  # 支持len和真值判断。
        return len(self.values)  # 返回元素数。
    def __getitem__(self, index):  # 支持索引并间接支持迭代。
        return self.values[index]  # 返回指定元素。
    def __contains__(self, value):  # 自定义in测试。
        return value in self.values  # 委托内部列表。
container = SizedContainer([1, 2]); assert len(container) == 2 and 2 in container and list(container) == [1, 2]  # 验证容器协议。

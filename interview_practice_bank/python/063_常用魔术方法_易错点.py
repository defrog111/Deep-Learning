"""
题目 063：常用魔术方法_易错点

要求：完成“常用魔术方法”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义可哈希值对象。
    def __init__(self, value):  # 保存不可变逻辑值。
        self.value = value  # 设置状态。
    def __eq__(self, other):  # 定义逻辑相等。
        return isinstance(other, HashKey) and self.value == other.value  # 检查类型和值。
    def __hash__(self):  # 相等对象必须返回相同哈希。
        return hash(self.value)  # 复用底层值哈希。
2. 验证eq与hash契约。

完成标准：
- 验证eq与hash契约。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

class HashKey:  # 定义可哈希值对象。
    def __init__(self, value):  # 保存不可变逻辑值。
        self.value = value  # 设置状态。
    def __eq__(self, other):  # 定义逻辑相等。
        return isinstance(other, HashKey) and self.value == other.value  # 检查类型和值。
    def __hash__(self):  # 相等对象必须返回相同哈希。
        return hash(self.value)  # 复用底层值哈希。
assert {HashKey(1): 'ok'}[HashKey(1)] == 'ok'  # 验证eq与hash契约。

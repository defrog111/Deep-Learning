"""
题目 079：collections高频工具_易错点

要求：完成“collections高频工具”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入高频容器。
2. 有界deque会自动丢弃最旧元素。
3. 验证滑动窗口语义。

完成标准：
- 验证滑动窗口语义。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from collections import Counter, defaultdict, deque  # 导入高频容器。
bounded = deque(maxlen=3); bounded.extend(range(5))  # 有界deque会自动丢弃最旧元素。
assert list(bounded) == [2, 3, 4]  # 验证滑动窗口语义。

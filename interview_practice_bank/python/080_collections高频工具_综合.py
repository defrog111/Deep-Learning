"""
题目 080：collections高频工具_综合

要求：完成“collections高频工具”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入高频容器。
2. Counter支持多重集合加减交并。
3. 验证计数合并。

完成标准：
- 验证计数合并。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from collections import Counter, defaultdict, deque  # 导入高频容器。
counter_left = Counter('aab'); counter_right = Counter('bcc'); combined_counter = counter_left + counter_right  # Counter支持多重集合加减交并。
assert combined_counter == Counter({'a': 2, 'b': 2, 'c': 2})  # 验证计数合并。

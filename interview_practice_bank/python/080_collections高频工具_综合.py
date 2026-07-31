"""
题目 080：collections高频工具_综合

要求：完成“collections高频工具”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建词序列。
2. 一行统计频次。
3. 缺失键自动创建空列表。
4. 按首字母分组。
5. 直接追加无需先判断键。
6. 创建固定长度双端队列。
7. 超过maxlen自动从左侧淘汰。
8. Counter支持多重集合加减交并。

完成标准：
- 验证最高频词。
- 验证计数合并。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from collections import Counter, defaultdict, deque  # 导入高频容器。
words = 'to be or not to be'.split()  # 创建词序列。
counts = Counter(words)  # 一行统计频次。
groups = defaultdict(list)  # 缺失键自动创建空列表。
for word in words:  # 按首字母分组。
    groups[word[0]].append(word)  # 直接追加无需先判断键。
queue = deque(maxlen=6)  # 创建固定长度双端队列。
queue.extend(words)  # 超过maxlen自动从左侧淘汰。
assert counts.most_common(1)[0] == ('to', 2)  # 验证最高频词。
print(counts, dict(groups), list(queue))  # 输出容器结果。
counter_left = Counter('aab'); counter_right = Counter('bcc'); combined_counter = counter_left + counter_right  # Counter支持多重集合加减交并。
assert combined_counter == Counter({'a': 2, 'b': 2, 'c': 2})  # 验证计数合并。

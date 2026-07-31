"""
题目 079：collections高频工具_易错点

要求：完成“collections高频工具”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建词序列。
2. 一行统计频次。
3. 缺失键自动创建空列表。
4. 按首字母分组。
5. 直接追加无需先判断键。
6. 创建固定长度双端队列。
7. 超过maxlen自动从左侧淘汰。

完成标准：
- 验证最高频词。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from collections import Counter, defaultdict, deque  # 导入高频容器。
words = 'to be or not to be'.split()  # 创建词序列。
counts = Counter(words)  # 一行统计频次。
groups = defaultdict(list)  # 缺失键自动创建空列表。
for word in words:  # 按首字母分组。
    groups[word[0]].append(word)  # 直接追加无需先判断键。
queue = deque(maxlen=5)  # 创建固定长度双端队列。
queue.extend(words)  # 超过maxlen自动从左侧淘汰。
assert counts.most_common(1)[0] == ('to', 2)  # 验证最高频词。
print(counts, dict(groups), list(queue))  # 输出容器结果。

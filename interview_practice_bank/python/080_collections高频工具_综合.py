"""
题目 080：collections高频工具_综合

要求：完成“collections高频工具”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
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

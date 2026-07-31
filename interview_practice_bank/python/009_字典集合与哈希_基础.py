"""
题目 009：字典集合与哈希_基础

要求：完成“字典集合与哈希”的基础题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

records = [('a', 1), ('b', 2), ('a', 3)]  # 创建重复键记录。
latest = dict(records)  # 字典后写入的重复键覆盖旧值。
unique_keys = set(latest)  # set和dict键都要求可哈希。
counts = {}  # 初始化频次字典。
for key, _ in records:  # 遍历每条记录。
    counts[key] = counts.get(key, 0) + 1  # 使用get安全累加。
assert latest['a'] == 3 and counts['a'] == 2  # 验证覆盖和频次差异。
print(latest, unique_keys, counts)  # 输出哈希容器结果。

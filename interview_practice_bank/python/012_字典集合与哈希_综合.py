"""
题目 012：字典集合与哈希_综合

要求：完成“字典集合与哈希”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建重复键记录。
2. 字典后写入的重复键覆盖旧值。
3. set和dict键都要求可哈希。
4. 初始化频次字典。
5. 遍历每条记录。
6. 使用get安全累加。
7. 前层配置覆盖后层默认值。

完成标准：
- 验证覆盖和频次差异。
- 验证ChainMap查找顺序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

records = [('a', 1), ('b', 2), ('a', 3)]  # 创建重复键记录。
latest = dict(records)  # 字典后写入的重复键覆盖旧值。
unique_keys = set(latest)  # set和dict键都要求可哈希。
counts = {}  # 初始化频次字典。
for key, _ in records:  # 遍历每条记录。
    counts[key] = counts.get(key, 0) + 1  # 使用get安全累加。
assert latest['a'] == 3 and counts['a'] == 2  # 验证覆盖和频次差异。
print(latest, unique_keys, counts)  # 输出哈希容器结果。
from collections import ChainMap  # 导入多层配置查找结构。
settings = ChainMap({'timeout': 10}, {'timeout': 30, 'retries': 3})  # 前层配置覆盖后层默认值。
assert settings['timeout'] == 10 and settings['retries'] == 3  # 验证ChainMap查找顺序。

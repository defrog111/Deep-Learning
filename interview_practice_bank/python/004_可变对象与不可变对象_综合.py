"""
题目 004：可变对象与不可变对象_综合

要求：完成“可变对象与不可变对象”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义不应意外修改调用者数据的函数。
2. 同时复制字典和嵌套标签列表。
3. 返回新的业务记录。
4. 创建含嵌套列表的原记录。
5. 使用复制后更新模式生成新记录。
6. 用frozenset表达无顺序且不可变的标签集合。
7. 把不可变内容组合成可哈希字典键。
8. 使用不可变组合键建立缓存。

完成标准：
- 验证函数没有副作用。
- 验证复制更新和可哈希键。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

def add_tag(record, tag):  # 定义不应意外修改调用者数据的函数。
    copied = {**record, 'tags': [*record.get('tags', []), tag]}  # 同时复制字典和嵌套标签列表。
    return copied  # 返回新的业务记录。
source = {'name': 'Ada', 'tags': ['python']}  # 创建含嵌套列表的原记录。
updated = add_tag(source, 'ml')  # 使用复制后更新模式生成新记录。
immutable_tags = frozenset(source['tags'])  # 用frozenset表达无顺序且不可变的标签集合。
immutable_key = (source['name'], immutable_tags)  # 把不可变内容组合成可哈希字典键。
cache = {immutable_key: updated}  # 使用不可变组合键建立缓存。
assert source == {'name': 'Ada', 'tags': ['python']}  # 验证函数没有副作用。
assert updated['tags'] == ['python', 'ml'] and cache[immutable_key] is updated  # 验证复制更新和可哈希键。
print(source, updated, cache)  # 输出综合不可变设计结果。

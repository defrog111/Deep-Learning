"""
题目 004：可变对象与不可变对象_综合

要求：完成“可变对象与不可变对象”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. tuple是不可变对象。
2. list是可变对象。
3. 赋值只复制引用而不复制对象。
4. 原地修改列表。
5. 拼接tuple会创建新对象。

完成标准：
- 验证身份语义。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

immutable = (1, 2, 3)  # tuple是不可变对象。
mutable = [1, 2, 3]  # list是可变对象。
same_list = mutable  # 赋值只复制引用而不复制对象。
mutable.append(6)  # 原地修改列表。
new_tuple = immutable + (4,)  # 拼接tuple会创建新对象。
assert same_list is mutable and new_tuple is not immutable  # 验证身份语义。
print(mutable, immutable, new_tuple, id(mutable))  # 输出对象和值。

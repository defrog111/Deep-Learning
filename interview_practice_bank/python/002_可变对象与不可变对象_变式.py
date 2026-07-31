"""
题目 002：可变对象与不可变对象_变式

要求：完成“可变对象与不可变对象”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. tuple是不可变对象。
2. list是可变对象。
3. 赋值只复制引用而不复制对象。
4. 原地修改列表。
5. 拼接tuple会创建新对象。
6. 比较tuple不可变与list原地扩展。
7. list的+=保持身份，tuple拼接创建新对象。

完成标准：
- 验证身份语义。
- list的+=保持身份，tuple拼接创建新对象。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

immutable = (1, 2, 3)  # tuple是不可变对象。
mutable = [1, 2, 3]  # list是可变对象。
same_list = mutable  # 赋值只复制引用而不复制对象。
mutable.append(4)  # 原地修改列表。
new_tuple = immutable + (4,)  # 拼接tuple会创建新对象。
assert same_list is mutable and new_tuple is not immutable  # 验证身份语义。
print(mutable, immutable, new_tuple, id(mutable))  # 输出对象和值。
immutable = (1, 2, 3); mutable = [1, 2, 3]; before_id = id(mutable); mutable += [4]  # 比较tuple不可变与list原地扩展。
assert id(mutable) == before_id and immutable + (4,) == (1, 2, 3, 4)  # list的+=保持身份，tuple拼接创建新对象。

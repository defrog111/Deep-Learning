"""
题目 067：浅拷贝与深拷贝_易错点

要求：完成“浅拷贝与深拷贝”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建嵌套可变对象。
2. 浅拷贝只复制最外层容器。
3. 深拷贝递归复制内部对象。
4. 修改共享的内部列表会影响original。
5. 修改深拷贝内部列表不会影响original。
6. 弱引用不增加强引用所有权。

完成标准：
- 验证复制深度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import copy  # 导入复制工具。
original = {'items': [[1, 2], [3, 4]]}  # 创建嵌套可变对象。
shallow = copy.copy(original)  # 浅拷贝只复制最外层容器。
deep = copy.deepcopy(original)  # 深拷贝递归复制内部对象。
shallow['items'][0].append(99)  # 修改共享的内部列表会影响original。
deep['items'][1].append(88)  # 修改深拷贝内部列表不会影响original。
assert 99 in original['items'][0] and 88 not in original['items'][1]  # 验证复制深度。
print(original, shallow, deep)  # 输出三份对象。
import weakref  # 导入弱引用。
class Payload:  # 定义可弱引用对象。
    pass  # 无需额外行为。
payload = Payload(); reference = weakref.ref(payload); assert reference() is payload  # 弱引用不增加强引用所有权。

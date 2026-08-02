"""
题目 019：sorted的key与稳定排序_易错点

要求：完成“sorted的key与稳定排序”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. Python排序稳定，等键保持原顺序。
2. 验证稳定性。

完成标准：
- 验证稳定性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

stable_items = [('a', 2), ('b', 1), ('c', 2)]; stable_answer = sorted(stable_items, key=lambda item: item[1])  # Python排序稳定，等键保持原顺序。
assert stable_answer == [('b', 1), ('a', 2), ('c', 2)]  # 验证稳定性。

"""
题目 018：sorted的key与稳定排序_变式

要求：完成“sorted的key与稳定排序”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 构造待排序记录。
2. 用tuple key实现分数降序姓名升序。
3. 构造能展示稳定性的序列。
4. 相同key保持原相对顺序。
5. 取得第variant大元素用于小数据。
6. casefold比lower更适合大小写无关排序。

完成标准：
- 验证Timsort稳定性。
- 验证key只参与比较不改变原值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

records = [{'name': 'B', 'score': 90}, {'name': 'A', 'score': 90}, {'name': 'C', 'score': 80}]  # 构造待排序记录。
answer = sorted(records, key=lambda row: (-row['score'], row['name']))  # 用tuple key实现分数降序姓名升序。
original = [('first', 1), ('second', 1), ('third', 2)]  # 构造能展示稳定性的序列。
stable = sorted(original, key=lambda item: item[1])  # 相同key保持原相对顺序。
kth = sorted([5, 1, 9, 3, 7])[-2]  # 取得第variant大元素用于小数据。
assert stable[:2] == [('first', 1), ('second', 1)]  # 验证Timsort稳定性。
print(answer, stable, kth)  # 输出排序结果。
case_insensitive = sorted(['Bob', 'alice', 'ALAN'], key=str.casefold)  # casefold比lower更适合大小写无关排序。
assert case_insensitive == ['ALAN', 'alice', 'Bob']  # 验证key只参与比较不改变原值。

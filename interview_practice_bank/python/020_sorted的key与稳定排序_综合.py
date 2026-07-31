"""
题目 020：sorted的key与稳定排序_综合

要求：完成“sorted的key与稳定排序”的综合题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

records = [{'name': 'B', 'score': 90}, {'name': 'A', 'score': 90}, {'name': 'C', 'score': 80}]  # 构造待排序记录。
answer = sorted(records, key=lambda row: (-row['score'], row['name']))  # 用tuple key实现分数降序姓名升序。
original = [('first', 1), ('second', 1), ('third', 2)]  # 构造能展示稳定性的序列。
stable = sorted(original, key=lambda item: item[1])  # 相同key保持原相对顺序。
kth = sorted([5, 1, 9, 3, 7])[-4]  # 取得第variant大元素用于小数据。
assert stable[:2] == [('first', 1), ('second', 1)]  # 验证Timsort稳定性。
print(answer, stable, kth)  # 输出排序结果。

"""
题目 003：可变对象与不可变对象_易错点

要求：完成“可变对象与不可变对象”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 使用可变默认参数会让多次调用共享同一个列表。
2. 原地修改跨调用保留的默认列表。
3. 返回发生累积的列表。
4. 使用None作为不会被修改的哨兵。
5. 每次未传列表时创建新对象。
6. 只修改本次调用专属列表。
7. 返回安全结果。
8. 第一次调用修改默认列表。
9. 第二次调用继续使用同一个默认列表。
10. 第一次安全调用创建新列表。

完成标准：
- 验证可变默认参数陷阱。
- 验证None哨兵修复方式。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

def append_wrong(value, bucket=[]):  # 使用可变默认参数会让多次调用共享同一个列表。
    bucket.append(value)  # 原地修改跨调用保留的默认列表。
    return bucket  # 返回发生累积的列表。
def append_safe(value, bucket=None):  # 使用None作为不会被修改的哨兵。
    bucket = [] if bucket is None else bucket  # 每次未传列表时创建新对象。
    bucket.append(value)  # 只修改本次调用专属列表。
    return bucket  # 返回安全结果。
wrong_first = append_wrong(1)  # 第一次调用修改默认列表。
wrong_second = append_wrong(2)  # 第二次调用继续使用同一个默认列表。
safe_first = append_safe(1)  # 第一次安全调用创建新列表。
safe_second = append_safe(2)  # 第二次安全调用再次创建新列表。
assert wrong_first is wrong_second and wrong_second == [1, 2]  # 验证可变默认参数陷阱。
assert safe_first == [1] and safe_second == [2]  # 验证None哨兵修复方式。
print(wrong_second, safe_first, safe_second)  # 输出错误与正确结果。

"""
题目 036：生成器与yield_综合

要求：完成“生成器与yield”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 定义生成器函数。
2. 初始化前两项。
3. 惰性生成直到上限。
4. 暂停并产出当前值。
5. 更新状态。
6. 创建生成器但尚未执行函数体。
7. 手动消费第一个值。
8. 消费剩余值用于变式4。
9. close向当前题已定义的生成器发送GeneratorExit并终止。
10. 检查关闭后的行为。
    next(generator_close)  # 已关闭生成器应停止。
except StopIteration:  # 捕获正常结束信号。
    closed = True  # 记录关闭成功。

完成标准：
- 验证边界。
- 验证生成器生命周期。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

def fibonacci(limit):  # 定义生成器函数。
    first, second = 0, 1  # 初始化前两项。
    while first < limit:  # 惰性生成直到上限。
        yield first  # 暂停并产出当前值。
        first, second = second, first + second  # 更新状态。
generator = fibonacci(30)  # 创建生成器但尚未执行函数体。
first_value = next(generator)  # 手动消费第一个值。
remaining = list(generator)  # 消费剩余值用于变式4。
assert first_value == 0 and all(value < 30 for value in remaining)  # 验证边界。
print(first_value, remaining)  # 输出生成序列。
generator_close = fibonacci(5); next(generator_close); generator_close.close()  # close向当前题已定义的生成器发送GeneratorExit并终止。
try:  # 检查关闭后的行为。
    next(generator_close)  # 已关闭生成器应停止。
except StopIteration:  # 捕获正常结束信号。
    closed = True  # 记录关闭成功。
assert closed  # 验证生成器生命周期。

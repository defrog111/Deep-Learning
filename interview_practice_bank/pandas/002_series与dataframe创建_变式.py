"""
题目 002：Series与DataFrame创建_变式

要求：完成“Series与DataFrame创建”的变式题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
series = pd.Series([4, 5, 6], index=['a', 'b', 'c'], name='score')  # 创建带标签的一维 Series。
frame = series.to_frame().reset_index(names='student')  # 转为 DataFrame 并把索引恢复成普通列。
frame['passed'] = frame['score'].ge(4)  # 使用向量化比较创建布尔列。
assert frame.shape == (3, 3)  # 验证行列数符合预期。
print(frame)  # 输出结果用于检查。

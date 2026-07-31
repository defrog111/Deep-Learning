"""
CSV数据处理练习 008：读取UTF8文本CSV

题目：读取评论CSV，使用string dtype保存文本，并安全处理缺失评论。

操作过程：
1. 定位评论CSV。
2. 显式使用UTF-8和Pandas字符串类型。
3. 用空文本填补缺失评论。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'reviews.csv'  # 定位评论CSV。
frame = pd.read_csv(csv_path, encoding='utf-8', dtype={'review_text': 'string', 'tags': 'string'})  # 显式使用UTF-8和Pandas字符串类型。
frame['review_text'] = frame['review_text'].fillna('')  # 用空文本填补缺失评论。
assert str(frame['review_text'].dtype) == 'string' and not frame['review_text'].isna().any()  # 验证文本类型和缺失处理。
print(frame[['review_id', 'review_text']].head())  # 输出评论文本。

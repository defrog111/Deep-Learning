"""
CSV数据处理练习 034：字符串正则提取与词频

题目：读取评论CSV，清洗大小写和标点，提取包含delivery或refund的评论并统计单词。

操作过程：
1. 定位评论CSV。
2. 读取可空文本。
3. 清洗空白、大小写和标点。
4. 使用非捕获组和单词边界筛选主题评论。
5. 拆词、展开并统计频率。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'reviews.csv'  # 定位评论CSV。
frame = pd.read_csv(csv_path, dtype={'review_text': 'string'})  # 读取可空文本。
frame['clean_text'] = frame['review_text'].fillna('').str.strip().str.lower().str.replace(r'[^a-z\s]', '', regex=True)  # 清洗空白、大小写和标点。
matched = frame[frame['clean_text'].str.contains(r'\b(?:delivery|refund)\b', regex=True, na=False)]  # 使用非捕获组和单词边界筛选主题评论。
word_counts = frame['clean_text'].str.split().explode().value_counts()  # 拆词、展开并统计频率。
assert len(matched) >= 2 and 'delivery' in word_counts.index  # 验证主题筛选和词频。
print(matched[['review_id', 'clean_text']], word_counts.head(), sep='\n')  # 输出匹配评论和高频词。

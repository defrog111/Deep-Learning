"""
CSV数据处理练习 029：explode展开多值标签

题目：读取评论CSV，把竖线分隔的tags拆成列表并展开为一行一个标签。

操作过程：
1. 定位评论CSV。
2. 读取标签字符串。
3. 向量化拆分为列表。
4. 每个标签展开成独立行。
5. 统计标签频率。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'reviews.csv'  # 定位评论CSV。
frame = pd.read_csv(csv_path, dtype={'tags': 'string'})  # 读取标签字符串。
frame['tag_list'] = frame['tags'].str.split('|')  # 向量化拆分为列表。
exploded = frame.explode('tag_list', ignore_index=True).rename(columns={'tag_list': 'tag'})  # 每个标签展开成独立行。
tag_counts = exploded['tag'].value_counts()  # 统计标签频率。
assert len(exploded) == len(frame) * 2 and tag_counts.index.notna().all()  # 验证每条评论的两个标签均被展开。
print(tag_counts)  # 输出标签热度。

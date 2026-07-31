"""
CSV数据处理练习 028：crosstab交叉频数与比例

题目：读取评论CSV，计算评分和月份的交叉频数及按月归一化比例。

操作过程：
1. 定位评论CSV。
2. 读取评论和日期。
3. 创建月份标签。
4. 计算频数和总计。
5. 每个月内部归一化。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'reviews.csv'  # 定位评论CSV。
frame = pd.read_csv(csv_path, parse_dates=['created_at'])  # 读取评论和日期。
frame['month'] = frame['created_at'].dt.to_period('M').astype(str)  # 创建月份标签。
counts = pd.crosstab(frame['month'], frame['rating'], margins=True)  # 计算频数和总计。
row_proportions = pd.crosstab(frame['month'], frame['rating'], normalize='index')  # 每个月内部归一化。
assert row_proportions.sum(axis=1).round(10).eq(1).all()  # 验证每月比例和为1。
print(counts, row_proportions, sep='\n')  # 输出频数和比例。

"""
CSV数据处理练习 007：chunksize分块读取CSV

题目：使用chunksize逐块读取销售CSV，在不一次载入全表的情况下累计地区销售额。

操作过程：
1. 定位销售CSV。
2. 保存每个数据块的地区汇总。
3. 每次只解析四行。
4. 在当前块计算净额。
5. 保存当前块分组结果。
6. 合并并再次求和得到全局结果。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas。
csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位销售CSV。
partial_totals = []  # 保存每个数据块的地区汇总。
for chunk in pd.read_csv(csv_path, chunksize=4):  # 每次只解析四行。
    chunk['net'] = chunk['quantity'] * chunk['unit_price'] * (1 - chunk['discount'])  # 在当前块计算净额。
    partial_totals.append(chunk.groupby('region')['net'].sum())  # 保存当前块分组结果。
totals = pd.concat(partial_totals, axis=1).fillna(0).sum(axis=1)  # 合并并再次求和得到全局结果。
assert totals.index.is_unique and totals.gt(0).all()  # 验证每个地区只有一个正数总额。
print(totals.sort_values(ascending=False))  # 输出流式聚合结果。

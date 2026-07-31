"""生成一题一文件的面试练习题库。

生成后的题目文件是正式学习材料；此脚本用于保证数量、格式和目录可重复生成。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from numpy_cases import numpy_case
from pytorch_cases import pytorch_case
from python_cases import python_case
from ml_cases import ml_case
from sklearn_cases import sklearn_case


ROOT = Path(__file__).parent
LEVELS = ["基础", "变式", "易错点", "综合"]


def render_file(
    category: str,
    number: int,
    title: str,
    task: str,
    code_lines: list[str],
) -> None:
    """把一道题写为独立、可运行的 Python 文件。"""
    folder = ROOT / category
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / f"{number:03d}_{title.lower().replace(' ', '_')}.py"
    header = [
        '"""',
        f"题目 {number:03d}：{title}",
        "",
        f"要求：{task}",
        "先自己实现，再运行本文件查看参考代码结果。",
        '"""',
        "",
    ]
    filename.write_text("\n".join(header + code_lines) + "\n", encoding="utf-8")


def clean_generated_folders() -> None:
    """只清理生成目标，不触碰 data、README 和生成器。"""
    for name in ("pandas", "numpy", "pytorch", "python", "ml", "sklearn"):
        folder = ROOT / name
        if folder.exists():
            shutil.rmtree(folder)


def pandas_case(family: int, variant: int) -> tuple[str, str, list[str]]:
    """返回 Pandas 25 类高频知识点之一，每类生成四种变式。"""
    titles = [
        "Series与DataFrame创建",
        "读取CSV与基本检查",
        "loc与iloc选择",
        "布尔筛选与query",
        "缺失值检测与填充",
        "数据类型转换",
        "排序排名与TopN",
        "重复值识别与删除",
        "字符串向量化处理",
        "日期时间处理",
        "GroupBy聚合",
        "GroupBy的transform与filter",
        "merge多表连接",
        "concat纵向与横向拼接",
        "pivot_table与交叉表",
        "melt与宽长表转换",
        "MultiIndex多级索引",
        "rolling与expanding窗口",
        "shift差分与增长率",
        "map替换与apply",
        "Categorical分类数据",
        "explode展开列表列",
        "cut与qcut分箱",
        "索引对齐与副本陷阱",
        "时间序列resample",
    ]
    title = titles[family]
    task = f"完成“{title}”的{LEVELS[variant - 1]}题，并解释输出的 shape、索引和数据类型。"
    seed = variant + 2
    if family == 0:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            f"series = pd.Series([{seed}, {seed + 1}, {seed + 2}], index=['a', 'b', 'c'], name='score')  # 创建带标签的一维 Series。",
            "frame = series.to_frame().reset_index(names='student')  # 转为 DataFrame 并把索引恢复成普通列。",
            "frame['passed'] = frame['score'].ge(4)  # 使用向量化比较创建布尔列。",
            "assert frame.shape == (3, 3)  # 验证行列数符合预期。",
            "print(frame)  # 输出结果用于检查。",
        ]
    elif family == 1:
        code = [
            "from pathlib import Path  # 导入跨平台路径工具。",
            "import pandas as pd  # 导入 Pandas。",
            "csv_path = Path(__file__).parents[1] / 'data' / 'sales.csv'  # 定位题库自带的销售 CSV。",
            "frame = pd.read_csv(csv_path, parse_dates=['date'])  # 读取 CSV 并在入口解析日期。",
            "summary = frame[['quantity', 'unit_price', 'discount']].describe()  # 计算数值列描述统计。",
            "assert not frame.empty and frame['date'].dtype.kind == 'M'  # 验证数据非空且日期类型正确。",
            "print(frame.head(3), '\\n', summary)  # 查看前三行和统计摘要。",
        ]
    elif family == 2:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'name': ['A', 'B', 'C'], 'score': [78, 92, 85]}, index=[10, 20, 30])  # 创建非默认索引。",
            "by_label = frame.loc[20, 'score']  # loc 使用索引标签选择。",
            "by_position = frame.iloc[1, 1]  # iloc 使用整数位置选择。",
            "subset = frame.loc[frame['score'].gt(80), ['name', 'score']]  # 组合行条件和列选择。",
            "assert by_label == by_position == 92  # 证明本例标签20恰好位于位置1。",
            "print(subset)  # 输出筛选结果。",
        ]
    elif family == 3:
        code = [
            "from pathlib import Path  # 导入路径工具。",
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.read_csv(Path(__file__).parents[1] / 'data' / 'sales.csv')  # 读取销售数据。",
            f"minimum = {100 * seed}  # 设置随变式变化的最低原价。",
            "frame['gross'] = frame['quantity'] * frame['unit_price']  # 向量化计算折扣前金额。",
            "answer = frame.query('gross >= @minimum and category == \"Electronics\"')  # query中用@引用外部变量。",
            "assert answer['gross'].ge(minimum).all()  # 验证每一行都满足条件。",
            "print(answer[['order_id', 'product', 'gross']])  # 输出关键列。",
        ]
    elif family == 4:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'team': ['A', 'A', 'B', None], 'score': [10.0, None, 30.0, None]})  # 构造含缺失值的数据。",
            "missing_count = frame.isna().sum()  # 按列统计缺失值。",
            "frame['team'] = frame['team'].fillna('Unknown')  # 类别缺失值使用明确标签填充。",
            "frame['score'] = frame['score'].fillna(frame.groupby('team')['score'].transform('median')).fillna(0)  # 先组内中位数再兜底。",
            "assert not frame.isna().any().any()  # 验证所有缺失值均已处理。",
            "print(missing_count, '\\n', frame)  # 对比处理前统计与处理后数据。",
        ]
    elif family == 5:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'id': ['1', '2', 'bad'], 'active': ['True', 'False', 'True'], 'group': ['x', 'y', 'x']})  # 构造脏类型数据。",
            "frame['id'] = pd.to_numeric(frame['id'], errors='coerce').astype('Int64')  # 非法数字转缺失并使用可空整数。",
            "frame['active'] = frame['active'].map({'True': True, 'False': False}).astype('boolean')  # 显式映射布尔字符串。",
            "frame['group'] = frame['group'].astype('category')  # 低基数文本转分类类型。",
            "assert str(frame['id'].dtype) == 'Int64'  # 验证可空整数类型。",
            "print(frame.dtypes, '\\n', frame)  # 输出类型和数据。",
        ]
    elif family == 6:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'name': list('ABCDE'), 'score': [88, 95, 88, 72, 91]})  # 构造含并列分数的数据。",
            "frame['dense_rank'] = frame['score'].rank(method='dense', ascending=False).astype(int)  # 计算无跳号排名。",
            f"top = frame.nlargest({min(variant + 1, 5)}, 'score')  # 使用nlargest高效取得前N名。",
            "sorted_frame = frame.sort_values(['score', 'name'], ascending=[False, True])  # 多列稳定排序。",
            "assert sorted_frame.iloc[0]['score'] == frame['score'].max()  # 验证最高分位于首行。",
            "print(top, '\\n', sorted_frame)  # 输出TopN与完整排序。",
        ]
    elif family == 7:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'id': [1, 1, 2, 2], 'value': [10, 12, 20, 20], 'time': [1, 2, 1, 1]})  # 构造业务主键重复数据。",
            "duplicate_mask = frame.duplicated(subset=['id'], keep=False)  # 标出所有重复业务键。",
            "latest = frame.sort_values('time').drop_duplicates('id', keep='last')  # 每个id保留最新记录。",
            "exact_unique = frame.drop_duplicates()  # 删除整行完全重复记录。",
            "assert latest['id'].is_unique  # 验证业务键唯一。",
            "print(frame[duplicate_mask], '\\n', latest, '\\n', exact_unique)  # 对比不同去重策略。",
        ]
    elif family == 8:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "series = pd.Series(['  Alice-Smith ', 'bob-jones', None], dtype='string')  # 使用Pandas字符串类型并保留缺失。",
            "clean = series.str.strip().str.lower().str.replace('-', ' ', regex=False)  # 链式向量化清洗字符串。",
            "parts = clean.str.extract(r'(?P<first>\\w+)\\s+(?P<last>\\w+)')  # 用正则提取命名分组。",
            "contains_o = clean.str.contains('o', na=False)  # 缺失值按False处理。",
            "assert parts.columns.tolist() == ['first', 'last']  # 验证提取结果列名。",
            "print(clean, '\\n', parts, '\\n', contains_o)  # 输出清洗和提取结果。",
        ]
    elif family == 9:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'timestamp': ['2025-01-01 08:30', '2025-02-15 17:45', 'bad']})  # 构造日期字符串。",
            "frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce')  # 非法日期转NaT。",
            "frame['month'] = frame['timestamp'].dt.to_period('M')  # 提取月份周期。",
            "frame['weekday'] = frame['timestamp'].dt.day_name()  # 提取星期名称。",
            f"frame['plus_days'] = frame['timestamp'] + pd.Timedelta(days={variant})  # 日期加减使用Timedelta。",
            "assert frame['timestamp'].isna().sum() == 1  # 验证非法日期被识别。",
            "print(frame)  # 输出日期衍生特征。",
        ]
    elif family == 10:
        code = [
            "from pathlib import Path  # 导入路径工具。",
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.read_csv(Path(__file__).parents[1] / 'data' / 'sales.csv')  # 读取销售数据。",
            "frame['net'] = frame['quantity'] * frame['unit_price'] * (1 - frame['discount'])  # 计算净销售额。",
            "summary = frame.groupby('region', as_index=False).agg(total_net=('net', 'sum'), orders=('order_id', 'nunique'), avg_quantity=('quantity', 'mean'))  # 命名聚合。",
            "summary = summary.sort_values('total_net', ascending=False)  # 按总销售额降序。",
            "assert summary['orders'].sum() == frame['order_id'].nunique()  # 验证分组订单数守恒。",
            "print(summary)  # 输出地区汇总。",
        ]
    elif family == 11:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'team': list('AAABBB'), 'score': [70, 80, 90, 40, 50, 60]})  # 构造分组数据。",
            "frame['team_mean'] = frame.groupby('team')['score'].transform('mean')  # transform返回与原表等长结果。",
            "frame['centered'] = frame['score'] - frame['team_mean']  # 计算组内中心化分数。",
            f"large_groups = frame.groupby('team').filter(lambda group: group['score'].mean() >= {50 + seed})  # filter按组保留原始行。",
            "assert frame.groupby('team')['centered'].sum().abs().lt(1e-9).all()  # 验证各组中心化后和为零。",
            "print(frame, '\\n', large_groups)  # 输出transform和filter结果。",
        ]
    elif family == 12:
        code = [
            "from pathlib import Path  # 导入路径工具。",
            "import pandas as pd  # 导入 Pandas。",
            "base = Path(__file__).parents[1] / 'data'  # 定位数据目录。",
            "sales = pd.read_csv(base / 'sales.csv')  # 读取事实表。",
            "employees = pd.read_csv(base / 'employees.csv')  # 读取员工维表。",
            "answer = sales.merge(employees[['name', 'department', 'city']], left_on='salesperson', right_on='name', how='left', validate='many_to_one', indicator=True)  # 多对一左连接并验证关系。",
            "assert answer['_merge'].eq('both').all() and len(answer) == len(sales)  # 防止连接丢行或膨胀。",
            "print(answer[['order_id', 'salesperson', 'department', 'city']].head())  # 查看连接结果。",
        ]
    elif family == 13:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "first = pd.DataFrame({'id': [1, 2], 'value': [10, 20]}).set_index('id')  # 创建第一块索引数据。",
            "second = pd.DataFrame({'id': [3, 4], 'value': [30, 40]}).set_index('id')  # 创建第二块索引数据。",
            "rows = pd.concat([first, second], axis=0, verify_integrity=True)  # 纵向拼接并检查索引冲突。",
            "labels = pd.Series({1: 'A', 2: 'B', 3: 'C', 4: 'D'}, name='label')  # 创建同索引标签列。",
            "columns = pd.concat([rows, labels], axis=1, join='inner')  # 横向按索引对齐。",
            "assert columns.shape == (4, 2)  # 验证拼接shape。",
            "print(columns)  # 输出拼接结果。",
        ]
    elif family == 14:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'team': ['A', 'A', 'B', 'B'], 'quarter': ['Q1', 'Q2', 'Q1', 'Q2'], 'sales': [10, 15, 20, 18]})  # 构造长表。",
            "pivot = frame.pivot_table(index='team', columns='quarter', values='sales', aggfunc='sum', fill_value=0, margins=True)  # 创建带总计的透视表。",
            "cross = pd.crosstab(frame['team'], frame['quarter'], normalize='index')  # 计算组内比例交叉表。",
            "assert cross.sum(axis=1).round(8).eq(1).all()  # 验证每组比例和为1。",
            "print(pivot, '\\n', cross)  # 输出透视表与交叉表。",
        ]
    elif family == 15:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "wide = pd.DataFrame({'id': [1, 2], 'math': [90, 80], 'english': [85, 88]})  # 构造宽表。",
            "long = wide.melt(id_vars='id', var_name='subject', value_name='score')  # 宽表转长表。",
            "restored = long.pivot(index='id', columns='subject', values='score').reset_index()  # 长表恢复宽表。",
            "restored.columns.name = None  # 清除列索引名称便于比较。",
            "assert restored[['id', 'math', 'english']].equals(wide)  # 验证往返转换保持数据。",
            "print(long, '\\n', restored)  # 输出两种形态。",
        ]
    elif family == 16:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "index = pd.MultiIndex.from_product([['A', 'B'], [2024, 2025]], names=['team', 'year'])  # 创建多级索引。",
            "frame = pd.DataFrame({'sales': [10, 12, 20, 25]}, index=index)  # 构造MultiIndex表。",
            "team_a = frame.xs('A', level='team')  # xs按某一级快速截面选择。",
            "unstacked = frame.unstack('year')  # 把year级别旋转到列。",
            "restored = unstacked.stack('year', future_stack=True).sort_index()  # stack恢复多级行索引。",
            "assert restored.equals(frame)  # 验证stack/unstack可逆。",
            "print(team_a, '\\n', unstacked)  # 输出截面和宽表。",
        ]
    elif family == 17:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "series = pd.Series([2, 4, 6, 8, 10], dtype=float)  # 构造时间顺序数值。",
            f"rolling_mean = series.rolling(window={min(variant + 1, 5)}, min_periods=1).mean()  # 计算移动窗口均值。",
            "expanding_mean = series.expanding(min_periods=1).mean()  # 计算从起点累计均值。",
            "ewm_mean = series.ewm(alpha=0.5, adjust=False).mean()  # 计算指数加权均值。",
            "assert expanding_mean.iloc[-1] == series.mean()  # 最终累计均值等于全局均值。",
            "print(pd.DataFrame({'value': series, 'rolling': rolling_mean, 'expanding': expanding_mean, 'ewm': ewm_mean}))  # 汇总窗口结果。",
        ]
    elif family == 18:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'month': pd.date_range('2025-01-01', periods=5, freq='MS'), 'sales': [100, 120, 90, 135, 150]})  # 构造月度序列。",
            "frame['previous'] = frame['sales'].shift(1)  # 取得上一期数值。",
            "frame['change'] = frame['sales'].diff()  # 计算绝对差分。",
            "frame['growth'] = frame['sales'].pct_change()  # 计算环比增长率。",
            "assert frame.loc[1, 'change'] == 20  # 验证第二期差额。",
            "print(frame)  # 输出滞后与增长特征。",
        ]
    elif family == 19:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'grade': ['A', 'B', 'C', 'A'], 'score': [95, 82, 70, 91]})  # 构造映射数据。",
            "frame['points'] = frame['grade'].map({'A': 4, 'B': 3, 'C': 2})  # 一对一字典映射。",
            "frame['bucket'] = frame['score'].apply(lambda value: 'high' if value >= 90 else 'regular')  # 对单列执行自定义函数。",
            "frame['grade'] = frame['grade'].replace({'A': 'Excellent'})  # 替换指定值。",
            "assert frame['points'].notna().all()  # 验证映射无遗漏。",
            "print(frame)  # 输出转换结果。",
        ]
    elif family == 20:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "priority = pd.CategoricalDtype(['low', 'medium', 'high'], ordered=True)  # 定义有序分类类型。",
            "frame = pd.DataFrame({'task': list('ABCD'), 'priority': ['high', 'low', 'medium', 'low']})  # 构造任务表。",
            "frame['priority'] = frame['priority'].astype(priority)  # 转为有序Categorical。",
            "sorted_frame = frame.sort_values('priority')  # 按业务顺序而非字母顺序排序。",
            "codes = frame['priority'].cat.codes  # 获取内部整数编码。",
            "assert sorted_frame.iloc[0]['priority'] == 'low'  # 验证排序顺序。",
            "print(sorted_frame, '\\n', codes)  # 输出分类排序和编码。",
        ]
    elif family == 21:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "frame = pd.DataFrame({'order': [1, 2], 'items': [['pen', 'book'], ['desk']], 'quantities': [[2, 1], [1]]})  # 构造列表列。",
            "long = frame.explode(['items', 'quantities'], ignore_index=True)  # 同时展开等长列表列。",
            "long['quantities'] = long['quantities'].astype(int)  # explode后恢复数值类型。",
            "assert len(long) == 3  # 验证一行多值已展开。",
            "print(long)  # 输出规范化长表。",
        ]
    elif family == 22:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "scores = pd.Series([45, 58, 67, 72, 81, 89, 95, 99], name='score')  # 构造分数序列。",
            "fixed = pd.cut(scores, bins=[0, 60, 80, 100], labels=['low', 'medium', 'high'], right=False)  # 按业务边界等距分箱。",
            "quantiles = pd.qcut(scores, q=4, labels=False, duplicates='drop')  # 按样本分位数等频分箱。",
            "table = pd.DataFrame({'score': scores, 'fixed': fixed, 'quantile': quantiles})  # 汇总两种分箱。",
            "assert table['fixed'].notna().all()  # 验证边界覆盖所有值。",
            "print(table)  # 输出分箱结果。",
        ]
    elif family == 23:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "left = pd.Series([10, 20], index=['a', 'b'])  # 创建第一组带标签数据。",
            "right = pd.Series([1, 2], index=['b', 'c'])  # 创建错位索引数据。",
            "aligned_sum = left.add(right, fill_value=0)  # 显式按索引对齐并填充缺失。",
            "frame = pd.DataFrame({'value': [1, 2, 3]})  # 构造用于安全修改的表。",
            "selected = frame.loc[frame['value'].gt(1)].copy()  # 显式copy避免链式赋值歧义。",
            "selected.loc[:, 'double'] = selected['value'] * 2  # 使用loc安全赋值。",
            "assert aligned_sum.index.tolist() == ['a', 'b', 'c']  # 验证索引并集。",
            "print(aligned_sum, '\\n', selected)  # 输出对齐和副本结果。",
        ]
    else:
        code = [
            "import pandas as pd  # 导入 Pandas。",
            "index = pd.date_range('2025-01-01', periods=10, freq='12h')  # 创建半日频率时间索引。",
            "series = pd.Series(range(10), index=index, name='value')  # 构造时间序列。",
            "daily = series.resample('D').agg(['sum', 'mean', 'count'])  # 降采样到每日并多重聚合。",
            f"shifted = series.shift({variant}, freq='h')  # 移动时间索引而不移动值。",
            "assert daily['count'].sum() == len(series)  # 验证重采样未丢数据。",
            "print(daily, '\\n', shifted.head())  # 输出每日聚合和索引平移。",
        ]
    return title, task, code


def build_category(
    category: str,
    count: int,
    family_count: int,
    case_builder,
) -> None:
    """按知识点族和难度变式生成指定数量的题。"""
    variants_per_family = count // family_count
    number = 1
    for family in range(family_count):
        for variant in range(1, variants_per_family + 1):
            title, task, code = case_builder(family, variant)
            render_file(category, number, f"{title}_{LEVELS[variant - 1]}", task, code)
            number += 1
    if number - 1 != count:
        raise ValueError(f"{category} 题目数量错误：{number - 1}")


def main() -> None:
    clean_generated_folders()
    build_category("pandas", 100, 25, pandas_case)
    print("已生成 Pandas 100 题。")
    build_category("numpy", 100, 25, numpy_case)
    print("已生成 NumPy 100 题。")
    build_category("pytorch", 100, 25, pytorch_case)
    print("已生成 PyTorch 100 题。")
    build_category("python", 100, 25, python_case)
    print("已生成 Python 100 题。")
    build_category("ml", 100, 25, ml_case)
    print("已生成 ML 100 题。")
    build_category("sklearn", 50, 25, sklearn_case)
    print("已生成 scikit-learn 50 题。")


if __name__ == "__main__":
    main()

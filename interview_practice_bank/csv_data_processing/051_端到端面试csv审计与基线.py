"""
CSV数据处理练习 051：端到端面试CSV审计与基线

题目：分别读取train、val和inference分区，检查标签比例、缺失值、重复业务键和字段类型，再训练DummyClassifier建立后续模型必须超过的基线。

操作过程：
1. 定位模拟面试CSV。
2. 从CSV读取训练分区并解析日期。
3. 从CSV读取验证分区。
4. 从CSV读取推理分区。
5. 查看训练数据前五行以理解每列含义和原始值格式。
6. 检查训练标签是否不平衡。
7. 统计训练分区每列缺失数量。
8. 检查所有字段完全相同的重复行。
9. 检查业务主键重复以防样本重复。
10. 检查训练ID没有进入验证或推理分区。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas执行数据审计。
from sklearn.dummy import DummyClassifier  # 导入不学习特征规律的基线分类器。
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # 导入基线分类评估指标。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_churn_interview.csv'  # 定位模拟面试CSV。
frame = pd.read_csv(csv_path, parse_dates=['signup_date'])  # 从CSV读取全部数据并解析日期。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练分区。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出推理分区。
preview = train_frame.head()  # 查看训练数据前五行以理解每列含义和原始值格式。
target_rate = train_frame['churn'].value_counts(normalize=True).sort_index()  # 检查训练标签是否不平衡。
missing_by_column = train_frame.isna().sum().sort_values(ascending=False)  # 统计训练分区每列缺失数量。
duplicate_rows = train_frame.duplicated().sum()  # 检查所有字段完全相同的重复行。
duplicate_ids = train_frame['customer_id'].duplicated().sum()  # 检查业务主键重复以防样本重复。
split_ids_disjoint = set(train_frame['customer_id']).isdisjoint(val_frame['customer_id']) and set(train_frame['customer_id']).isdisjoint(inference_frame['customer_id'])  # 检查训练ID没有进入验证或推理分区。
feature_names = ['tenure_months']  # 基线模型只需要占位特征，因为most_frequent不会利用特征。
# drop变体：feature_frame = train_frame.drop(columns=['customer_id', 'age', 'monthly_charges', 'support_calls', 'weekly_usage_hours', 'region', 'contract_type', 'signup_date', 'churn', 'split'])  # 排除其他字段只保留tenure_months。
# iloc变体：feature_frame = train_frame.iloc[:, 2:3]  # 按位置选择tenure_months并保持二维DataFrame。
baseline = DummyClassifier(strategy='most_frequent')  # 创建始终预测训练集多数类别的最低基线。
baseline.fit(train_frame[feature_names], train_frame['churn'])  # 只用训练分区拟合多数类别。
val_predictions = baseline.predict(val_frame[feature_names])  # 在验证分区评估基线。
val_accuracy = accuracy_score(val_frame['churn'], val_predictions)  # 计算基线准确率。
val_precision = precision_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算基线正类precision。
val_recall = recall_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算基线正类recall。
val_f1 = f1_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算基线正类F1并处理没有正类预测的情况。
inference_predictions = baseline.predict(inference_frame[feature_names])  # 在独立推理分区生成基线结果。
assert duplicate_rows == 0 and duplicate_ids == 0 and split_ids_disjoint and len(inference_predictions) == len(inference_frame)  # 验证重复行、主键、分区和推理数量。
print('shape:', train_frame.shape, 'head:', preview, 'dtypes:', train_frame.dtypes, 'missing:', missing_by_column, 'target_rate:', target_rate, 'duplicate_rows:', duplicate_rows, 'duplicate_ids:', duplicate_ids, 'baseline_accuracy:', val_accuracy, 'baseline_precision:', val_precision, 'baseline_recall:', val_recall, 'baseline_f1:', val_f1, sep='\n')  # 输出数据审计和基线分类指标。
# 面试表达：先确认目标churn是二分类、正类含义、预测时点和错误成本，再决定指标与特征。
# 易错点：customer_id只是标识符，split是数据分区，两者都不应作为模型特征。

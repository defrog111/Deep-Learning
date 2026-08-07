"""
CSV数据处理练习 054：验证集选择分类阈值

题目：分别读取三个CSV分区，训练概率分类器，只在验证集遍历阈值并按F1选择最佳阈值，最后冻结阈值对推理分区分类。

操作过程：
1. 定位带固定分区的CSV。
2. 从CSV读取训练分区。
3. 从CSV读取阈值选择分区。
4. 从CSV读取最终推理分区。
5. 指定数值字段。
6. 指定类别字段。
7. 汇总安全模型特征。
8. 定义完整预处理。
9. 创建概率分类流水线。
10. 仅使用训练分区拟合。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import numpy as np  # 导入NumPy生成候选阈值。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.compose import ColumnTransformer  # 导入列预处理工具。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.linear_model import LogisticRegression  # 导入概率分类器。
from sklearn.metrics import f1_score, precision_score, recall_score  # 导入阈值相关分类指标。
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入编码和缩放工具。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_churn_interview.csv'  # 定位带固定分区的CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练分区。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出阈值选择分区。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出最终推理分区。
numeric = ['age', 'tenure_months', 'monthly_charges', 'support_calls', 'weekly_usage_hours']  # 指定数值字段。
categorical = ['region', 'contract_type']  # 指定类别字段。
features = numeric + categorical  # 汇总安全模型特征。
# drop变体：feature_frame = train_frame.drop(columns=['customer_id', 'signup_date', 'churn', 'split'])  # 按列名排除非模型字段。
# iloc变体：feature_frame = train_frame.iloc[:, 1:8]  # 按位置选择age到contract_type的连续特征区间。
preprocessor = ColumnTransformer([('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric), ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)])  # 定义完整预处理。
model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])  # 创建概率分类流水线。
model.fit(train_frame[features], train_frame['churn'])  # 仅使用训练分区拟合。
val_probabilities = model.predict_proba(val_frame[features])[:, 1]  # 得到验证正类概率。
thresholds = np.linspace(0.10, 0.90, 17)  # 定义不接触推理数据的候选阈值网格。
threshold_scores = {float(threshold): f1_score(val_frame['churn'], val_probabilities >= threshold, zero_division=0) for threshold in thresholds}  # 计算每个阈值的验证F1。
best_threshold = max(threshold_scores, key=threshold_scores.get)  # 选择验证F1最大的阈值。
val_predictions = (val_probabilities >= best_threshold).astype(int)  # 用最佳阈值生成验证类别。
val_precision = precision_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算所选阈值的precision。
val_recall = recall_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算所选阈值的recall。
inference_probabilities = model.predict_proba(inference_frame[features])[:, 1]  # 对推理分区输出概率。
inference_predictions = (inference_probabilities >= best_threshold).astype(int)  # 使用冻结阈值生成最终推理类别。
assert 0 < best_threshold < 1 and len(inference_predictions) == len(inference_frame)  # 验证阈值范围和推理数量。
print('best_threshold:', best_threshold, 'validation_f1:', threshold_scores[best_threshold], 'validation_precision:', val_precision, 'validation_recall:', val_recall, 'inference_prediction:', inference_predictions, sep='\n')  # 输出阈值选择和推理结果。
# 面试表达：阈值取决于误报与漏报成本；召回优先的安全任务通常会降低阈值。
# 易错点：阈值必须在验证集选择并在推理前冻结，不能用推理标签反向调参。

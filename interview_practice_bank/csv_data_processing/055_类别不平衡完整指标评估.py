"""
CSV数据处理练习 055：类别不平衡完整指标评估

题目：分别读取三个CSV分区，训练带class_weight的分类器，在验证集同时报告混淆矩阵、precision、recall、F1、ROC-AUC和PR-AUC，再完成推理。

操作过程：
1. 定位面试CSV。
2. 从CSV读取训练数据。
3. 从CSV读取验证数据。
4. 从CSV读取推理数据。
5. 定义数值字段。
6. 定义类别字段。
7. 排除ID、标签、日期和split。
8. 创建无泄漏预处理。
9. 用balanced让少数类获得更高损失权重。
10. 只在训练分区拟合模型。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.compose import ColumnTransformer  # 导入按列预处理工具。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.linear_model import LogisticRegression  # 导入支持类别权重的分类器。
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score  # 导入不平衡分类高频指标。
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入预处理组件。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_churn_interview.csv'  # 定位面试CSV。
train_frame = pd.read_csv(csv_path).query("split == 'train'").copy()  # 从CSV读取训练数据。
val_frame = pd.read_csv(csv_path).query("split == 'val'").copy()  # 从CSV读取验证数据。
inference_frame = pd.read_csv(csv_path).query("split == 'inference'").copy()  # 从CSV读取推理数据。
numeric = ['age', 'tenure_months', 'monthly_charges', 'support_calls', 'weekly_usage_hours']  # 定义数值字段。
categorical = ['region', 'contract_type']  # 定义类别字段。
features = numeric + categorical  # 排除ID、标签、日期和split。
preprocessor = ColumnTransformer([('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric), ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)])  # 创建无泄漏预处理。
model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])  # 用balanced让少数类获得更高损失权重。
model.fit(train_frame[features], train_frame['churn'])  # 只在训练分区拟合模型。
val_probabilities = model.predict_proba(val_frame[features])[:, 1]  # 获取验证集连续概率。
val_predictions = (val_probabilities >= 0.5).astype(int)  # 使用预先约定的0.5阈值分类。
matrix = confusion_matrix(val_frame['churn'], val_predictions, labels=[0, 1])  # 固定标签顺序构造2乘2混淆矩阵。
precision, recall, f1, support = precision_recall_fscore_support(val_frame['churn'], val_predictions, labels=[0, 1], zero_division=0)  # 分类别计算precision、recall、F1和样本量。
roc_auc = roc_auc_score(val_frame['churn'], val_probabilities)  # 衡量全部阈值下正负样本排序能力。
pr_auc = average_precision_score(val_frame['churn'], val_probabilities)  # 用PR-AUC重点观察正类表现。
inference_probabilities = model.predict_proba(inference_frame[features])[:, 1]  # 对推理分区输出概率。
assert matrix.shape == (2, 2) and matrix.sum() == len(val_frame) and len(inference_probabilities) == len(inference_frame)  # 验证指标覆盖全部验证样本且推理数量正确。
print('confusion_matrix:', matrix, 'precision:', precision, 'recall:', recall, 'f1:', f1, 'support:', support, 'roc_auc:', roc_auc, 'pr_auc:', pr_auc, sep='\n')  # 输出完整评估结果。
# 面试表达：类别极不平衡时accuracy可能虚高，应结合业务正类查看PR-AUC、recall、precision和混淆矩阵。
# 权衡：提高recall通常会增加false positive并降低precision，最终取舍必须结合错误成本。

"""
CSV数据处理练习 056：交叉验证与超参数搜索

题目：分别读取三个CSV分区，只在训练分区内部进行分层交叉验证和参数搜索，再用独立验证集报告泛化指标，最后对推理分区预测。

操作过程：
1. 定位带固定外部验证和推理分区的CSV。
2. 从CSV读取交叉验证训练池。
3. 从CSV读取独立验证分区。
4. 从CSV读取最终推理分区。
5. 定义数值特征。
6. 定义类别特征。
7. 汇总模型输入字段。
8. 把预处理放进待交叉验证流水线。
9. 创建每个fold都会重新拟合的完整流水线。
10. 定义正则强度和类别权重候选。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.compose import ColumnTransformer  # 导入列预处理工具。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.linear_model import LogisticRegression  # 导入待调参分类器。
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # 导入独立验证分类指标。
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # 导入网格搜索和分层交叉验证。
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入预处理组件。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_churn_interview.csv'  # 定位带固定外部验证和推理分区的CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出交叉验证训练池。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出独立验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出最终推理分区。
numeric = ['age', 'tenure_months', 'monthly_charges', 'support_calls', 'weekly_usage_hours']  # 定义数值特征。
categorical = ['region', 'contract_type']  # 定义类别特征。
features = numeric + categorical  # 汇总模型输入字段。
# drop变体：feature_frame = train_frame.drop(columns=['customer_id', 'signup_date', 'churn', 'split'])  # 按列名排除非模型字段。
# iloc变体：feature_frame = train_frame.iloc[:, 1:8]  # 按位置选择age到contract_type的连续特征区间。
preprocessor = ColumnTransformer([('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric), ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)])  # 把预处理放进待交叉验证流水线。
pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])  # 创建每个fold都会重新拟合的完整流水线。
parameter_grid = {'classifier__C': [0.1, 1.0, 10.0], 'classifier__class_weight': [None, 'balanced']}  # 定义正则强度和类别权重候选。
cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 创建保持每折标签比例的可复现划分。
search = GridSearchCV(pipeline, parameter_grid, scoring='roc_auc', cv=cross_validation, n_jobs=1, refit=True)  # 按ROC-AUC搜索并在全部训练池重拟合最佳组合。
search.fit(train_frame[features], train_frame['churn'])  # 只让网格搜索看到训练分区。
val_probabilities = search.best_estimator_.predict_proba(val_frame[features])[:, 1]  # 用独立验证分区检查最佳模型。
val_predictions = (val_probabilities >= 0.5).astype(int)  # 使用固定0.5阈值得到验证类别。
val_precision = precision_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算最佳参数模型正类precision。
val_recall = recall_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算最佳参数模型正类recall。
val_f1 = f1_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算最佳参数模型正类F1。
val_auc = roc_auc_score(val_frame['churn'], val_probabilities)  # 计算不参与参数选择的验证ROC-AUC。
inference_probabilities = search.best_estimator_.predict_proba(inference_frame[features])[:, 1]  # 对最终推理分区生成概率。
assert search.best_params_ and len(inference_probabilities) == len(inference_frame)  # 验证找到参数并为每个推理样本输出结果。
print('best_cv_auc:', search.best_score_, 'best_params:', search.best_params_, 'validation_auc:', val_auc, 'validation_precision:', val_precision, 'validation_recall:', val_recall, 'validation_f1:', val_f1, 'inference_probability:', inference_probabilities, sep='\n')  # 输出交叉验证、独立分类指标和推理结果。
# 面试表达：交叉验证用于更稳定地选择参数，外部val用于检查选择后的泛化，inference只用于最终应用。
# 易错点：预处理必须位于Pipeline内部，否则每个fold的验证部分会泄漏到填补、缩放和类别编码参数中。

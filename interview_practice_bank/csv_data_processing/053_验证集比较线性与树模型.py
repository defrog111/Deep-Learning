"""
CSV数据处理练习 053：验证集比较线性与树模型

题目：分别读取三个CSV分区，在相同预处理和相同验证集上比较LogisticRegression与RandomForestClassifier，根据ROC-AUC选择模型后再做推理。

操作过程：
1. 定位面试CSV。
2. 从CSV读取训练分区。
3. 从CSV读取模型选择分区。
4. 从CSV读取最终推理分区。
5. 定义数值字段。
6. 定义类别字段。
7. 创建无ID和标签的特征列表。
8. 构造两个模型共用的无泄漏预处理。
9. 定义可解释线性模型和非线性树模型。
10. 准备保存每个已拟合候选流水线。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.base import clone  # 导入估计器复制工具防止候选模型共享拟合状态。
from sklearn.compose import ColumnTransformer  # 导入按列预处理工具。
from sklearn.ensemble import RandomForestClassifier  # 导入能学习非线性关系的随机森林。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.linear_model import LogisticRegression  # 导入线性基线模型。
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # 导入模型比较和分类指标。
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入编码和缩放工具。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_churn_interview.csv'  # 定位面试CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练分区。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出模型选择分区。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出最终推理分区。
numeric = ['age', 'tenure_months', 'monthly_charges', 'support_calls', 'weekly_usage_hours']  # 定义数值字段。
categorical = ['region', 'contract_type']  # 定义类别字段。
features = numeric + categorical  # 创建无ID和标签的特征列表。
# drop变体：feature_frame = train_frame.drop(columns=['customer_id', 'signup_date', 'churn', 'split'])  # 按列名排除非模型字段。
# iloc变体：feature_frame = train_frame.iloc[:, 1:8]  # 按位置选择age到contract_type的连续特征区间。
preprocessor = ColumnTransformer([('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric), ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)])  # 构造两个模型共用的无泄漏预处理。
candidates = {'logistic': LogisticRegression(max_iter=1000, class_weight='balanced'), 'forest': RandomForestClassifier(n_estimators=150, max_depth=5, class_weight='balanced', random_state=42)}  # 定义可解释线性模型和非线性树模型。
fitted_models = {}  # 准备保存每个已拟合候选流水线。
validation_scores = {}  # 准备保存公平比较的验证ROC-AUC。
for name, classifier in candidates.items():  # 使用相同训练验证分区逐个评估候选模型。
    candidate = Pipeline([('preprocessor', clone(preprocessor)), ('classifier', classifier)])  # 为当前候选创建独立预处理流水线。
    candidate.fit(train_frame[features], train_frame['churn'])  # 只使用训练分区拟合当前候选。
    validation_scores[name] = roc_auc_score(val_frame['churn'], candidate.predict_proba(val_frame[features])[:, 1])  # 在同一验证集计算ROC-AUC。
    fitted_models[name] = candidate  # 保存已拟合模型供选择后推理。
best_name = max(validation_scores, key=validation_scores.get)  # 依据预先声明的验证指标选择模型。
best_model = fitted_models[best_name]  # 取得验证表现最佳的完整流水线。
val_probabilities = best_model.predict_proba(val_frame[features])[:, 1]  # 获取最佳模型的验证正类概率。
val_predictions = (val_probabilities >= 0.5).astype(int)  # 使用固定0.5阈值得到验证类别。
val_precision = precision_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算最佳模型正类precision。
val_recall = recall_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算最佳模型正类recall。
val_f1 = f1_score(val_frame['churn'], val_predictions, zero_division=0)  # 计算最佳模型正类F1。
inference_probabilities = best_model.predict_proba(inference_frame[features])[:, 1]  # 只在选择结束后查看推理分区预测。
assert set(validation_scores) == set(candidates) and len(inference_probabilities) == len(inference_frame)  # 验证候选均参与比较且推理数量正确。
print('validation_auc:', validation_scores, 'selected_model:', best_name, 'validation_precision:', val_precision, 'validation_recall:', val_recall, 'validation_f1:', val_f1, 'inference_probability:', inference_probabilities, sep='\n')  # 输出模型比较、分类指标和推理结果。
# 面试表达：树模型能捕捉非线性和交互，Logistic更可解释；小数据上应优先控制复杂度并报告方差风险。
# 易错点：不能反复查看inference结果后选模型，否则推理集事实上变成了验证集。

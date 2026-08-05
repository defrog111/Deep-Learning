"""
CSV数据处理练习 052：混合特征预处理与Logistic流水线

题目：分别读取三个CSV分区，用ColumnTransformer对数值缺失值和类别缺失值分别处理，再训练LogisticRegression并输出验证ROC-AUC和推理概率。

操作过程：
1. 定位含混合字段和缺失值的CSV。
2. 从CSV读取训练数据。
3. 从CSV读取验证数据。
4. 从CSV读取推理数据。
5. 定义数值特征。
6. 定义需要独热编码的类别特征。
7. 汇总最终模型输入列并排除ID、日期、标签和split。
8. 用训练中位数填补数值缺失再标准化。
9. 填补类别缺失并安全处理推理新类别。
10. 按字段类型组合预处理分支。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
from sklearn.compose import ColumnTransformer  # 导入按列类型分支处理工具。
from sklearn.impute import SimpleImputer  # 导入缺失值填补器。
from sklearn.linear_model import LogisticRegression  # 导入可解释的二分类模型。
from sklearn.metrics import roc_auc_score  # 导入不依赖固定阈值的排序指标。
from sklearn.pipeline import Pipeline  # 导入防止数据泄漏的流水线。
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 导入类别编码和数值标准化工具。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_churn_interview.csv'  # 定位含混合字段和缺失值的CSV。
train_frame = pd.read_csv(csv_path).query("split == 'train'").copy()  # 从CSV读取训练数据。
val_frame = pd.read_csv(csv_path).query("split == 'val'").copy()  # 从CSV读取验证数据。
inference_frame = pd.read_csv(csv_path).query("split == 'inference'").copy()  # 从CSV读取推理数据。
numeric_features = ['age', 'tenure_months', 'monthly_charges', 'support_calls', 'weekly_usage_hours']  # 定义数值特征。
categorical_features = ['region', 'contract_type']  # 定义需要独热编码的类别特征。
features = numeric_features + categorical_features  # 汇总最终模型输入列并排除ID、日期、标签和split。
numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])  # 用训练中位数填补数值缺失再标准化。
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # 填补类别缺失并安全处理推理新类别。
preprocessor = ColumnTransformer([('numeric', numeric_pipeline, numeric_features), ('categorical', categorical_pipeline, categorical_features)])  # 按字段类型组合预处理分支。
model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])  # 把全部预处理与模型封装成单一流水线。
model.fit(train_frame[features], train_frame['churn'])  # 只在训练分区拟合填补值、缩放参数、类别词表和模型参数。
val_probabilities = model.predict_proba(val_frame[features])[:, 1]  # 在验证分区输出正类概率。
val_auc = roc_auc_score(val_frame['churn'], val_probabilities)  # 用验证数据评估ROC-AUC。
inference_probabilities = model.predict_proba(inference_frame[features])[:, 1]  # 对独立推理分区输出正类概率。
inference_result = inference_frame[['customer_id']].assign(churn_probability=inference_probabilities, churn_prediction=(inference_probabilities >= 0.5).astype(int))  # 把推理结果与业务ID重新对齐。
assert inference_result['customer_id'].is_unique and inference_result['churn_probability'].between(0, 1).all()  # 验证一客一行且概率有效。
print('validation_auc:', val_auc, 'inference_result:', inference_result, sep='\n')  # 输出验证指标和可交付推理表。
# 面试表达：Logistic Regression是强基线，训练快、概率和系数易解释，适合先验证端到端流程。
# 易错点：不能在全数据上先fillna、get_dummies或StandardScaler，再划分数据，否则验证信息会泄漏。

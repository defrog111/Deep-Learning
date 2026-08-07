"""
CSV数据处理练习 063：Logistic Regression表格二分类详解

题目：分别从CSV读取train、val和inference客户数据，用中位数填补、标准化和Logistic Regression完成二分类，并解释系数、odds ratio、概率和阈值。

操作过程：
1. 定位客户流失CSV。
2. 从CSV读取训练分区。
3. 从CSV读取验证分区。
4. 从CSV读取推理分区。
5. 选择数值特征并排除ID、标签和split。
6. 串联填补、标准化和带L2正则的逻辑回归。
7. 只用训练分区拟合所有流水线步骤。
8. 输出验证样本属于正类的概率。
9. 使用0.5阈值生成验证类别。
10. 使用连续概率计算验证ROC-AUC。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import numpy as np  # 导入NumPy计算odds ratio。
import pandas as pd  # 导入Pandas读取CSV和展示系数。
from sklearn.impute import SimpleImputer  # 导入数值缺失值填补器。
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归分类器。
from sklearn.metrics import classification_report, roc_auc_score  # 导入分类报告和概率排序指标。
from sklearn.pipeline import Pipeline  # 导入无泄漏流水线。
from sklearn.preprocessing import StandardScaler  # 导入数值标准化工具。
csv_path = Path(__file__).parents[1] / 'data' / 'customer_churn_interview.csv'  # 定位客户流失CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练分区。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出推理分区。
feature_names = ['age', 'tenure_months', 'monthly_charges', 'support_calls', 'weekly_usage_hours']  # 选择数值特征并排除ID、标签和split。
# drop变体：feature_frame = train_frame.drop(columns=['customer_id', 'region', 'contract_type', 'signup_date', 'churn', 'split'])  # 按列名排除非数值模型字段。
# iloc变体：feature_frame = train_frame.iloc[:, 1:6]  # 按位置选择age到weekly_usage_hours五个数值特征。
model = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('classifier', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000))])  # 串联填补、标准化和带L2正则的逻辑回归。
model.fit(train_frame[feature_names], train_frame['churn'])  # 只用训练分区拟合所有流水线步骤。
val_probabilities = model.predict_proba(val_frame[feature_names])[:, 1]  # 输出验证样本属于正类的概率。
val_predictions = (val_probabilities >= 0.5).astype(int)  # 使用0.5阈值生成验证类别。
val_auc = roc_auc_score(val_frame['churn'], val_probabilities)  # 使用连续概率计算验证ROC-AUC。
report = classification_report(val_frame['churn'], val_predictions, output_dict=True, zero_division=0)  # 汇总precision、recall和F1。
classifier = model.named_steps['classifier']  # 从已拟合流水线取得逻辑回归步骤。
coefficients = pd.Series(classifier.coef_[0], index=feature_names, name='standardized_log_odds_coefficient')  # 将标准化特征系数与列名对齐。
odds_ratios = np.exp(coefficients).rename('odds_ratio_per_one_standard_deviation')  # 指数化系数得到特征增加一个标准差对应的odds倍数。
inference_probabilities = model.predict_proba(inference_frame[feature_names])[:, 1]  # 对独立推理分区输出正类概率。
inference_result = inference_frame[['customer_id']].assign(churn_probability=inference_probabilities, churn_prediction=(inference_probabilities >= 0.5).astype(int))  # 将概率和类别与客户ID对齐。
assert inference_result['churn_probability'].between(0, 1).all() and coefficients.index.tolist() == feature_names  # 验证概率范围和系数对应关系。
print('validation_auc:', val_auc, 'classification_report:', report, 'coefficients:', coefficients, 'odds_ratios:', odds_ratios, 'inference:', inference_result, sep='\n')  # 输出模型评估、解释和推理结果。
# Logistic回归建模的是log(p/(1-p))的线性组合；正系数提高正类log-odds，负系数降低它。
# 因为特征已标准化，这里的odds ratio表示增加一个训练标准差的影响，而不是增加一个原始单位。
# 高频参数：C越小正则越强；penalty控制正则类型；class_weight可提高少数类损失权重。

"""
题目 012：Pipeline防止数据泄漏_变式

要求：完成“Pipeline防止数据泄漏”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 综合题统一导入NumPy用于shape、数值和标签检查。
3. 并行生成原始与交互特征。

完成标准：
- 验证FeatureUnion横向拼接。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入Iris数据。
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归。
from sklearn.model_selection import cross_val_score  # 导入交叉验证评分。
from sklearn.pipeline import Pipeline  # 导入流水线。
from sklearn.preprocessing import StandardScaler  # 导入标准化器。
features, labels = load_iris(return_X_y=True)  # 加载数据。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures  # 导入自定义函数转换和多项式特征。
from sklearn.pipeline import FeatureUnion  # 导入并行特征组合器。
positive_features = np.abs(features[:, :2]) + 1; union = FeatureUnion([('identity', FunctionTransformer(validate=True)), ('polynomial', PolynomialFeatures(degree=2, include_bias=False))]); union_result = union.fit_transform(positive_features)  # 并行生成原始与交互特征。
assert union_result.shape[0] == len(features) and union_result.shape[1] > positive_features.shape[1]  # 验证FeatureUnion横向拼接。

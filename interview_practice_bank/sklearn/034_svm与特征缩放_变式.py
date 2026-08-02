"""
题目 034：SVM与特征缩放_变式

要求：完成“SVM与特征缩放”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. SVM对尺度敏感。
2. 加载数据。
3. 标准化后训练RBF SVM。
4. 综合题统一导入NumPy用于shape、数值和标签检查。
5. LinearSVC适合高维线性问题但不提供predict_proba。
6. 对比LinearSVC决策分数与启用probability的SVC接口。

完成标准：
- 对比LinearSVC决策分数与启用probability的SVC接口。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_breast_cancer  # 导入分类数据。
from sklearn.model_selection import cross_val_score  # 导入交叉验证。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # SVM对尺度敏感。
from sklearn.svm import SVC  # 导入核SVM。
features, labels = load_breast_cancer(return_X_y=True)  # 加载数据。
model = make_pipeline(StandardScaler(), SVC(C=2, kernel='rbf', probability=True, random_state=42))  # 标准化后训练RBF SVM。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.svm import LinearSVC  # 导入高维线性SVM。
stable_svm_features = features.astype(np.float32); linear_svm = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual='auto', random_state=42)).fit(stable_svm_features, labels); linear_coefficients = linear_svm.named_steps['linearsvc'].coef_  # LinearSVC适合高维线性问题但不提供predict_proba。
assert linear_coefficients.shape[1] == features.shape[1] and not hasattr(linear_svm, 'predict_proba') and hasattr(model, 'predict_proba')  # 对比LinearSVC决策分数与启用probability的SVC接口。

"""
题目 024：LogisticRegression分类_变式

要求：完成“LogisticRegression分类”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 分层切分。
3. 训练多分类逻辑回归。
4. 概率最大类别即预测。
5. 综合题统一导入NumPy用于shape、数值和标签检查。
6. partial_fit分批训练，再用训练集内部独立校准集拟合概率映射。

完成标准：
- 验证准确率。
- 验证在线模型与校准接口。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入Iris数据。
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归。
from sklearn.metrics import accuracy_score, log_loss  # 导入分类指标。
from sklearn.model_selection import train_test_split  # 导入切分函数。
features, labels = load_iris(return_X_y=True)  # 加载数据。
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.25, random_state=42, stratify=labels)  # 分层切分。
model = LogisticRegression(max_iter=500).fit(train_x, train_y)  # 训练多分类逻辑回归。
probabilities = model.predict_proba(test_x)  # 输出每类概率。
predictions = probabilities.argmax(axis=1)  # 概率最大类别即预测。
assert accuracy_score(test_y, predictions) > 0.85  # 验证准确率。
print(accuracy_score(test_y, predictions), log_loss(test_y, probabilities))  # 输出硬指标和概率指标。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.calibration import CalibratedClassifierCV  # 导入概率校准器。
from sklearn.linear_model import SGDClassifier  # 导入支持在线学习的线性分类器。
from sklearn.frozen import FrozenEstimator  # 导入新版预训练估计器冻结包装器。
online_train_x, calibration_x, online_train_y, calibration_y = train_test_split(train_x, train_y, test_size=0.25, stratify=train_y, random_state=7); online = SGDClassifier(loss='log_loss', random_state=42); classes = np.unique(labels); midpoint = len(online_train_x) // 2; online.partial_fit(online_train_x[:midpoint].astype(np.float32), online_train_y[:midpoint], classes=classes); online.partial_fit(online_train_x[midpoint:].astype(np.float32), online_train_y[midpoint:]); calibrated = CalibratedClassifierCV(FrozenEstimator(online), method='sigmoid').fit(calibration_x.astype(np.float32), calibration_y)  # partial_fit分批训练，再用训练集内部独立校准集拟合概率映射。
assert calibrated.predict_proba(test_x[:2].astype(np.float32)).shape == (2, len(classes))  # 验证在线模型与校准接口。

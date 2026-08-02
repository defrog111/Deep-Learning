"""
题目 040：classification_report与混淆矩阵_变式

要求：完成“classification_report与混淆矩阵”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 导入分类数据。
2. 导入分类器。
3. 导入综合指标。
4. 导入无泄漏交叉验证预测。
5. 综合题统一导入NumPy用于shape、数值和标签检查。
6. 导入多标签和分项指标。
7. 多标签不能直接套普通单标签混淆矩阵。
8. 验证多标签指标shape。

完成标准：
- 验证多标签指标shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.ensemble import RandomForestClassifier  # 导入分类器。
from sklearn.metrics import classification_report, confusion_matrix  # 导入综合指标。
from sklearn.model_selection import cross_val_predict  # 导入无泄漏交叉验证预测。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support  # 导入多标签和分项指标。
multi_true = np.array([[1, 0, 1], [0, 1, 0]]); multi_pred = np.array([[1, 1, 0], [0, 1, 0]]); per_label_confusion = multilabel_confusion_matrix(multi_true, multi_pred); macro_parts = precision_recall_fscore_support(multi_true, multi_pred, average='macro', zero_division=0)  # 多标签不能直接套普通单标签混淆矩阵。
assert per_label_confusion.shape == (3, 2, 2) and len(macro_parts) == 4  # 验证多标签指标shape。

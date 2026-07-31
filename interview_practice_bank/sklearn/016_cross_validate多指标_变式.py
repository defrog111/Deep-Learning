"""
题目 016：cross_validate多指标_变式

要求：完成“cross_validate多指标”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 建立可复现的分类模型。
3. 同时评估多个指标。

完成标准：
- 验证AUC合理。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_breast_cancer  # 导入二分类数据。
from sklearn.ensemble import RandomForestClassifier  # 导入非线性分类器。
from sklearn.model_selection import cross_validate  # 导入多指标交叉验证。
features, labels = load_breast_cancer(return_X_y=True)  # 加载数据。
model = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42)  # 建立可复现的分类模型。
result = cross_validate(model, features, labels, cv=3, scoring=['accuracy', 'roc_auc'], return_train_score=True)  # 同时评估多个指标。
assert result['test_roc_auc'].mean() > 0.9  # 验证AUC合理。
print({key: value.mean() for key, value in result.items() if key.startswith('test_')})  # 输出测试指标均值。

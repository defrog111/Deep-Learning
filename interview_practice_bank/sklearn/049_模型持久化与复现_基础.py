"""
题目 049：模型持久化与复现_基础

要求：完成“模型持久化与复现”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

from pathlib import Path  # 导入路径工具。
import tempfile  # 导入临时目录工具。
import joblib  # 导入scikit-learn推荐持久化工具。
import numpy as np  # 导入 NumPy。
from sklearn.datasets import load_iris  # 导入数据。
from sklearn.pipeline import make_pipeline  # 导入Pipeline。
from sklearn.preprocessing import StandardScaler  # 导入标准化器。
from sklearn.linear_model import LogisticRegression  # 导入分类器。
features, labels = load_iris(return_X_y=True)  # 加载数据。
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=41)).fit(features, labels)  # 固定随机种子训练完整Pipeline。
expected = model.predict(features[:5])  # 保存前取得基准预测。
with tempfile.TemporaryDirectory() as folder:  # 创建自动清理目录。
    path = Path(folder) / 'pipeline.joblib'  # 定义模型文件路径。
    joblib.dump(model, path)  # 持久化预处理和模型。
    restored = joblib.load(path)  # 只加载可信来源的模型文件。
    actual = restored.predict(features[:5])  # 恢复后重新预测。
assert np.array_equal(expected, actual)  # 验证持久化前后预测一致。
print(actual)  # 输出恢复模型预测。

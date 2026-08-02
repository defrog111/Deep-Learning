"""
题目 050：模型持久化与复现_变式

要求：完成“模型持久化与复现”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 综合题统一导入NumPy用于shape、数值和标签检查。
3. 分批训练时首次调用必须传完整classes。
4. 使用自动清理目录验证在线模型持久化。
    online_path = Path(online_folder) / 'online.joblib'  # 定义模型路径。
    joblib.dump(online_model, online_path)  # 保存已学习统计量。
    online_restored = joblib.load(online_path)  # 仅加载可信文件。
    restored_online_predictions = online_restored.predict(features[:5])  # 恢复后预测。

完成标准：
- 验证partial_fit模型持久化一致。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.naive_bayes import GaussianNB  # 导入支持partial_fit的朴素贝叶斯。
online_model = GaussianNB(); classes = np.unique(labels); online_model.partial_fit(features[:75], labels[:75], classes=classes); online_model.partial_fit(features[75:], labels[75:]); online_predictions = online_model.predict(features[:5])  # 分批训练时首次调用必须传完整classes。
with tempfile.TemporaryDirectory() as online_folder:  # 使用自动清理目录验证在线模型持久化。
    online_path = Path(online_folder) / 'online.joblib'  # 定义模型路径。
    joblib.dump(online_model, online_path)  # 保存已学习统计量。
    online_restored = joblib.load(online_path)  # 仅加载可信文件。
    restored_online_predictions = online_restored.predict(features[:5])  # 恢复后预测。
assert np.array_equal(online_predictions, restored_online_predictions)  # 验证partial_fit模型持久化一致。

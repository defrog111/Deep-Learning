"""
题目 028：DecisionTree与过拟合控制_变式

要求：完成“DecisionTree与过拟合控制”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 分层切分。
3. 综合题统一导入NumPy用于shape、数值和标签检查。
4. cost-complexity pruning用alpha折叠弱分支。

完成标准：
- 验证后剪枝降低复杂度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.model_selection import train_test_split  # 导入切分函数。
from sklearn.tree import DecisionTreeClassifier, export_text  # 导入决策树和文本导出。
features, labels = load_iris(return_X_y=True)  # 加载数据。
train_x, test_x, train_y, test_y = train_test_split(features, labels, stratify=labels, random_state=42)  # 分层切分。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.tree import DecisionTreeClassifier  # 导入决策树。
unpruned = DecisionTreeClassifier(random_state=42).fit(train_x, train_y); pruning_path = unpruned.cost_complexity_pruning_path(train_x, train_y); pruned = DecisionTreeClassifier(ccp_alpha=float(pruning_path.ccp_alphas[-2]), random_state=42).fit(train_x, train_y)  # cost-complexity pruning用alpha折叠弱分支。
assert pruned.get_n_leaves() <= unpruned.get_n_leaves()  # 验证后剪枝降低复杂度。

# CSV 数据处理专项练习

本目录包含 76 个独立、可运行的例子。每个例子都从 `pd.read_csv(...)` 开始，
不会用代码临时构造 DataFrame 来绕过 CSV 读取。

建议学习顺序：

1. `001–008`：读取参数、dtype、日期、缺失标记、分块和文本。
2. `009–015`：重复值、文本/金额/日期清洗、填补、插值和异常值。
3. `016–021`：筛选、特征工程、排序、GroupBy 聚合与 transform/filter。
4. `022–030`：merge、反连接、关系验证、concat、pivot、melt、crosstab、explode、分箱。
5. `031–036`：resample、rolling、shift、正则文本、Categorical 和索引对齐。
6. `037–042`：NumPy、sklearn、训练测试切分、PyTorch、CSV导出和端到端流水线。
7. `043`：扫描文件夹中的全部CSV，记录来源、审计结构并兼容不同列完成合并。
8. `044–046`：读取同一房价CSV，分别使用NumPy、PyTorch和scikit-learn完成线性回归。
9. `047–050`：PyTorch两层FFN分类，覆盖二分类、多分类、多标签、加权损失和DataLoader。
10. `051–056`：模型开发面试端到端流程，覆盖审计、基线、预处理、模型选择、阈值、指标和交叉验证。
11. `057–060`：原生TransformerEncoder和两层GRU的递归类别预测与递归连续值预测。
12. `061`：完整Encoder-Decoder Transformer、teacher forcing、causal mask和四步自回归预测。
13. `062`：文本description与数值字段的多模态Transformer分类和特征融合。
14. `063–065`：表格Logistic解释，以及高维稀疏文本的TF-IDF配合Logistic和Linear SVM。
15. `066`：Hugging Face预训练BERT文本表示与PyTorch数值分支的实用多模态融合。
16. `067–069`：读取数值CSV完成KMeans聚类、线性SVM分类和随机森林分类。
17. `070–075`：在含姓名、地址、邮编、国家区号、电话、时间、数量等混合字段的客户表上比较Logistic、Random Forest、SVM、Gradient Boosting、KNN和GaussianNB。
18. `076`：选择模型前的全面CSV审计，覆盖字段角色、dtype、类别取值、NA、分布、异常值、重复、高基数、标签平衡和泄漏风险。

运行单题：

```bash
python csv_data_processing/011_货币字符串转数值.py
```

验证并运行全部题目：

```bash
python validate_csv_examples.py
```

使用的数据位于上级 `data/` 目录，Iris 例子复用项目 `transformer_learning/data/iris.csv`。

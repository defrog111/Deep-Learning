# CSV 数据处理专项练习

本目录包含 42 个独立、可运行的例子。每个例子都从 `pd.read_csv(...)` 开始，
不会用代码临时构造 DataFrame 来绕过 CSV 读取。

建议学习顺序：

1. `001–008`：读取参数、dtype、日期、缺失标记、分块和文本。
2. `009–015`：重复值、文本/金额/日期清洗、填补、插值和异常值。
3. `016–021`：筛选、特征工程、排序、GroupBy 聚合与 transform/filter。
4. `022–030`：merge、反连接、关系验证、concat、pivot、melt、crosstab、explode、分箱。
5. `031–036`：resample、rolling、shift、正则文本、Categorical 和索引对齐。
6. `037–042`：NumPy、sklearn、训练测试切分、PyTorch、CSV导出和端到端流水线。

运行单题：

```bash
python csv_data_processing/011_货币字符串转数值.py
```

验证并运行全部题目：

```bash
python validate_csv_examples.py
```

使用的数据位于上级 `data/` 目录，Iris 例子复用项目 `transformer_learning/data/iris.csv`。

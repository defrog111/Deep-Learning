# CNN Multi-Label 分类

- 标签：核心
- Branch：`cnn-core-multilabel-classification`

## 题目
用简单 CNN 做多标签分类，演示 BCEWithLogitsLoss、sigmoid 输出和多标签常见指标。

## 代码要求
1. `main.py` 里给最小可运行例子。
2. 每一句代码都带中文注释。
3. 注释里尽量说明 shape 和变量作用。
4. 这道题对应的主流 metric 也要给出来。

## 常用 Metric
- Micro Precision
- Micro Recall
- Micro F1
- Hamming Loss
- Exact Match Ratio

## 面试讲法
1. 先讲 multi-class 和 multi-label 的区别。
2. 再讲为什么输出层不做 softmax，而是对每个标签独立做 sigmoid。
3. 再讲 BCEWithLogitsLoss 和常见多标签 metric。
4. 最后讲阈值 0.5 只是默认值，真实项目可以调。

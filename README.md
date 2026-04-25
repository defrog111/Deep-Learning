# 多模态 Transformer 分类

- 标签：核心
- Branch：multimodal-core-transformer-fusion-classification

## 题目
输入数值特征、类别特征、图像和文本，用 Transformer 融合后做分类。

## 你可以怎么讲
1. 先讲输入张量 shape。
2. 再讲模型每一层把 shape 变成什么样。
3. 再讲 loss 怎么定义。
4. 最后讲这个模型适合什么任务，和常见替代方案的区别。

## 文件说明
- main.py：最小可运行示例，逐句中文注释。
- 代码重点是帮助面试讲清楚结构和数据流，不是追求完整训练工程。

## å¸¸ç”¨ Metric
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Ablation by modality

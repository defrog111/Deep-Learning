# CNN 常见 Finetuning 方法

- 标签：核心
- Branch：`finetune-core-cnn-finetuning-methods`

## 题目
用一个简单 CNN 例子演示 linear probing、partial unfreeze、full finetuning、differential learning rates 和 LoRA finetuning。

## 代码要求
1. `main.py` 里给最小可运行例子。
2. 每一句代码都带中文注释。
3. 注释里尽量说明 shape 和变量作用。
4. 这道题对应的主流 metric 也要给出来。

## 常用 Metric
- Accuracy
- Precision
- Recall
- F1
- Trainable Parameter Count
- Ablation by Finetuning Method

## 面试讲法
1. 先讲 backbone 和 classifier head 的分工。
2. 再讲为什么 linear probing 先冻结 backbone。
3. 再讲 partial unfreeze、full finetuning 和 differential learning rates 的差异。
4. 最后讲 LoRA finetuning 适合什么场景。

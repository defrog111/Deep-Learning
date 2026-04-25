# CNN 分类与模型压缩

- 标签：核心
- Branch：`cnn-core-model-compression-classification`

## 题目
用一个简单 CNN 分类例子，把 LoRA、量化、剪枝这几类常见 model compression 方法都串进去。

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
- Model Size
- Sparsity
- Latency Proxy

## 面试讲法
1. 先讲原始 CNN 分类流程。
2. 再讲 LoRA、量化、剪枝分别压的是什么。
3. 再讲哪些 metric 用来看效果，哪些指标用来看压缩收益。
4. 最后讲三种方法什么时候适合一起用。
